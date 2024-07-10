#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

set -o pipefail
ulimit -n 65535
sudo sysctl -w net.ipv4.ip_local_reserved_ports=41000
export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export FI_EFA_FORK_SAFE=1

if [ -z "${SLURM_NNODES}" ]
then
    echo "Single-node, non-SLURM runs"
    HOSTS=(localhost)
    NODEID=0
    NTASKS=1
    export NEMO_EXPM_VERSION=$(date "+%Y-%m-%d_%H-%M-%S")
    DATA_PATH=~/tao_codebase/examples_datasets
    SCRIPT_PATH=~/tao_codebase/Trn-ext-nemo-neox
else
    echo "SLURM runs, single or multi-node"
    IPS=""
    for h in $(scontrol show hostname); do
        # IPS="$IPS $(nslookup $h  | awk '/^Address: / { print $2 }')";
        IPS="$IPS $(ping -c 1 $h | grep PING | awk '{print $3}' | tr -d '()')";
    done
    HOSTS=(${IPS//\ / })
    NODEID=$SLURM_NODEID
    NTASKS=$SLURM_NTASKS
    export NEMO_EXPM_VERSION=$SLURM_JOB_ID
    # DATA_PATH=/fsx/fsx/gaurav/tao_codebase/examples_datasets
    # SCRIPT_PATH=/fsx/fsx/gaurav/tao_codebase/Trn-ext-nemo-neox
    DATA_PATH=/fsx/users/tao/examples_datasets
    SCRIPT_PATH=/fsx/users/tao/Trn-ext-nemo-neox
fi

# export HYDRA_FULL_ERROR=1
export PROCESSES_PER_NODE=8
export MASTER_ADDR=${HOSTS[0]}
export MASTER_PORT=41000
echo "IPS $IPS HOSTS $HOSTS and MASTER_ADDR $MASTER_ADDR NTASKS $NTASKS NODEID $NODEID"
DISTRIBUTED_ARGS="--nproc_per_node $PROCESSES_PER_NODE --nnodes $NTASKS --node_rank $NODEID --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
echo $DISTRIBUTED_ARGS

# export MALLOC_ARENA_MAX=128
# Make sure vocab file and data are under same folder

DATA_SET=gpt2/my-gpt2_text_document
EXP_PATH=$SCRIPT_PATH/log_gpt_6p7B_ckpt/full_bf16_mckahan_$(date "+%y-%m-%d-%H-%M")-gpt-neox-6p7B-nemo
DOCKER_PATH="${EXP_PATH/"$SCRIPT_PATH"/"/workspace/gpt_neox"}"
CHECKPOINT_PATH=$DOCKER_PATH/chkpt
TB_DIR=$DOCKER_PATH/tensorboard
TB_DIR="${TB_DIR//\"}"
if [ $NODEID -eq 0 ]
then
	echo $EXP_PATH
    mkdir -p $EXP_PATH
    cp "$0" "$EXP_PATH"/
fi

REPO=gauravaz
TAG=nemo-23.11-efa
REGION=us-west-2

# Grab current AWS account from sts
AWS_ACCOUNT=`aws sts get-caller-identity --region ${REGION} --endpoint-url https://sts.${REGION}.amazonaws.com --query Account --output text`
# Log in to ECR
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com
IMAGE_URI=${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:${TAG}
DOCKER_NAME=nvidia-docker-nemo
docker rm $DOCKER_NAME
docker stop $DOCKER_NAME
docker pull $IMAGE_URI
docker images
docker run  --name $DOCKER_NAME --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -d \
    --net=host --uts=host \
    -e NCCL_SOCKET_IFNAME=ens5 \
    -e NCCL_SOCKET_IFNAME="^lo,docker" \
    -e RDMAV_FORK_SAFE=1 \
    -e GPU_NUM_DEVICES=8 \
    -e FI_EFA_USE_DEVICE_RDMA=1  \
    -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
    -e PYTHONPATH=/workspace/gpt_neox \
    --security-opt seccomp=unconfined \
    --privileged \
    --shm-size=561g \
    -v $DATA_PATH:/workspace/examples_datasets \
    -v $SCRIPT_PATH:/workspace/gpt_neox \
    $IMAGE_URI

# docker exec -it $DOCKER_NAME /bin/bash

echo ">>> Installing packages ..."
docker exec $DOCKER_NAME pip install transformers==4.31.0
docker exec $DOCKER_NAME pip install ujson
docker exec $DOCKER_NAME pip install numexpr==2.7.3

echo ">>> Package installed."
echo ">>> Running script ..."
export CUDA_DEVICE_MAX_CONNECTIONS=1

TP=8
PP=1
UBS=4
GBS=256
SCRIPT_PATH=/workspace/gpt_neox
SEQ_LEN=2048
HS=4096
N_LAYERS=32
N_AH=32
# to be used with model.activations_checkpoint_num_layers=$AL
AL=1
ACT_CHK=null
precision='bf16'
optim_name='adamw_base'
AMP_O2=False
FULL_BF16=False
kahan='mckahan'
SEQ_PAR=False


TRAIN_ITERS=20000
VAL_ITERS=250
EXP_NAME=log_gpt_6p7B
EXP_NAME+=_$precision

export FFN_HS=$(($HS*4))
echo "SEQ_LEN=$SEQ_LEN, HS=$HS, FFN_HS=$FFN_HS TP=$TP PP=$PP N_LAYERS=$N_LAYERS N_AH=$N_AH GBS=$GBS UBS=$UBS"

if [ $NODEID -eq 0 ]
then
	echo $SCRIPT_PATH_$EXP_PATH_$TP_$BS_$AL_$VERSION |& tee $EXP_PATH/run_config
fi

GPT_ARGS="
    --config-path=conf \
    --config-name=megatron_gpt_config \
    model.make_vocab_size_divisible_by=$TP \
    model.use_flash_attention=True \
    model.transformer_engine=False \
    trainer.devices=$PROCESSES_PER_NODE \
    trainer.num_nodes=$NTASKS \
    trainer.max_epochs=null \
    trainer.max_steps=$TRAIN_ITERS\
    trainer.val_check_interval=$VAL_ITERS \
    trainer.log_every_n_steps=1 \
    trainer.limit_val_batches=1 \
    trainer.limit_test_batches=1 \
    trainer.accumulate_grad_batches=1 \
    trainer.precision=$precision \
    model.micro_batch_size=$UBS \
    model.global_batch_size=$GBS \
    model.tensor_model_parallel_size=$TP \
    model.pipeline_model_parallel_size=$PP \
    model.max_position_embeddings=$SEQ_LEN \
    model.share_embeddings_and_output_weights=False \
    model.encoder_seq_length=$SEQ_LEN \
    model.hidden_size=$HS \
    model.ffn_hidden_size=$FFN_HS \
    model.num_layers=$N_LAYERS \
    model.num_attention_heads=$N_AH \
    model.init_method_std=0.021 \
    model.hidden_dropout=0 \
    model.layernorm_epsilon=1e-5 \
    model.tokenizer.vocab_file=/workspace/examples_datasets/gpt2/gpt2-vocab.json \
    model.tokenizer.merge_file=/workspace/examples_datasets/gpt2/gpt2-merges.txt \
    model.data.data_prefix=[1.0,/workspace/examples_datasets/gpt2/my-gpt2_text_document] \
    model.data.num_workers=4 \
    model.data.seq_length=$SEQ_LEN \
    model.position_embedding_type=rope \
    model.rotary_percentage=1.0 \
    model.data.splits_string='980,10,10' \
    model.megatron_amp_O2=$AMP_O2 \
    +model.full_bf16=$FULL_BF16 \
    model.optim.name=$optim_name \
    +model.optim.capturable=True \
    +model.optim.foreach=False \
    model.optim.lr=0.00012 \
    model.optim.betas=[0.9,0.95] \
    model.optim.weight_decay=0.1 \
    +model.optim.weight_decay_style='compact' \
    +model.optim.Kahan=$kahan \
    model.optim.sched.name=CosineAnnealing \
    model.optim.sched.warmup_steps=200 \
    model.optim.sched.constant_steps=0 \
    model.optim.sched.min_lr=0.000012 \
    model.sequence_parallel=$SEQ_PAR  \
    model.activations_checkpoint_granularity=$ACT_CHK \
    model.activations_checkpoint_method=uniform \
    model.activations_checkpoint_num_layers=null \
    +exp_manager.create_tensorboard_logger=True \
    exp_manager.create_checkpoint_callback=True \
    exp_manager.exp_dir=$TB_DIR \
    exp_manager.name=$EXP_NAME \
    +exp_manager.checkpoint_callback_params.every_n_epochs=0 \
    +exp_manager.checkpoint_callback_params.every_n_train_steps=5000 \
    +model.use_legacy_ptl_save=True \
    model.use_cpu_initialization=False
    "

if [ $NODEID -eq 0 ]
then
	docker exec $DOCKER_NAME torchrun $DISTRIBUTED_ARGS $SCRIPT_PATH/examples/nlp/language_modeling/megatron_gpt_pretraining.py \
        $GPT_ARGS \
        |& tee $EXP_PATH/run_log &
    wait %1
else
    docker exec $DOCKER_NAME torchrun $DISTRIBUTED_ARGS $SCRIPT_PATH/examples/nlp/language_modeling/megatron_gpt_pretraining.py \
        $GPT_ARGS &
    wait %1
fi