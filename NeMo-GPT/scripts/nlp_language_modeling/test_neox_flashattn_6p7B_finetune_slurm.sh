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
    # Single-node, non-SLURM runs
    HOSTS=(localhost)
    NODEID=0
    NTASKS=1
    export NEMO_EXPM_VERSION=$(date "+%Y-%m-%d_%H-%M-%S")
else
    # SLURM runs, single or multi-node
    IPS=""
    for h in $(scontrol show hostname); do
        IPS="$IPS $(nslookup $h  | awk '/^Address: / { print $2 }')";
    done
    HOSTS=(${IPS//\ / })
    NODEID=$SLURM_NODEID
    NTASKS=$SLURM_NTASKS
    export NEMO_EXPM_VERSION=$SLURM_JOB_ID
fi

export HYDRA_FULL_ERROR=1
export PROCESSES_PER_NODE=8
export MASTER_ADDR=${HOSTS[0]}
export MASTER_PORT=41000

DISTRIBUTED_ARGS="--nproc_per_node $PROCESSES_PER_NODE --nnodes $NTASKS --node_rank $NODEID --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
echo $DISTRIBUTED_ARGS

# export MALLOC_ARENA_MAX=128

# Make sure vocab file and data are under same folder
DATA_PATH=/fsx/fsx/gaurav/examples_datasets
SCRIPT_PATH=/fsx/fsx/gaurav/Python/benchmarking/GPT/neox_gpu/NeMo
DATA_SET=gpt2/my-gpt2_text_document
MODEL_PATH=/fsx/fsx/gaurav/pythia

NUM_CORES=8

EXP_PATH=$SCRIPT_PATH/test_result/$(date "+%y-%m-%d-%H-%M-%S")-gpt-neox-6p7B-nemo
DOCKER_PATH="${EXP_PATH/"$SCRIPT_PATH"/"/workspace/gpt_neox"}"

CHECKPOINT_PATH=$DOCKER_PATH/chkpt
TB_DIR=$DOCKER_PATH/tensorboard

mkdir -p $EXP_PATH
cp "$0" "$EXP_PATH"/

TRAIN_ITERS=2000

# DOCKER_NAME=nvidia-docker-nemo
# IMAGE_URI=nvcr.io/nvidia/nemo:23.06

REPO=gauravaz
TAG=nemo-23.06-efa
REGION=us-west-2

# Grab current AWS account from sts
AWS_ACCOUNT=`aws sts get-caller-identity --region ${REGION} --endpoint-url https://sts.${REGION}.amazonaws.com --query Account --output text`

# Log in to ECR
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com

IMAGE_URI=${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:${TAG}
DOCKER_NAME=nvidia-docker-nemo

docker stop $DOCKER_NAME

docker pull $IMAGE_URI
docker images
docker run  --name $DOCKER_NAME --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -d \
    --net=host --uts=host\
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
    -v $MODEL_PATH:/workspace/model \
    $IMAGE_URI

# docker exec -it $DOCKER_NAME /bin/bash
echo ">>> Installing packages ..."
docker exec $DOCKER_NAME pip install transformers==4.31.0
docker exec $DOCKER_NAME pip install ujson
echo ">>> Package installed."

echo ">>> Running script ..."
export CUDA_DEVICE_MAX_CONNECTIONS=1

TP=8
PP=1
UBS=32
# GBS=128
GBS=256
# GBS=1024

AL=1
SCRIPT_PATH=/workspace/gpt_neox
SEQ_LEN=2048
HS=4096
N_LAYERS=32
N_AH=32

export FFN_HS=$(($HS*4))

# Note: Choose factor "make_vocab_size_divisible_by" such that the overall multiplier=make_vocab_size_divisible_by*TP = 256 (the EleutherAI checkpoint is dependent on this)
export VOCAB_DIV_FACTOR=$((256/$TP))

echo "SEQ_LEN=$SEQ_LEN, HS=$HS, FFN_HS=$FFN_HS TP=$TP PP=$PP N_LAYERS=$N_LAYERS N_AH=$N_AH GBS=$GBS UBS=$UBS VOCAB_DIV_FACTOR=$VOCAB_DIV_FACTOR"

echo $SCRIPT_PATH_$EXP_PATH_$TP_$BS_$AL_$VERSION |& tee $EXP_PATH/run_config


GPT_ARGS="
    --config-path=conf \
    --config-name=megatron_neox_config \
    model.make_vocab_size_divisible_by=$VOCAB_DIV_FACTOR \
    model.use_flash_attention=True \
    model.transformer_engine=False \
    trainer.devices=$NUM_CORES \
    trainer.num_nodes=$NTASKS \
    trainer.max_epochs=null \
    trainer.max_steps=$TRAIN_ITERS\
    trainer.val_check_interval=$TRAIN_ITERS \
    trainer.log_every_n_steps=1 \
    trainer.limit_val_batches=1 \
    trainer.limit_test_batches=1 \
    trainer.accumulate_grad_batches=1 \
    trainer.precision='bf16' \
    model.micro_batch_size=$UBS \
    model.global_batch_size=$GBS \
    model.tensor_model_parallel_size=$TP \
    model.pipeline_model_parallel_size=$PP \
    model.max_position_embeddings=$SEQ_LEN \
    model.position_embedding_type=rope \
    model.rotary_percentage=0.25 \
    model.transformer_block_type=pre_ln \
    model.share_embeddings_and_output_weights=False \
    model.encoder_seq_length=$SEQ_LEN \
    model.hidden_size=$HS \
    model.ffn_hidden_size=$FFN_HS \
    model.num_layers=$N_LAYERS \
    model.num_attention_heads=$N_AH \
    model.init_method_std=0.021 \
    model.hidden_dropout=0 \
    model.layernorm_epsilon=1e-5 \
    model.tokenizer.type=EleutherAI/gpt-neox-20b \
    +model.tokenizer.use_fast=True \
    model.data.data_prefix=[1.0,/workspace/examples_datasets/gpt2/my-gpt2_text_document] \
    model.data.num_workers=2 \
    model.data.seq_length=$SEQ_LEN \
    model.data.splits_string='980,10,10' \
    model.optim.name=adamw \
    +model.optim.capturable=True \
    model.optim.lr=1.2e-4 \
    model.optim.betas=[0.9,0.95] \
    model.optim.weight_decay=0.01 \
    model.optim.sched.name=CosineAnnealing \
    model.optim.sched.warmup_steps=100 \
    model.optim.sched.constant_steps=0 \
    model.optim.sched.min_lr=1.2e-5 \
    model.sequence_parallel=True  \
    model.activations_checkpoint_granularity=full \
    model.activations_checkpoint_method=uniform \
    model.activations_checkpoint_num_layers=1 \
    +exp_manager.create_tensorboard_logger=True \
    exp_manager.create_checkpoint_callback=False \
    exp_manager.exp_dir=$TB_DIR \
    model.use_cpu_initialization=False \
    exp_manager.resume_if_exists=False \
    exp_manager.resume_ignore_no_checkpoint=False \
    exp_manager.create_checkpoint_callback=True \
    model.resume_from_checkpoint='/workspace/model/nemo/6p9B_nemo_ckpt/mp_rank_00/model_optim_rng.ckpt' \
    +model.restore_optim_states=False \
    "
# +model.save_xser=True \

docker exec $DOCKER_NAME torchrun $DISTRIBUTED_ARGS $SCRIPT_PATH/examples/nlp/language_modeling/megatron_gpt_pretraining.py \
    $GPT_ARGS \
    |& tee $EXP_PATH/run_log &
wait %1