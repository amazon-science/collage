#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

set -o pipefail

# Make sure vocab file and data are under same folder
DATA_PATH=/home/ubuntu/examples_datasets
SCRIPT_PATH=/home/ubuntu/Python/benchmarking/GPT/neox/NeMo
MEGATRON_PATH=/home/ubuntu/Python/benchmarking/GPT/neox/Megatron-LM
DATA_SET=gpt2/my-gpt2_text_document

NUM_CORES=8

DISTRIBUTED_ARGS="--nproc_per_node $NUM_CORES"
echo $DISTRIBUTED_ARGS

EXP_PATH=$SCRIPT_PATH/test_result/$(date "+%y-%m-%d-%H-%M-%S")-gpt-neox-6p7B-nemo
DOCKER_PATH="${EXP_PATH/"$SCRIPT_PATH"/"/workspace/gpt_neox"}"

CHECKPOINT_PATH=$DOCKER_PATH/chkpt
TB_DIR=$DOCKER_PATH/tensorboard

mkdir -p $EXP_PATH
cp "$0" "$EXP_PATH"/

TRAIN_ITERS=10000

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
    --net=host \
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
# echo ">>> Installing lightning packages ..."
# docker exec $DOCKER_NAME pip install -r /workspace/gpt_neox/requirements/requirements_lightning.txt
# echo ">>> Installing common packages ..."
# docker exec $DOCKER_NAME pip install -r /workspace/gpt_neox/requirements/requirements_common.txt
# echo ">>> Installing nlp packages ..."
# docker exec $DOCKER_NAME pip install -r /workspace/gpt_neox/requirements/requirements_nlp.txt
# docker exec $DOCKER_NAME pip install -r /workspace/gpt_neox/requirements/requirements.txt
docker exec $DOCKER_NAME pip install transformers==4.31.0
docker exec $DOCKER_NAME pip install ujson
echo ">>> Package installed."

echo ">>> Running script ..."
export CUDA_DEVICE_MAX_CONNECTIONS=1

TP=8
PP=1
UBS=32
GBS=256
AL=1
SCRIPT_PATH=/workspace/gpt_neox
SEQ_LEN=2048
HS=4096
N_LAYERS=32
N_AH=32
N_TASKS=1

export FFN_HS=$(($HS*4))
echo "SEQ_LEN=$SEQ_LEN, HS=$HS, FFN_HS=$FFN_HS TP=$TP PP=$PP N_LAYERS=$N_LAYERS N_AH=$N_AH GBS=$GBS UBS=$UBS"

echo $SCRIPT_PATH_$EXP_PATH_$TP_$BS_$AL_$VERSION |& tee $EXP_PATH/run_config

GPT_ARGS="
    --config-path=conf \
    --config-name=megatron_neox_config \
    model.make_vocab_size_divisible_by=$TP \
    trainer.devices=$NUM_CORES \
    trainer.num_nodes=$N_TASKS \
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
    model.rotary_percentage=1.0 \
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
    model.tokenizer.vocab_file=/workspace/examples_datasets/gpt2/gpt2-vocab.json \
    model.tokenizer.merge_file=/workspace/examples_datasets/gpt2/gpt2-merges.txt \
    model.data.data_prefix=[1.0,/workspace/examples_datasets/gpt2/my-gpt2_text_document] \
    model.data.num_workers=4 \
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
    model.activations_checkpoint_num_layers=$AL \
    +exp_manager.create_tensorboard_logger=True \
    exp_manager.create_checkpoint_callback=False \
    exp_manager.exp_dir=$TB_DIR \
    model.use_cpu_initialization=False
    "
# +model.save_xser=True \

docker exec $DOCKER_NAME torchrun $DISTRIBUTED_ARGS $SCRIPT_PATH/examples/nlp/language_modeling/megatron_gpt_pretraining.py \
    $GPT_ARGS \
    |& tee $EXP_PATH/run_log &
wait %1