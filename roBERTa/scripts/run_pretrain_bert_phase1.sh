# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

NUM_GPUS=8
# BERT-base
torchrun --nproc_per_node=$NUM_GPUS --master_port 29501 pretrain_llm.py \
        --data_dir ./examples_datasets/bert_pretrain_wikicorpus_tokenized_hdf5_seqlen128/ \
        --output_dir ./output_bert_base_phase1 \
        --model_type bert \
        --model_size base \
        --lr 4e-4 \
        --beta1 0.9 \
        --beta2 0.999 \
        --weight_decay_style compact \
        --full_bf16 \
        --Collage plus \
        --max_steps 28125 \
        --phase1_end_step 28125 \
        --warmup_steps 2000 \
        --batch_size 128 \
        --grad_accum_usteps 16 |& tee bert_base_phase1.txt
# --monitor_metrics