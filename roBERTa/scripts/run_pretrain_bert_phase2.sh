# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

NUM_GPUS=8
# BERT-base
torchrun --nproc_per_node=$NUM_GPUS --master_port 29501 pretrain_llm.py \
        --data_dir ./examples_datasets/bert_pretrain_wikicorpus_tokenized_hdf5_seqlen512/ \
        --output_dir ./output_bert_base_phase1/insert_phase1_ckpt_here \
        --model_type bert \
        --model_size base \
        --lr 2.8e-4 \
        --beta1 0.9 \
        --beta2 0.999 \
        --weight_decay_style compact \
        --enable_checkpointing \
        --full_bf16 \
        --Collage plus \
        --max_steps 1563 \
        --phase1_end_step 28125 \
        --warmup_steps 781 \
        --max_pred_len 80 \
        --phase2 \
        --resume_ckpt \
        --batch_size 128 \
        --grad_accum_usteps 32 |& tee bert_base_phase2.txt
# --monitor_metrics