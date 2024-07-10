# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

NUM_GPUS=8
# RoBERTa-base
torchrun --nproc_per_node=$NUM_GPUS --master_port 29501 pretrain_llm.py \
        --data_dir ./examples_datasets/roberta_pretrain_wikicorpus_tokenized_hdf5_seqlen512/ \
        --output_dir ./output_roberta_base \
        --model_type roberta \
        --model_size base \
        --lr 6e-4 \
        --beta1 0.9 \
        --beta2 0.98 \
        --weight_decay_style compact \
        --full_bf16 \
        --Collage light \
        --max_steps 28125 \
        --phase1_end_step 28125 \
        --warmup_steps 2000 \
        --batch_size 16 \
        --grad_accum_usteps 64 |& tee roberta_base.txt
# --monitor_metrics