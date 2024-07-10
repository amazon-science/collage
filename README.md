# Collage: Light-Weight Low-Precision Strategy for LLM Training

[![preprint](https://img.shields.io/static/v1?label=arXiv&message=2403.07815&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2405.03637)
[![License: MIT](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

Introduction
------------

Collage is a low-precision training strategy for large language models (LLMs). 
Collage makes use of multi-component floats to reduce the memory footprint during training, particularly for optimization, 
with purely low-precision (e.g. BFloat16) arithmetic without consorting to Float32. 

It is simple to use Collage by replacing `AdamW` with our `AdamW_collage` optimizer and use different `collage` options, i.e., `light` or `plus` 
(check our paper for more details). We provide Collage training for BERT & RoBERTa and also multi-size GPTs with the NeMo Megatron framework. 

Requirements
------------
1) python 3.8 or above
2) transformers 4.31.0 or above
3) pytorch 1.13.1 + CUDA 11.7 or above

We recommend using NeMo ``r1.22.0`` with released container ``nemo:23.11``

.. code-block:: bash

    docker pull nvcr.io/nvidia/nemo:23.11.framework

Datasets
--------

Please follow `"AWS-Neuron-Tutorials-BERT" <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/tutorials/training/bert.html#hf-bert-pretraining-tutorial>`_ to 
download the tokenized wikicorpus file for BERT and RoBERTa

.. code-block:: bash

    mkdir -p ./examples_datasets/
    pushd ./examples_datasets/
    aws s3 cp s3://neuron-s3/training_datasets/bert_pretrain_wikicorpus_tokenized_hdf5/bert_pretrain_wikicorpus_tokenized_hdf5_seqlen128.tar .  --no-sign-request
    tar -xf bert_pretrain_wikicorpus_tokenized_hdf5_seqlen128.tar
    rm bert_pretrain_wikicorpus_tokenized_hdf5_seqlen128.tar
    aws s3 cp s3://neuron-s3/training_datasets/bert_pretrain_wikicorpus_tokenized_hdf5/bert_pretrain_wikicorpus_tokenized_hdf5_seqlen512.tar .  --no-sign-request
    tar -xf bert_pretrain_wikicorpus_tokenized_hdf5_seqlen512.tar
    rm bert_pretrain_wikicorpus_tokenized_hdf5_seqlen512.tar
    popd

Please follow `"AWS-Neuron-Examples-GPT" <https://github.com/aws-neuron/aws-neuron-parallelcluster-samples/blob/master/examples/jobs/neuronx-nemo-megatron-gpt-job.md>`_ to 
download the wikipedia dataset that is stored in s3

.. code-block:: bash

    export DATA_DIR=./examples_datasets/gpt2
    mkdir -p ${DATA_DIR} && cd ${DATA_DIR}
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
    aws s3 cp s3://neuron-s3/training_datasets/gpt/wikipedia/my-gpt2_text_document.bin .  --no-sign-request
    aws s3 cp s3://neuron-s3/training_datasets/gpt/wikipedia/my-gpt2_text_document.idx .  --no-sign-request
    aws s3 cp s3://neuron-s3/training_datasets/gpt/wikipedia/license.txt .  --no-sign-request

Examples
--------

Scripts for training BERT and RoBERTa are provided in `"roBERTa/scripts" <https://github.com/ydtydr/Trn-ext-nemo-neox/tree/gpu_nemo_r1.22.0/roBERTa/scripts>`_ folder.
Scripts for multi-size (125M, 1.3B, 2.7B and 6.7B) GPTs can be found in `"NeMo-GPT/scripts/nlp_language_modeling" <https://github.com/ydtydr/Trn-ext-nemo-neox/tree/gpu_nemo_r1.22.0/NeMo-GPT/scripts/nlp_language_modeling>`_ folder. 

Cite us
------------

If you find our works helpful in your research, please consider citing the following paper:

.. code:: bibtex

    @inproceedings{yu2024collage,
        title={Collage: Light-Weight Low-Precision Strategy for LLM Training},
        author={Yu, Tao and Gupta, Gaurav and Gopalswamy, Karthick and Mamidala, Amith and Zhou, Hao and Huynh, Jeffrey and Park, Youngsuk and Diamant, Ron and Deoras, Anoop and Huan, Luke},
        booktitle={Proceedings of the 41st International Conference on Machine Learning (ICML 2024)},
        year={2024},
        organization={PMLR}
    }

License
-------
NeMo-GPT is modified from NVIDIA NeMo, which is released under an `Apache 2.0 license <https://github.com/NVIDIA/NeMo/blob/stable/LICENSE>`.

Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

This code is being released solely for academic and scientific reproducibility purposes, in support of the methods and findings described in the associated publication. Pull requests are not being accepted in order to maintain the code exactly as it was used in the paper, but interested parties are encouraged to open an issue requesting open source community development.
