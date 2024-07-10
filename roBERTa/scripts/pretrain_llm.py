# coding=utf-8
# Original copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Original copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script is BERT pretraining script adapted from
# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/run_pretraining.py
# Modifications done:
# - Add HuggingFace model/optimizer
# - Add AdamW_collage optimizer
# - Add and Tweak arg names to be more intuitive (distringuish micro-steps from global steps)
# - Changed arg defaults
# - Added logger class to print log and also log to TensorBoard database

import os

import torch
import glob
import h5py
import sys
import time
import argparse
import random
import json
import queue
from typing import Any, Dict, List
from datetime import datetime
from collections import deque, namedtuple

import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import Dataset

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from transformers import BertForPreTraining, RobertaConfig, RobertaModel, RobertaForMaskedLM
from transformers import (
    AdamW,
    DataCollatorForLanguageModeling,
    set_seed,
)
from transformers.optimization import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
import copy
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from torch.utils.tensorboard import SummaryWriter
from concurrent.futures import ThreadPoolExecutor
import inspect
import requests
import gc

sys.path.insert(0, '..')
from utils.monitor_metrics import get_param_norm, get_grad_norm, training_metrics_closure
from utils.adamw_collage import AdamW_collage

os.environ["NEURON_CC_FLAGS"] = "--model-type=transformer"
Metric = namedtuple("Metric", ["name", "value", "units", "additional_data"])

class TrainingMetrics:
    def __init__(self, json_file):
        self.json_file = json_file

    def read_modify_write_file(self, data, key: str = "metrics") -> None:
        """
        data (dict of training parameters or list of metrics): Data to update in the file.
        key (str): the dictionary key under which data is to be recorded
        """
        result_dict = {}
        print(f"Writing data to the provided results file: {self.json_file}")
        if os.path.exists(self.json_file):
            with open(self.json_file) as json_file:
                result_dict = json.loads(json_file.read()) or result_dict
        print(f"Updating with {key} data: {data}")
        if result_dict:
            try:
                # handle internal named entity if present
                results = result_dict[next(iter(result_dict))]
            except Exception:
                results = result_dict
            current = results.get(key)
            if not current:
                results[key] = data
            else:
                if isinstance(current, list):
                    current.extend(data)
                elif isinstance(current, dict):
                    current.update(data)
        else:
            result_dict[key] = data
        with open(self.json_file, 'w') as json_file:
            json.dump(result_dict, json_file)

    def store_metrics(self, metrics: List[Metric]) -> None:
        """
        Writes collected metrics to the file.

        """
        data = [
            {
                "MetricName": metric.name,
                "MeasuredValue": metric.value,
                "Units": metric.units,
                "Timestamp": datetime.now().isoformat(),
                "AdditionalData": metric.additional_data,
            } for metric in metrics
        ]
        self.update(data=data, key="metrics")

    def store_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Writes specified model and configuration parameters to the file.

        """
        self.update(data=parameters, key="parameters")

    def update(self, **kwargs: Any) -> None:
        """
        Write specified data to the output file.
        """
        self.read_modify_write_file(**kwargs)


class Throughput:
    def __init__(self, batch_size, world_size, grad_accum_usteps, moving_avg_window_size=10):
        self.seqs_per_iteration = batch_size * world_size * grad_accum_usteps
        self.moving_avg_window_size = moving_avg_window_size
        self.moving_avg_window = queue.Queue()
        self.window_time = 0
        self.start_time = time.time()

    def get_throughput(self):
        step_time = time.time() - self.start_time
        self.start_time += step_time
        self.window_time += step_time
        self.moving_avg_window.put(step_time)
        window_size = self.moving_avg_window.qsize()
        if window_size > self.moving_avg_window_size:
            self.window_time -= self.moving_avg_window.get()
            window_size -= 1
        throughput = window_size * self.seqs_per_iteration / self.window_time
        return throughput


class Logger:
    def __init__(self, args, world_size, model_dtype):
        xla = 'torch_xla' in sys.modules
        self.throughputs = []
        dtype_short = model_dtype.replace("torch.", "")
        folder_name = args.output_dir
        # folder_name = os.path.join(args.output_dir,
        #                                     f"{time.strftime('%m%d%y_%H%M')}"
        #                                     f"_{dtype_short}"
        #                                     f"_mixprec{args.upcast_optim_states}"
        #                                     f"_mw{args.enable_master_weight}"
        #                                     f"_Collage{args.Collage}"
        #                                     f"_max{args.max_steps}"
        #                                     f"_{args.lr_type}lr"
        #                                     f"_{args.weight_decay_style}wd"
        #                                     f"_seed{args.seed}"
        #                                     )
        self.tb = SummaryWriter(folder_name)
        print('This run log folder directory', folder_name)
        self.tb.add_text('script', "```\n" + inspect.getsource(sys.modules[__name__]) + "\n```", 0)
        self.golden_steploss = []
        golden = "golden_steploss.txt"
        if os.path.exists(golden):
            with open(golden, "r") as f:
                self.golden_steploss = [float(i) for i in f]
            print(f"Read {len(self.golden_steploss)} golden step loss values from {golden}")

    def log(self, epoch, step, metrics_dict):
        time_now = time.asctime()
        print(f'LOG {time_now} - ({epoch}, {step}) step_loss: {metrics_dict["step_loss"]}, throughput: {metrics_dict["throughput"]}', flush=True)
        self.throughputs.append(metrics_dict['throughput'])
        for metric_name in metrics_dict:
            if isinstance(metrics_dict[metric_name], dict):
                for layer_keys in metrics_dict[metric_name].keys():
                    self.tb.add_scalar(f'{metric_name}/{layer_keys}', metrics_dict[metric_name][layer_keys], step)
            else:
                self.tb.add_scalar(metric_name, metrics_dict[metric_name], step)
        step_0start = step - 1
        if step_0start < len(self.golden_steploss) and step_0start >= 0:
            np.testing.assert_allclose(metrics_dict['step_loss'], self.golden_steploss[step_0start], rtol=2.3e-1)
        assert (not np.isnan(metrics_dict['step_loss'])), "Encountered NaN in loss!"

# Workaround because python functions are not picklable
class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, id):
        set_seed(self.seed)

def create_pretraining_dataset(input_file, max_pred_length, mini_batch_size, worker_init):
    train_data = pretraining_dataset(input_file=input_file, max_pred_length=max_pred_length)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data,
                                  sampler=train_sampler,
                                  batch_size=mini_batch_size,
                                  num_workers=0,
                                  worker_init_fn=worker_init,
                                  drop_last=True,
                                  pin_memory=True)
    return train_dataloader, input_file


class pretraining_dataset(Dataset):
    def __init__(self, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, 'r')
        keys = ['input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids',
                'next_sentence_labels']
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs[0])

    def __getitem__(self, index):
        [input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels] = [
            torch.from_numpy(input[index].astype(np.int64)) if indice < 5 else torch.from_numpy(
                np.asarray(input[index].astype(np.int64))) for indice, input in enumerate(self.inputs)]

        # in torch.nn.NLLLoss, the default ignore-index is -100
        ignore_index = -100
        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * ignore_index
        index = self.max_pred_length
        # store number of  masked tokens in index
        padded_mask_indices = (masked_lm_positions == 0).nonzero()
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        return [input_ids, segment_ids, input_mask,
                masked_lm_labels, next_sentence_labels]

    @property
    def sequence_length(self) -> int:
        """
        Returns the sequence length derived from the specified pre-tokenized dataset.

        """
        return len(self.inputs[0][0])


def get_model(flags):
    if flags.model_type == 'bert':
        # medium BERT size L12_A12_H768. Large BERT L24_A16_H1024 causes OOM on GPU V100
        base_model = BertForPreTraining.from_pretrained(f'{flags.model_type}-{flags.model_size}-{flags.bert_case}')
    elif flags.model_type == 'roberta':
        base_model = RobertaForMaskedLM.from_pretrained(f'{flags.model_type}-{flags.model_size}')
    else:
        raise NotImplemented
    my_config = copy.deepcopy(base_model.config)
    if flags.debug:
        my_config.num_hidden_layers = 1
        my_config.num_attention_heads = 2
        my_config.hidden_size = 16
    if flags.model_type == 'bert':
        my_model = BertForPreTraining(my_config)
    else:
        my_model = RobertaForMaskedLM(my_config)
    if flags.full_bf16:
        my_model.to(torch.bfloat16)
    return my_model

# fix NVidia checkpoint param names to match HF
def fix_ckpt_params(state_dict):
    keys = [k for k in state_dict.keys() if 'dense_act' in k]
    for k in keys:
        new_k = k.replace('dense_act', 'dense')
        state_dict[new_k] = state_dict[k]
        del state_dict[k]
    keys = [k for k in state_dict.keys() if k.startswith('module.')]
    for k in keys:
        new_k = k.replace('module.', '')
        state_dict[new_k] = state_dict[k]
        del state_dict[k]

def train_model_hdf5(index, world_size, world_rank, flags, **kwags):
    rank = world_rank
    is_root = rank == 0
    set_seed(flags.seed)
    worker_init = WorkerInitObj(flags.seed)
    device = index  # local rank
    model = get_model(flags)
    model.to(device)
    model.tie_weights()
    # Additional tie needed
    # https://github.com/huggingface/transformers/blob/v4.12.0/src/transformers/models/bert/modeling_bert.py#L669
    if flags.model_type == 'bert':
        model.cls.predictions.decoder.bias = model.cls.predictions.bias
    else:
        model.lm_head.decoder.bias = model.lm_head.bias
    model.train()
    model_dtype = str(model.dtype)
    running_loss = torch.zeros(1, dtype=torch.double).to(device)

    no_decay = ['bias', 'LayerNorm']  # gamma/beta are in LayerNorm.weight
    # old ways
    # param_optimizer = list(model.named_parameters())
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': flags.weight_decay},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]

    # new ways for better metric logging
    optimizer_grouped_parameters = []
    if flags.model_type == 'bert':
        optimizer_grouped_parameters.extend(
            [{'params': [p for n, p in model.bert.embeddings.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, 'name': 'model.embeddings'},
            {'params': [p for n, p in model.bert.embeddings.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'name': 'model.embeddings'}]
            )
        for bert_layer_index, bert_layer in model.bert.encoder.layer.named_children():
            optimizer_grouped_parameters.extend(
                [{'params': [p for n, p in bert_layer.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, 'name': f'model.encoder.layer{bert_layer_index}'},
                {'params': [p for n, p in bert_layer.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'name': f'model.encoder.layer{bert_layer_index}'}]
                )
        optimizer_grouped_parameters.extend(
            [{'params': [p for n, p in model.cls.named_parameters() if not any(nd in n for nd in no_decay) and n!='predictions.decoder.weight'], 'weight_decay': 0.01, 'name': 'model.last'},
            {'params': [p for n, p in model.cls.named_parameters() if any(nd in n for nd in no_decay) and n!='predictions.decoder.weight'], 'weight_decay': 0.0, 'name': 'model.last'}]
            )
    else:
        optimizer_grouped_parameters.extend(
            [{'params': [p for n, p in model.roberta.embeddings.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, 'name': 'model.embeddings'},
            {'params': [p for n, p in model.roberta.embeddings.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'name': 'model.embeddings'}]
            )
        for bert_layer_index, bert_layer in model.roberta.encoder.layer.named_children():
            optimizer_grouped_parameters.extend(
                [{'params': [p for n, p in bert_layer.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, 'name': f'model.encoder.layer{bert_layer_index}'},
                {'params': [p for n, p in bert_layer.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'name': f'model.encoder.layer{bert_layer_index}'}]
                )
        optimizer_grouped_parameters.extend(
            [{'params': [p for n, p in model.lm_head.named_parameters() if not any(nd in n for nd in no_decay) and n!='decoder.weight'], 'weight_decay': 0.01, 'name': 'model.last'},
            {'params': [p for n, p in model.lm_head.named_parameters() if any(nd in n for nd in no_decay) and n!='decoder.weight'], 'weight_decay': 0.0, 'name': 'model.last'}]
            )

    optimizer = AdamW_collage(optimizer_grouped_parameters, flags.lr, betas=(flags.beta1, flags.beta2), 
                                weight_decay_style=flags.weight_decay_style, 
                                upcast_optim_states=flags.upcast_optim_states,
                                enable_master_weight=flags.enable_master_weight,
                                Collage=flags.Collage, monitor_metrics=flags.monitor_metrics)
    optimizer.zero_grad()

    if is_root:
        if not os.path.exists(flags.output_dir):
            os.makedirs(flags.output_dir, exist_ok=True)
        logger = Logger(flags, world_size, model_dtype)
        metric_writer = TrainingMetrics(flags.metrics_file)
        throughput = Throughput(flags.batch_size, world_size, flags.grad_accum_usteps)
        print('--------TRAINING CONFIG----------')
        print(flags)
        print('--------MODEL CONFIG----------')
        print(model.config)
        print('---------------------------------')
        metric_writer.store_parameters(
            {
                "Model": model.name_or_path,
                "Model configuration": str(model.config),
                "World size": world_size,
                "Data parallel degree": world_size,
                "Batch size": flags.batch_size,
                "Max steps": flags.max_steps,
                "Steps this run": flags.steps_this_run,
                "Seed": flags.seed,
                "Optimizer": str(optimizer),
                "Data type": model_dtype,
                "Gradient accumulation microsteps": flags.grad_accum_usteps,
                "Warmup steps": flags.warmup_steps,
                "Shards per checkpoint": flags.shards_per_ckpt,
                "Dataset": os.path.basename(os.path.normpath(flags.data_dir))
            }
        )

    def train_loop_fn(model, optimizer, train_loader, epoch, 
                      global_step, training_ustep, running_loss
                      ):
        running_loss_cpu = 0.0
        for i, data in enumerate(train_loader):
            training_ustep += 1
            input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = data
            with torch.autocast(enabled=flags.enable_pt_autocast, dtype=torch.bfloat16, device_type='cuda'):
                if flags.model_type == 'bert':
                    outputs = model(input_ids=input_ids,
                                    attention_mask=input_mask,
                                    # token_type_ids=segment_ids,
                                    labels=masked_lm_labels,
                                    next_sentence_label=next_sentence_labels
                                    )
                else:
                    outputs = model(input_ids=input_ids,
                                    attention_mask=input_mask,
                                    # token_type_ids=segment_ids,
                                    labels=masked_lm_labels,
                                    # next_sentence_label=next_sentence_labels
                                    )
                loss = outputs.loss / flags.grad_accum_usteps
            loss.backward()
            running_loss += loss.detach()

            if training_ustep % flags.grad_accum_usteps == 0:
                # dict of metrics
                metrics_dict = {}
                # loss averaging
                running_loss_div = running_loss / world_size
                dist.all_reduce(running_loss_div, dist.ReduceOp.SUM, async_op=False)
                running_loss_cpu = running_loss_div.detach().cpu().item()
                metrics_dict['step_loss'] = running_loss_cpu
                running_loss.zero_()
                # all-reduce and then clip. Order matters.
                torch.nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)  # Gradient clipping is not in AdamW anymore
                if flags.monitor_metrics:
                    _, partial_metrics = optimizer.step()
                    metrics_dict.update(partial_metrics)
                else:
                    optimizer.step()

                if is_root and flags.monitor_metrics:
                    with torch.no_grad():
                        model_for_metrics = model.module if hasattr(model, 'module') else model  # unwrap model if needed (DDP)
                        metrics_dict['param_norm'] = get_param_norm(model_for_metrics, report_per_block=True)
                        metrics_dict['grad_norm'] = get_grad_norm(model_for_metrics, report_per_block=True)

                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                if is_root:
                    metrics_dict['learning_rate'] = optimizer.param_groups[0]['lr']
                    metrics_dict['throughput'] = throughput.get_throughput()
                    training_metrics_closure(logger, epoch, global_step, metrics_dict)
                if global_step >= flags.steps_this_run:
                    # NOTE: Prevent runtime "Call to recv failed : Broken pipe" issue
                    break
        return global_step, training_ustep, running_loss, running_loss_cpu

    scheduler_state_dict = None
    if flags.resume_ckpt:
        if flags.resume_ckpt_path:
            ckpt_path = flags.resume_ckpt_path
            assert (os.path.exists(
                ckpt_path)), "Checkpoint path passed to resume_ckpt_path option is not a path: {}".format(ckpt_path)
            ckpt_file = os.path.basename(ckpt_path)
            global_step = int(ckpt_file.split('.pt')[0].split('_')[1].strip())
        else:
            if flags.resume_step == -1 or flags.phase2:
                assert (os.path.exists(flags.output_dir) and os.path.isdir(flags.output_dir)), \
                    "Resuming from last checkpoint in {}, but it doesn't exist or is not a dir. ".format(
                        flags.output_dir) \
                    + "You can also specify path to checkpoint using resume_ckpt_path option."
                model_names = [f for f in os.listdir(flags.output_dir) if f.endswith(".pt")]
                assert len(model_names) > 0, "Make sure there are ckpt_*.pt files in {}".format(flags.output_dir)
                global_step = max([int(x.split('.pt')[0].split('_')[1].strip()) for x in model_names])
            else:
                global_step = flags.resume_step
            ckpt_path = os.path.join(flags.output_dir, "ckpt_{}.pt".format(global_step))

        # Checkpoint loading must be flow controlled across the world to avoid host OOM.
        num_loading_workers = 16
        all_workers = list(range(world_size))
        for worker_start in range(0, world_size, num_loading_workers):
            if rank in all_workers[worker_start:worker_start + num_loading_workers]:
                print(f'Worker {rank} resuming from checkpoint {ckpt_path} at step {global_step}', flush=True)
                check_point = torch.load(ckpt_path, map_location='cpu')
                fix_ckpt_params(check_point['model'])
                model.load_state_dict(check_point['model'], strict=True)
                if not flags.phase2 and 'optimizer' in check_point:
                    optimizer.load_state_dict(check_point['optimizer'])
                    scheduler_state_dict = check_point.pop('scheduler')
                files = check_point['files'][1:]
                file_start_idx = check_point['files'][0]
                epoch = check_point.get('epoch', 0)
                del check_point
                gc.collect()
        if flags.phase2:
            global_step -= flags.phase1_end_step
    else:
        global_step = 0
        epoch = 0

    model = DDP(model, device_ids=[index], output_device=index)

    # compress allreduce to bf16 via DDP communication hook:
    # https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_wrapper
    # from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as default
    # model.register_comm_hook(dist.group.WORLD, default.bf16_compress_hook)

    train_start = time.time()
    training_ustep = 0
    if flags.lr_type == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=flags.warmup_steps,
                                                    num_training_steps=flags.max_steps,
                                                    last_epoch=epoch if scheduler_state_dict else -1)
    elif flags.lr_type == 'constant':
        scheduler = get_constant_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=flags.warmup_steps,
                                                    last_epoch=epoch if scheduler_state_dict else -1)
    else:
        raise NotImplemented

    if scheduler_state_dict:
        scheduler.load_state_dict(scheduler_state_dict)

    thread_pool = ThreadPoolExecutor(1)
    chkpt_files = deque([])

    assert (os.path.exists(os.path.expanduser(flags.data_dir))), "ERROR: Data directory {} doesn't exist!".format(
        flags.data_dir)
    while True:
        if flags.resume_ckpt and not flags.phase2:
            flags.resume_ckpt = False
        else:
            files = glob.glob(os.path.expanduser(flags.data_dir) + "/*_{}_*.hdf5".format("training"))
            files.sort()
            random.Random(epoch).shuffle(files)
            file_start_idx = 0

        num_files = len(files)
        assert (num_files > 0), "ERROR: There are no tokenized dataset shard files in {}".format(flags.data_dir)
        assert (
                    world_size <= num_files), "ERROR: Please ensure there are at least {} (world_size) tokenized dataset shards in {} (currently I see only {}).".format(
            world_size, flags.data_dir, num_files)
        mini_batch_size = flags.batch_size
        # prep first iteration input data file
        data_file = files[(file_start_idx * world_size + rank) % num_files]
        prev_file = data_file
        train_dataloader, _ = create_pretraining_dataset(data_file, flags.max_pred_len, mini_batch_size, worker_init)
        if flags.seq_len is not None:
            assert flags.seq_len == train_dataloader.dataset.sequence_length, (
                f"ERROR: User-specified sequence length ({flags.seq_len}) does not match "
                f"that of the pre-tokenized dataset ({train_dataloader.dataset.sequence_length})"
            )
        train_device_loader = train_dataloader  # pl.MpDeviceLoader(train_dataloader, device)
        if is_root:
            metric_writer.store_parameters(
                {"Sequence length": train_dataloader.dataset.sequence_length}
            )

        # use DP dataloader
        for f in range(file_start_idx + 1, len(files)):
            # select data file to preload for the next iteration
            data_file = files[(f * world_size + rank) % num_files]

            future_train_dataloader = thread_pool.submit(create_pretraining_dataset, data_file, flags.max_pred_len,
                                                         mini_batch_size, worker_init)
            if is_root:
                print('Epoch {} file index {} begin {}'.format(epoch, f, time.asctime()), flush=True)
            print(f'Rank {rank} working on shard {prev_file}', flush=True)
            global_step, training_ustep, running_loss, final_loss = train_loop_fn(
                model, optimizer, train_device_loader, 
                epoch, global_step, training_ustep, running_loss)
            print('GPU {} Epoch {} step {} file index {} loss {} '.format(
                rank, epoch, global_step, f, final_loss), flush=True)
            if is_root:
                final_time = time.time()
                time_diff = final_time - train_start
                print('Epoch {} step {} file index {} end {} loss {} perf {} seq/sec (at train microstep {} time {} from beginning time {})'.format(
                        epoch, global_step, f, time.asctime(), final_loss, logger.throughputs[-1], training_ustep,
                        final_time, train_start), flush=True)
                additional_data = {"Epoch": epoch, "Global step": global_step, "Microstep": training_ustep,
                                   "File index": f}
                metric_data = [
                    Metric("Loss", final_loss, "", additional_data),
                    Metric("Throughput", logger.throughputs[-1], "seq/s", additional_data)
                ]
                metric_writer.store_metrics(metric_data)
                
                if flags.enable_checkpointing:
                    if (f % flags.shards_per_ckpt == 0) or (global_step >= flags.steps_this_run):
                        if flags.phase2:
                            chkpt_file = os.path.join(flags.output_dir,
                                                    "ckpt_{}.pt".format(global_step + flags.phase1_end_step))
                        else:
                            chkpt_file = os.path.join(flags.output_dir, "ckpt_{}.pt".format(global_step))
                        files_info = [f] + files
                        print('Checkpointing...', flush=True)
                        model_to_save = model.module if hasattr(model, 'module') else model  # unwrap model if needed (DDP)
                        if flags.minimal_ckpt:
                            data = {'model': model_to_save.state_dict(),
                                    'files': files_info,
                                    'epoch': epoch}
                        else:
                            data = {'model': model_to_save.state_dict(),
                                    'optimizer': optimizer.state_dict(),
                                    'scheduler': scheduler.state_dict(),
                                    'files': files_info,
                                    'epoch': epoch}
                        cpu_data = data
                        torch.save(cpu_data, chkpt_file)
                        print('Checkpointing done...', flush=True)
                        del cpu_data
                        chkpt_files.append(chkpt_file)
                        if flags.num_ckpts_to_keep >= 0 and len(chkpt_files) > flags.num_ckpts_to_keep:
                            old_file = chkpt_files.popleft()
                            if os.path.isfile(old_file):
                                print('Keeping only {} checkpoints. Deleting {}'.format(flags.num_ckpts_to_keep, old_file))
                                os.remove(old_file)
            if global_step >= flags.steps_this_run:
                if is_root:
                    # record aggregate & final statistics in the metrics file
                    additional_data = {
                        "Epoch": epoch, "Global step": global_step, "Microstep": training_ustep
                    }
                    average_throughput = round(sum(logger.throughputs) / len(logger.throughputs), 4)
                    metric_data = [
                        Metric("Final loss", final_loss, "", additional_data),
                        Metric("Time to train", round(time_diff / 60, 4), "minutes", additional_data),
                        Metric("Average throughput", average_throughput, "seq/s", additional_data),
                        Metric("Peak throughput", max(logger.throughputs), "seq/s", additional_data)
                    ]
                    metric_writer.store_metrics(metric_data)
                return
            del train_device_loader
            del train_dataloader
            gc.collect()
            train_dataloader, _ = future_train_dataloader.result(timeout=1000)
            train_device_loader = train_dataloader  # pl.MpDeviceLoader(train_dataloader, device)
            prev_file = data_file

        epoch += 1

def _mp_fn(index, world_size, world_rank, flags, run_fn):
    # torch.set_default_tensor_type('torch.FloatTensor')
    if not world_rank:
        world_rank = index
    dist.init_process_group("nccl", rank=world_rank, world_size=world_size)
    run_fn(index, world_size, world_rank, flags)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='~/tao_codebase/examples_datasets/bert_pretrain_wikicorpus_tokenized_hdf5_seqlen128/',
                        help="Pre-tokenized HDF5 dataset directory.")
    parser.add_argument('--output_dir', type=str, default='./output', help="Directory for checkpoints and logs.")
    parser.add_argument('--metrics_file', type=str, default='results.json', help="training metrics results file")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Worker batch size. (for GPU fp32 use 32; for Trainium fp32 use 8)")
    parser.add_argument('--max_steps', type=int, default=28125, help="Maximum total accumulation-steps to run.")
    parser.add_argument('--steps_this_run', type=int, default=-1,
                        help="Exit early at <value> steps and not go to max_steps. -1 to mean no early exit.")
    parser.add_argument('--shards_per_ckpt', type=int, default=1,
                        help="Number of dataset shards before saving checkpoint.")
    parser.add_argument('--enable_checkpointing', default=False, action="store_true", help="Enable Checkpointing.")
    parser.add_argument('--seed', type=int, default=12349, help="Random seed. Worker seed is this value + worker rank.")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate.")
    parser.add_argument('--lr_type', default='linear', type=str, choices=['linear', 'constant'], help="Linear or constant lr schedule with warmup.")
    parser.add_argument('--beta1', type=float, default=0.9, help="adam beta1.")
    parser.add_argument('--beta2', type=float, default=0.999, help="adam beta1.")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="max grad norm.")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="weight_decay.")
    parser.add_argument('--weight_decay_style', type=str, default='compact', help="weight_decay place.")
    # parser.add_argument("--seq_len", type=int, default=None,
    #                     help="Sequence length; if specified, must match that of the pre-tokenized dataset, else derived from the dataset (via `--data_dir`)")
    parser.add_argument("--debug", action="store_true", help="Debug mode to help debug scripting.")
    parser.add_argument("--max_pred_len", type=int, default=20,
                        help="Maximum length of masked tokens in each input sequence.")
    parser.add_argument("--num_ckpts_to_keep", type=int, default=1,
                        help="Keep last N checkpoints only. -1 is to keep all.")
    parser.add_argument('--resume_ckpt', action="store_true", help="Resume from checkpoint at resume_step.")
    parser.add_argument('--resume_ckpt_path',
                        help="Checkpoint file to use rather than default. If not specified, then resume from last checkpoint or at resume_step (default file output/ckpt_<step>.pt).")
    parser.add_argument('--resume_step', type=int, default=-1,
                        help="Accumulated step to resume. Checkpoint file corresponding to accumulation-step count must exist. -1 means find the last checkpoint.")
    parser.add_argument("--warmup_steps", type=int, default=2000,
                        help="Number of warmup accumulation-steps for learning rate .")
    parser.add_argument("--grad_accum_usteps", type=int, default=64,
                        help="Gradient accumulation micro-steps (an accumulation-step has <value> micro-steps.")
    parser.add_argument('--minimal_ckpt', default=False, action='store_true',
                        help="When specified, don't store optimizer/lr-schedule states in checkpoints.")
    parser.add_argument('--enable_pt_autocast', default=False, action="store_true", help="Enable pytorch autocast.")
    parser.add_argument('--full_bf16', default=False, action="store_true", help="Train the model in BF16.")
    parser.add_argument('--upcast_optim_states', default=False, action="store_true", help="upcast_optim_states.")
    parser.add_argument('--enable_master_weight', default=False, action="store_true", help="enable_master_weight.")
    parser.add_argument('--monitor_metrics', default=False, action="store_true", help="print lost error due to imprecision in optimizer.")
    parser.add_argument('--Collage', type=str, default='none', help="which Collage option for the optimization.")
    parser.add_argument('--phase1_end_step', type=int, default=28125,
                        help="Number of training steps in Phase1 - seq len 128")
    parser.add_argument('--phase2', default=False, action='store_true', help="Whether to train with seq len 512")
    parser.add_argument('--model_type', default='bert', type=str, choices=['bert', 'roberta'], help="which model to use (bert | roberta).")
    parser.add_argument('--model_size', default='base', type=str, choices=['base', 'large'], help="Size of the model to train.")
    parser.add_argument('--bert_case', default='uncased', type=str, choices=['uncased', 'cased'], help="Casing of BERT model to train.")
    args = parser.parse_args(sys.argv[1:])

    if not args.phase2:
        args.output_dir = os.path.join(args.output_dir,
            f"{time.strftime('%m%d%y_%H%M')}"
            f"_BF16{args.full_bf16}"
            f"_mixprec{args.upcast_optim_states}"
            f"_mw{args.enable_master_weight}"
            f"_Collage{args.Collage}"
            f"_max{args.max_steps}"
            f"_phase1_end_step{args.phase1_end_step}"
            f"_phase2{args.phase2}"
            f"_{args.lr_type}lr"
            f"_{args.weight_decay_style}wd"
            f"_seed{args.seed}")
    else:
        args.metrics_file = 'phase2_' + args.metrics_file
    args.metrics_file = os.path.join(args.output_dir, args.metrics_file)

    if args.phase2:
        assert args.model_type == 'bert'
        args.seq_len = 512
    else:
        if args.model_type == 'bert':
            args.seq_len = 128
        else:
            args.seq_len = 512

    if args.full_bf16:
        torch.set_default_tensor_type(torch.BFloat16Tensor)

    if args.steps_this_run < 0:
        args.steps_this_run = args.max_steps

    if os.environ.get("OMPI_COMM_WORLD_SIZE"):
        local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        _mp_fn(local_rank, world_size, world_rank, args, train_model_hdf5)
    elif os.environ.get("WORLD_SIZE"):
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        world_rank = int(os.environ['RANK'])
        _mp_fn(local_rank, world_size, world_rank, args, train_model_hdf5)
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        try:
            world_size = int(os.environ['GPU_NUM_DEVICES'])
        except:
            world_size = 8
        mp.spawn(_mp_fn, args=(world_size, None, args, train_model_hdf5), nprocs=world_size, join=True)
        # _mp_fn(0, world_size, None, args, train_model_hdf5)