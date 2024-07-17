# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
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

import itertools
import os
import re
import shutil
import tempfile
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterator, List, Literal, Mapping, Optional, Sized, Union

import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.cloud_io import get_filesystem
from omegaconf import OmegaConf
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks.progress.tqdm_progress import _update_n
from pytorch_lightning.loops.fetchers import _DataFetcher
from pytorch_lightning.overrides.base import _LightningModuleWrapperBase
from pytorch_lightning.plugins import ClusterEnvironment
from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO
from pytorch_lightning.plugins.precision import MixedPrecisionPlugin
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.trainer.trainer import Trainer
from torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks import noop_hook
from torch.nn.parallel import DistributedDataParallel

from nemo.collections.nlp.modules.common.megatron.module import Float16Module
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
from nemo.core.optim import MainParamsOptimizerWrapper
from nemo.utils import AppState, logging
from nemo.utils.get_rank import is_global_rank_zero
from nemo.utils.model_utils import ckpt_to_dir, inject_model_parallel_rank, uninject_model_parallel_rank

try:
    from apex.transformer.pipeline_parallel.utils import get_num_microbatches

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

try:
    from megatron.core import dist_checkpointing, parallel_state
    from megatron.core.dist_checkpointing.dict_utils import dict_list_map_outplace
    from megatron.core.dist_checkpointing.optimizer import (
        get_param_id_to_sharded_param_map,
        make_sharded_optimizer_tensor,
        optim_state_to_sharding_state,
    )
    from megatron.core.transformer.module import Float16Module as MCoreFloat16Module

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


NEMO_MEGATRON_MODEL_PARALLEL_APPSTATE_OVERRIDE = "NEMO_MEGATRON_MODEL_PARALLEL_APPSTATE_OVERRIDE"


class NLPDDPStrategy(DDPStrategy):
    """ DDP plugin for Pytorch Lightning. Needed to customize DDP for model parallel models.

    Args:
        no_ddp_communication_hook: Disable DDP communication hook when using AMP-O2
        with FP32 gradient accumulation.
    """

    def __init__(
        self,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: ClusterEnvironment = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        no_ddp_communication_hook: bool = False,
        **kwargs: Union[Any, Dict[str, Any]],
    ) -> None:
        if not HAVE_APEX:
            raise ImportError(
                "Apex was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )

        if not HAVE_MEGATRON_CORE:
            raise ImportError(
                "megatron-core was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )
        super().__init__(parallel_devices, cluster_environment, checkpoint_io, **kwargs)

        self.no_ddp_communication_hook = no_ddp_communication_hook

    def setup(self, trainer: "pl.Trainer") -> None:
        """
        Override setup() of DDPStrategy to avoid _sync_module_states(self.model) during eval as it can cause PP > 1 to hang
        due to assumption in DDPStrategy class that the same model is replicated across GPUs
        """
        trainer_fn = trainer.state.fn
        if trainer_fn == TrainerFn.FITTING:
            super().setup(trainer)
        else:
            assert self.accelerator is not None
            self.accelerator.setup(trainer)

            # move the model to the correct device
            self.model_to_device()
            self.setup_precision_plugin()
            assert self.model is not None

    def setup_distributed(self, global_rank: int = None, world_size: int = None) -> None:
        # call PTL init ddp
        super().setup_distributed()

        # init model parallel if needed
        if not parallel_state.model_parallel_is_initialized():
            app_state = AppState()

            if app_state.model_parallel_size is not None:
                self.init_model_parallel(app_state.global_rank, app_state.world_size)

    def configure_ddp(self):
        """ Override LightningModule ddp if using model parallel.
            Sets find_unused_parameters to False to use activation-checkpoint-recomputation.
        """

        if (hasattr(self.model, 'megatron_amp_O2') and self.model.megatron_amp_O2) or (
            hasattr(self.model, 'with_distributed_adam') and self.model.with_distributed_adam
        ):
            # do not use DDP if using megatron amp O2 or distributed optimizer
            self._model = _LightningModuleWrapperBase(self.model)
        else:
            app_state = AppState()

            if app_state.model_parallel_size is not None:

                logging.info(f"Configuring DDP for model parallelism.")

                # With model parallelism, multiple GPUs form a large "logical GPU"
                # this means that data parallel groups span multiple GPUs
                # and are non-trivial
                # TODO: for megatron-lm self.model is a list
                # Removing self.pre_configure_ddp() as DDP's 'find_unused_parameters' now defaults
                # to False in PTL 2.0 and hence pre_configure_ddp() is removed in ddp.py
                # self.pre_configure_ddp()
                # device_ids = self.determine_ddp_device_ids()
                self._model = DistributedDataParallel(
                    _LightningModuleWrapperBase(self.model),
                    process_group=parallel_state.get_data_parallel_group(),
                    **self._ddp_kwargs,
                )

                if self.no_ddp_communication_hook:
                    # When using custom gradient accumulation and allreduce, disable
                    # DDP communication hook that works on the gradient bucket.
                    # Instead, use the custom gradient function and communication hook,
                    # which is defined in the master optimizer wrapper.
                    self._model.require_backward_grad_sync = False
                    self._model.register_comm_hook(None, noop_hook)

            else:
                super().configure_ddp()

    def init_model_parallel(self, global_rank: int, world_size: int) -> None:
        """ Initializes Megatron-LM model parallel if using model parallelism.

        Args:
            global_rank (int): the global process index.
            world_size (int): the total number of GPUs, num_nodes * num_devices
            is_slurm_managing_tasks (bool, optional): is the cluster managed by SLURM.
        """
        app_state = AppState()

        # we initialize megatron-lm model parallel and data parallel groups
        # after initializing DDP with PTL.
        if app_state.model_parallel_size is not None:
            # destroy groups in case they have already been created
            # this happens with multiple calls to trainer.test for example
            parallel_state.destroy_model_parallel()
            if torch.distributed.is_initialized():
                parallel_state.initialize_model_parallel(
                    tensor_model_parallel_size=app_state.tensor_model_parallel_size,
                    pipeline_model_parallel_size=app_state.pipeline_model_parallel_size,
                    virtual_pipeline_model_parallel_size=app_state.virtual_pipeline_model_parallel_size,
                    pipeline_model_parallel_split_rank=app_state.pipeline_model_parallel_split_rank,
                )

                # assert that fake tp and pp rank match after model parallel init
                assert app_state.tensor_model_parallel_rank == parallel_state.get_tensor_model_parallel_rank()
                assert app_state.pipeline_model_parallel_rank == parallel_state.get_pipeline_model_parallel_rank()

                app_state.tensor_model_parallel_group = parallel_state.get_tensor_model_parallel_group()
                app_state.data_parallel_group = parallel_state.get_data_parallel_group()
                app_state.data_parallel_rank = parallel_state.get_data_parallel_rank()
                app_state.data_parallel_size = parallel_state.get_data_parallel_world_size()
                app_state.pipeline_model_parallel_group = parallel_state.get_pipeline_model_parallel_group()

                # create MPI process group for UCX-based communication APIs
                if app_state.init_mpi_proc_group:
                    torch.distributed.new_group(backend='mpi')

    def optimizer_sharded_state_dict(self):
        """
        Sharded state dictionary for an MainParamsOptimizerWrapper.
        Used to save and load the optimizer state when training with distributed_checkpoint.
        Returns:
            dict: The sharded state dictionary for the optimizer
        Raises:
            ValueError: If a parameter ID does not match any model sharded parameter.
        """

        optimizer = self.lightning_module.optimizers(use_pl_optimizer=False)  # MainParamsOptimizerWrapper

        model_sharded_state_dict = self.lightning_module.sharded_state_dict()

        # remove _extra_state
        model_sharded_state_dict = {
            key: value for key, value in model_sharded_state_dict.items() if not key.endswith('_extra_state')
        }

        if not isinstance(optimizer, MainParamsOptimizerWrapper):
            return optimizer.sharded_state_dict(model_sharded_state_dict)

        optimizer_state_dict = optimizer.state_dict()

        id_to_sharded_param_map = get_param_id_to_sharded_param_map(
            model_sharded_state_dict=model_sharded_state_dict,
            optim_params_iter=itertools.chain.from_iterable(g for g in optimizer.float16_groups),
        )

        # Convert fp32_from_fp16_params
        assert len(optimizer_state_dict['fp32_from_fp16_params']) == len(
            optimizer_state_dict['optimizer']['param_groups']
        )

        def get_safe(param_id):
            try:
                return id_to_sharded_param_map[param_id]
            except KeyError as e:
                raise ValueError(f'Param id {param_id} does not match any model sharded param') from e

        optimizer_state_dict['fp32_from_fp16_params'] = [
            [
                make_sharded_optimizer_tensor(get_safe(param_id), fp32_param, prefix=f'optimizer.state.fp32_param')
                for param_id, fp32_param in zip(state_group['params'], fp32_group)
            ]
            for fp32_group, state_group in zip(
                optimizer_state_dict['fp32_from_fp16_params'], optimizer_state_dict['optimizer']['param_groups']
            )
        ]

        # Convert state
        optim_state_to_sharding_state(optimizer_state_dict['optimizer'], id_to_sharded_param_map,
            exclude_keys=("step",))

        return optimizer_state_dict

    def save_checkpoint(
        self, checkpoint: Dict[str, Any], filepath: Union[str, Path], storage_options: Optional[Any] = None
    ) -> None:
        app_state = AppState()
        """ PTL method which we override to accomodate distributed checkpoints and 
            the legacy model parallel checkpoints.

            When using megatron core, the distributed checkpointing library expects save functions to be
            called on every rank and internally does the rank checking.
        """

        # check if using distributed checkpointing
        if (
            hasattr(self.lightning_module, 'sharded_state_dict')
            and self.lightning_module.sharded_state_dict() is not None
            and not self._lightning_module.use_legacy_ptl_save
        ):
            # converts the optimizer states to their sharded equivalents
            checkpoint['optimizer_states'] = [self.optimizer_sharded_state_dict()]

            # dist_checkpointing expects a directory so we will name the directory
            # using the path with the file extension removed
            checkpoint_dir = ckpt_to_dir(filepath)

            fs = get_filesystem(checkpoint_dir)
            if fs.isdir(checkpoint_dir) and dist_checkpointing.check_is_distributed_checkpoint(checkpoint_dir):
                logging.info(f'Distributed checkpoint at path {checkpoint_dir} already exists, skipping saving')
                return

            if is_global_rank_zero():
                fs.makedirs(checkpoint_dir, exist_ok=True)

            # remove device state_dict
            checkpoint['state_dict'] = OrderedDict([])

            dist_checkpointing.save(sharded_state_dict=checkpoint, checkpoint_dir=checkpoint_dir)
        else:
            # PTL override to accomodate model parallel checkpoints
            filepath = inject_model_parallel_rank(filepath)
            if self.is_global_zero or app_state.data_parallel_rank == 0:
                self.checkpoint_io.save_checkpoint(checkpoint, filepath, storage_options=storage_options)

    def load_model_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        # if using distributed checkpointing, the state dict logic is at the model level
        if (
            hasattr(self.lightning_module, 'sharded_state_dict')
            and self.lightning_module.sharded_state_dict() is not None
        ):
            return

        # legacy state dict logic, does not use megatron core
        else:

            # Release strict state dict matching when using Megatron AMP-O2 to skip matching
            # half-precision module wrapper module.
            # TODO: Refactor this to be more generic.
            model_key = None
            model_attr = None
            if hasattr(self.lightning_module, 'model'):
                model_key = 'model'
                model_attr = self.lightning_module.model
            elif hasattr(self.lightning_module, 'enc_dec_model'):
                model_key = 'enc_dec_model'
                model_attr = self.lightning_module.enc_dec_model
            if model_key is not None:
                if isinstance(model_attr, Float16Module) or isinstance(model_attr, MCoreFloat16Module):
                    new_state_dict = {}
                    for key in checkpoint['state_dict'].keys():
                        new_key = key.replace(f'{model_key}.', f'{model_key}.module.', 1)
                        new_state_dict[new_key] = checkpoint['state_dict'][key]
                    checkpoint['state_dict'] = new_state_dict

            self.lightning_module.load_state_dict(checkpoint["state_dict"])

    def _fix_tensors_device(self, ckpt: Dict) -> Dict:
        """ Ensure checkpoint tensors are on the correct device."""
        assert torch.cuda.is_initialized(), (torch.cuda.is_available(), torch.cuda.is_initialized())
        cur_dev = torch.device("cuda", index=torch.cuda.current_device())

        def _fix_device(t):
            if isinstance(t, torch.Tensor) and t.is_cuda and t.device != cur_dev:
                t = t.to(cur_dev)
            return t

        return dict_list_map_outplace(_fix_device, ckpt)

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """ PTL method which we override to integrate distributed checkpoints for model parallel models.
            In order to load distributed checkpoints we need to provide the sharded_state_dict to 
            the distributed load function. We get the sharded_state_dict from self.lightning_module
            which makes it convenient to have the loading logic happen at the strategy level.
        """

        fs = get_filesystem(checkpoint_path)

        # Check if using distributed checkpointing
        if (
            hasattr(self.lightning_module, 'sharded_state_dict')
            and self.lightning_module.sharded_state_dict() is not None
            and not self._lightning_module.use_legacy_ptl_save
        ):

            # Distributed checkpoints must be directories.
            if not fs.isdir(checkpoint_path):
                raise ValueError(f'Distributed checkpoints should be a directory. Found: {checkpoint_path}.')

            sharded_state_dict = self.lightning_module.sharded_state_dict()

            checkpoint = {}

            # after dist_checkpointing.load, sharded tensors will be replaced with tensors
            checkpoint['state_dict'] = sharded_state_dict
            checkpoint['optimizer_states'] = [self.optimizer_sharded_state_dict()]

            checkpoint = dist_checkpointing.load(sharded_state_dict=checkpoint, checkpoint_dir=checkpoint_path)

            checkpoint = self._fix_tensors_device(checkpoint)

            return checkpoint

        # Legacy model parallel checkpointing logic, does not use megatron core
        else:
            # Try to read the checkpoint at `path`. If not exist, do not restore checkpoint.
            checkpoint_path = inject_model_parallel_rank(checkpoint_path)
            if not fs.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint at {checkpoint_path} not found. Aborting training.")
            torch.cuda.empty_cache()
            return self.checkpoint_io.load_checkpoint(checkpoint_path)

    def remove_checkpoint(self, filepath: Union[str, Path]) -> None:
        # check if filepath is a distributed checkpoint
        if (
            hasattr(self.lightning_module, 'sharded_state_dict')
            and self.lightning_module.sharded_state_dict() is not None
        ):
            if self.is_global_zero:
                shutil.rmtree(ckpt_to_dir(filepath))

        # legacy checkpoint logic, does not use megatron core
        else:
            app_state = AppState()
            # PTL override to accomodate model parallel checkpoints
            filepath = inject_model_parallel_rank(filepath)
            if self.is_global_zero or app_state.data_parallel_rank == 0:
                logging.info(f'Removing checkpoint: {filepath}')
                self.checkpoint_io.remove_checkpoint(filepath)

    @property
    def distributed_sampler_kwargs(self):
        app_state = AppState()
        if app_state.model_parallel_size is not None:
            # When using model parallel, data parallel groups are non-trivial and they
            # correspond to the logical GPUs. This means that the GPUs that form a
            # single logical GPU all need to get the same batch of data.
            distributed_sampler_kwargs = dict(
                num_replicas=app_state.data_parallel_size, rank=app_state.data_parallel_rank
            )
            return distributed_sampler_kwargs

        else:
            return super(NLPDDPStrategy, self).distributed_sampler_kwargs

    @property
    def restore_checkpoint_after_setup(self) -> bool:
        """ This needs to be True for distributed checkpointing because
            we require the model to have configured the optimizer before 
            deserializing the checkpoint.
        """
        return True


class NLPDDPStrategyNotebook(NLPDDPStrategy):
    """ Version of NLPDDPStrategy to be used in a Jupyter Notebook
    A large portion of Megatron code has DDP dependency, so it has been necessary to use NLPDDPStrategy even for
    single-GPU training (e.g. in a Jupyter notebook)
    A PTL 2.0 changes has prevented DDPStrategy to be used in a notebook.
    This version of NLPDDPStrategy enables megatron training in a notebook in PTL 2.0.
    """

    def _configure_launcher(self):
        self._launcher = None


class NLPSaveRestoreConnector(SaveRestoreConnector):
    def __init__(self) -> None:
        if not HAVE_APEX:
            logging.warning(
                "Apex was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/apex\n"
                "Megatron-based models require Apex to function correctly."
            )
            # raise ImportError(
            #    "Apex was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            # )
        if not HAVE_MEGATRON_CORE:
            logging.warning(
                "megatron-core was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )
        super().__init__()

    def save_to(self, model, save_path: str):
        app_state = AppState()

        # Check if using distributed checkpointing
        dist_ckpt = hasattr(model, 'sharded_state_dict') and model.sharded_state_dict() is not None

        dist_ckpt_dir = None

        if (app_state.model_parallel_size is not None and app_state.model_parallel_size > 1) or dist_ckpt:

            dir_name = os.path.dirname(save_path)

            # dist ckpt calls save on every rank
            if dist_ckpt:
                # model weights is a directory
                dist_ckpt_dir = ckpt_to_dir(os.path.join(dir_name, self.model_weights_ckpt))
                fs = get_filesystem(dist_ckpt_dir)

                if fs.isdir(dist_ckpt_dir) and dist_checkpointing.check_is_distributed_checkpoint(dist_ckpt_dir):
                    logging.info(f'Distributed checkpoint at path {dist_ckpt_dir} already exists, skipping saving')
                else:
                    if is_global_rank_zero():
                        fs.makedirs(dist_ckpt_dir, exist_ok=True)

                    sharded_state_dict = model.sharded_state_dict()
                    # dist checkpoint needs torch.distributed to save the checkpoint
                    if parallel_state.is_unitialized():

                        def dummy():
                            return

                        if model.trainer.strategy.launcher is not None:
                            model.trainer.strategy.launcher.launch(dummy, trainer=model.trainer)
                        model.trainer.strategy.setup_environment()
                    dist_checkpointing.save(sharded_state_dict=sharded_state_dict, checkpoint_dir=dist_ckpt_dir)

            else:

                # first we save the weights for each model parallel rank
                if app_state.data_parallel_rank == 0:
                    if app_state.pipeline_model_parallel_size == 1:
                        mp_model_weights = os.path.join(
                            dir_name, f'mp_rank_{app_state.tensor_model_parallel_rank:02d}_' + self.model_weights_ckpt
                        )
                    else:
                        mp_model_weights = os.path.join(
                            dir_name,
                            f'tp_rank_{app_state.tensor_model_parallel_rank:02d}_pp_rank_{app_state.pipeline_model_parallel_rank:03d}_'
                            + self.model_weights_ckpt,
                        )

                    self._save_state_dict_to_disk(model.state_dict(), mp_model_weights)

            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            # create nemo file from folder with all mp_ranks checkpoints
            if (
                app_state.pipeline_model_parallel_rank == 0
                and app_state.tensor_model_parallel_rank == 0
                and app_state.data_parallel_rank == 0
            ):
                with tempfile.TemporaryDirectory() as tmpdir:

                    if dist_ckpt:
                        shutil.move(str(dist_ckpt_dir), tmpdir)

                    elif app_state.pipeline_model_parallel_size == 1:
                        # move weights to the tmpdir
                        for tp_rank in range(app_state.tensor_model_parallel_size):
                            os.makedirs(os.path.join(tmpdir, f'mp_rank_{tp_rank:02d}'))
                            mp_model_weights = os.path.join(
                                dir_name, f'mp_rank_{tp_rank:02d}_' + self.model_weights_ckpt
                            )
                            shutil.move(
                                mp_model_weights,
                                os.path.join(tmpdir, f'mp_rank_{tp_rank:02d}', self.model_weights_ckpt),
                            )
                    else:
                        # move weights to the tmpdir
                        for tp_rank, pp_rank in itertools.product(
                            range(app_state.tensor_model_parallel_size), range(app_state.pipeline_model_parallel_size),
                        ):
                            os.makedirs(os.path.join(tmpdir, f'tp_rank_{tp_rank:02d}_pp_rank_{pp_rank:03d}'))
                            mp_model_weights = os.path.join(
                                dir_name, f'tp_rank_{tp_rank:02d}_pp_rank_{pp_rank:03d}_' + self.model_weights_ckpt
                            )
                            shutil.move(
                                mp_model_weights,
                                os.path.join(
                                    tmpdir, f'tp_rank_{tp_rank:02d}_pp_rank_{pp_rank:03d}', self.model_weights_ckpt
                                ),
                            )

                    # create config and artifacts in tmpdir
                    config_yaml = os.path.join(tmpdir, self.model_config_yaml)
                    model.to_config_file(path2yaml_file=config_yaml)
                    if hasattr(model, 'artifacts') and model.artifacts is not None:
                        self._handle_artifacts(model, nemo_file_folder=tmpdir)
                        self._update_artifact_paths(model, path2yaml_file=config_yaml)

                    # create tar file
                    self._make_nemo_file_from_folder(save_path, tmpdir)

        else:
            return super().save_to(model, save_path)

    def modify_state_dict(self, conf, state_dict):
        if conf.get('megatron_legacy', False):
            new_state_dict = {}
            for key in state_dict.keys():
                new_key = key.replace('bert_model.language_model', 'bert_model.model.language_model')
                new_key = new_key.replace('transformer', 'encoder')
                new_key = new_key.replace('.attention.', '.self_attention.')
                new_state_dict[new_key] = state_dict[key]
            state_dict = new_state_dict

        if conf.get('megatron_amp_O2', False):
            new_state_dict = {}
            for key in state_dict.keys():
                new_key = key.replace('model.', 'model.module.', 1)
                new_state_dict[new_key] = state_dict[key]
            state_dict = new_state_dict

        # compatibility for inductor in inference
        if not conf.get('inductor', False):
            new_state_dict = {}
            for key in state_dict.keys():
                new_key = key.replace('._orig_mod', '', 1)
                new_state_dict[new_key] = state_dict[key]
            state_dict = new_state_dict

        # Modify state key for Dreambooth inference
        if (
            conf.get('target')
            == 'nemo.collections.multimodal.models.stable_diffusion.ldm.ddpm.MegatronLatentDiffusion'
        ):
            new_state_dict = {}
            for key in state_dict.keys():
                new_key = key.replace('unet', 'model.diffusion_model')
                new_key = new_key.replace('vae', 'first_stage_model')
                new_key = new_key.replace('text_encoder', 'cond_stage_model')
                new_key = new_key.replace('.noise_scheduler', '')
                new_state_dict[new_key] = state_dict[key]
            state_dict = new_state_dict

        loaded_keys = state_dict.keys()
        if 'model.model.diffusion_model.input_blocks.1.0.in_layers.2.weight' in loaded_keys:
            new_state_dict = {}
            # GroupNormOpt fuses activation function to one layer, thus the indexing of weights are shifted for following
            def should_process(key):
                base_str = "model.model.diffusion_model."
                blocks = ["input_blocks", "middle_block", "output_blocks"]
                for block in blocks:
                    for layer_type in ["in_layers", "out_layers"]:
                        for index in [2, 3]:  # The layers index.
                            for param in ["weight", "bias"]:
                                if block == 'middle_block':
                                    for num in [0, 2]:
                                        template = f"{base_str}{block}.{num}.{layer_type}.{index}.{param}"
                                        if key == template:
                                            return True
                                else:
                                    for num in range(12):  # 12 blocks, adjust as needed.
                                        template = f"{base_str}{block}.{num}.0.{layer_type}.{index}.{param}"
                                        if key == template:
                                            return True
                return False

            for key_ in state_dict.keys():
                if key_ == "model.cond_stage_model.transformer.text_model.embeddings.position_ids":
                    continue
                if should_process(key_):
                    s = key_.split('.')
                    idx = int(s[-2])
                    new_key_ = ".".join(s[:-2] + [str(int(idx - 1))] + [s[-1]])
                    new_state_dict[new_key_] = state_dict[key_]
                else:
                    new_state_dict[key_] = state_dict[key_]
            state_dict = new_state_dict

        return state_dict

    def _load_state_dict_from_disk(self, model_weights, map_location=None):
        # if model_weights with the extension removed is a directory, we assume it is a distributed checkpoint
        # we need to defer loading the state dict so we return None
        uninject_model_weights = uninject_model_parallel_rank(model_weights)

        # legacy model_weights will have mp rank injected
        if os.path.isfile(model_weights):
            return super()._load_state_dict_from_disk(model_weights, map_location)

        # dist checkpoint will be a dir
        elif os.path.isdir(os.path.splitext(uninject_model_weights)[0]):
            return None
        else:
            raise ValueError(f'Expected {model_weights} to be a file or directory.')

    def restore_from(
        self,
        calling_cls,
        restore_path: str,
        override_config_path: Optional[Union[OmegaConf, str]] = None,
        map_location: Optional[torch.device] = None,
        strict: bool = True,
        return_config: bool = False,
        trainer: Trainer = None,
    ):
        """
        Restores model instance (weights and configuration) into .nemo file

        Args:
            restore_path: path to .nemo file from which model should be instantiated
            override_config_path: path to a yaml config that will override the internal
                config file or an OmegaConf / DictConfig object representing the model config.
            map_location: Optional torch.device() to map the instantiated model to a device.
                By default (None), it will select a GPU if available, falling back to CPU otherwise.
            strict: Passed to load_state_dict. By default True
            return_config: If set to true, will return just the underlying config of the restored
                model as an OmegaConf DictConfig object without instantiating the model.

        Example:
            ```
            model = nemo.collections.nlp.models.TextClassification.restore_from('asr.nemo')
            assert isinstance(model, nemo.collections.nlp.models.TextClassification)
            ```

        Returns:
            An instance of type cls or its underlying config (if return_config is set).
        """

        # Get path where the command is executed - the artifacts will be "retrieved" there
        # (original .nemo behavior)
        loaded_params = super().load_config_and_state_dict(
            calling_cls, restore_path, override_config_path, map_location, strict, return_config, trainer,
        )
        if not isinstance(loaded_params, tuple) or return_config is True:
            return loaded_params
        conf, instance, state_dict = loaded_params

        # if we're using dist checkpointing then state_dict will be None
        if state_dict is None:
            # dist checkpointing needs torch.distributed to load the checkpoint
            if parallel_state.is_unitialized():

                def dummy():
                    return

                if trainer.strategy.launcher is not None:
                    trainer.strategy.launcher.launch(dummy, trainer=trainer)
                trainer.strategy.setup_environment()

            with tempfile.TemporaryDirectory() as tmpdir:
                # Check if self.model_extracted_dir is set, and is a valid path
                if self.model_extracted_dir is not None and os.path.isdir(self.model_extracted_dir):
                    # Log that NeMo will use the provided `model_extracted_dir`
                    logging.info(
                        f"Restoration will occur within pre-extracted directory : " f"`{self.model_extracted_dir}`."
                    )

                    # Override `tmpdir` above with the pre-extracted `model_extracted_dir`
                    tmpdir = self.model_extracted_dir

                else:
                    # Extract the nemo file into the temporary directory
                    self._unpack_nemo_file(
                        path2file=restore_path, out_folder=tmpdir, extract_config_only=return_config is True
                    )
                checkpoint = {}
                sharded_state_dict = instance.sharded_state_dict()
                checkpoint['state_dict'] = sharded_state_dict
                # remove model weights extension
                tmp_model_weights_ckpt = os.path.join(tmpdir, self.model_weights_ckpt)
                tmp_model_weights_dir = os.path.splitext(tmp_model_weights_ckpt)[0]
                assert os.path.isdir(tmp_model_weights_dir), f'Expected {tmp_model_weights_dir} to be a directory.'
                checkpoint = dist_checkpointing.load(
                    sharded_state_dict=checkpoint, checkpoint_dir=tmp_model_weights_dir
                )
                instance.on_load_checkpoint(checkpoint)
                if hasattr(instance, 'setup_transformer_engine_tp_groups'):
                    instance.setup_transformer_engine_tp_groups()

        else:
            state_dict = self.modify_state_dict(conf, state_dict)
            super().load_instance_with_state_dict(instance, state_dict, strict)
        logging.info(f'Model {instance.__class__.__name__} was successfully restored from {restore_path}.')
        return instance


class PipelineMixedPrecisionPlugin(MixedPrecisionPlugin):
    """ Overrides PTL autocasting to not wrap training/val/test_step.
        We do this because we have the megatron-core fwd/bwd functions in training_step.
        This means .backward is being called in training_step so we do not want the whole
        step wrapped in autocast.

        We instead wrap the fwd_output_and_loss_func that is passed to the megatron-core fwd/bwd functions.
    """

    def __init__(
        self,
        precision: Literal["16-mixed", "bf16-mixed"],
        device: str,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> None:
        super().__init__(precision, device, scaler=scaler)
        dtype = None
        # MixedPrecisionPlugin class in PTL >= 2.0 takes only "16-mixed" or "bf16-mixed" for precision arg
        if precision == '16-mixed':
            dtype = torch.float16
        elif precision == 'bf16-mixed':
            dtype = torch.bfloat16

        torch.set_autocast_gpu_dtype(dtype)

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """Have the PTL context manager do nothing."""
        yield


class GradScaler(torch.cuda.amp.GradScaler):
    """
    Gradient sclaer for model-parallel inf check. The inf in gradients are checked across tensor-parallel
    ranks in (1) executing optimizer step and (2) gradient scaler update.

    """

    def __init__(
        self,
        init_scale=2.0 ** 16,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
        enabled=True,
        hysteresis=1,
    ):
        super().__init__(
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            enabled=enabled,
        )
        self.optimizer_update_skipped: Optional[bool] = None
        self.hysteresis = hysteresis
        self._hysteresis_tracker = self.hysteresis

    def _unscale_grads_(self, optimizer, *args):
        if getattr(optimizer, "_custom_amp_unscale_grads", False):
            return optimizer.unscale_grads(*args)
        else:
            return super()._unscale_grads_(optimizer, *args)

    def _maybe_opt_step(self, optimizer, optimizer_state, *args, **kwargs):
        retval = None
        found_inf = torch.cuda.FloatTensor([sum(v.item() for v in optimizer_state["found_inf_per_device"].values())])

        # Update across all model parallel instances.
        torch.distributed.all_reduce(
            found_inf, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_model_parallel_group()
        )

        if found_inf.item() == 0:
            retval = optimizer.step(*args, **kwargs)
            self.optimizer_update_skipped = False
        else:
            self.optimizer_update_skipped = True
        return retval

    def update(self, new_scale=None):
        """
        Updates to native grad scaler update function.
        1. Check inf across model-parallel ranks.
        2. Update hysteresis tracker.
        3. Apply hysteresis to grad scale update.
        """
        if not self._enabled:
            return

        _scale, _growth_tracker = self._check_scale_growth_tracker("update")

        if new_scale is not None:
            # Accept a new user-defined scale.
            if isinstance(new_scale, float):
                self._scale.fill_(new_scale)  # type: ignore[union-attr]
            else:
                reason = "new_scale should be a float or a 1-element torch.cuda.FloatTensor with requires_grad=False."
                assert isinstance(new_scale, torch.cuda.FloatTensor), reason  # type: ignore[attr-defined]
                assert new_scale.numel() == 1, reason
                assert new_scale.requires_grad is False, reason
                self._scale.copy_(new_scale)  # type: ignore[union-attr]
        else:
            # Consume shared inf/nan data collected from optimizers to update the scale.
            # If all found_inf tensors are on the same device as self._scale, this operation is asynchronous.
            found_infs = [
                found_inf.to(device=_scale.device, non_blocking=True)
                for state in self._per_optimizer_states.values()
                for found_inf in state["found_inf_per_device"].values()
            ]

            assert len(found_infs) > 0, "No inf checks were recorded prior to update."

            found_inf_combined = found_infs[0]

            # Update across all model parallel instances.
            torch.distributed.all_reduce(
                found_inf_combined, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_model_parallel_group()
            )

            if len(found_infs) > 1:
                for i in range(1, len(found_infs)):
                    found_inf = found_infs[i]
                    # Update across all model parallel instances.
                    torch.distributed.all_reduce(
                        found_inf, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_model_parallel_group()
                    )
                    found_inf_combined += found_inf

            if found_inf_combined > 0:
                self._hysteresis_tracker -= 1
                if self._hysteresis_tracker <= 0:
                    # When hysteresis becomes zero, follow the native grad scale update rule.
                    # Increase scale and reset growth tracker
                    torch._amp_update_scale_(
                        _scale,
                        _growth_tracker,
                        found_inf_combined,
                        self._growth_factor,
                        self._backoff_factor,
                        self._growth_interval,
                    )
                else:
                    # Only reset the growth tracker when hysteresis is larger than zero
                    _growth_tracker.fill_(0.0)
            else:
                # When no inf found, follow the native grad scale update rule.
                # Increment growth_tracker, update scale when growth tracker reaches the interval, and
                # reset the hysteresis tracker.
                torch._amp_update_scale_(
                    _scale,
                    _growth_tracker,
                    found_inf_combined,
                    self._growth_factor,
                    self._backoff_factor,
                    self._growth_interval,
                )
                self._hysteresis_tracker = self.hysteresis

        # To prepare for next iteration, clear the data collected from optimizers this iteration.
        self._per_optimizer_states = defaultdict(torch.cuda.amp.grad_scaler._refresh_per_optimizer_state)

    def state_dict(self):
        """
        Add hysteresis_tracker to the native functions' state_dict
        """
        return (
            {
                "scale": self.get_scale(),
                "growth_factor": self._growth_factor,
                "backoff_factor": self._backoff_factor,
                "growth_interval": self._growth_interval,
                "_growth_tracker": self._get_growth_tracker(),
                "_hysteresis_tracker": self._hysteresis_tracker,
            }
            if self._enabled
            else {}
        )

    def load_state_dict(self, state_dict):
        """
        Load hysteresis_tracker in addition to the state dict of the native function
        """
        if not self._enabled:
            return

        if len(state_dict) == 0:
            raise RuntimeError(
                "The source state dict is empty, possibly because it was saved "
                "from a disabled instance of GradScaler."
            )

        self._init_scale = state_dict["scale"]
        if self._scale is not None:
            self._scale.fill_(state_dict["scale"])
        self._growth_factor = state_dict["growth_factor"]
        self._backoff_factor = state_dict["backoff_factor"]
        self._growth_interval = state_dict["growth_interval"]
        self._init_growth_tracker = state_dict["_growth_tracker"]
        if self._growth_tracker is not None:
            self._growth_tracker.fill_(state_dict["_growth_tracker"])
        if "_hysterisis_tracker" in state_dict:
            self._hysteresis_tracker = state_dict["_hysterisis_tracker"]
        else:
            self._hysteresis_tracker = 1


class MegatronHalfPrecisionPlugin(MixedPrecisionPlugin):
    """
    Plugin for Half (FP16 and BF16) precision training.
    This plugin assumes the use of the optimizer with master parameters (fp32).
    This plugin uses half-precision at all operators in the model so need of input precision
    at each layer operator.

    Args:
        precision: Whether to use ``torch.float16`` (``16``) or ``torch.bfloat16`` (``'bf16'``).
        device: The device for ``torch.autocast``.
        scaler: An optional :class:`torch.cuda.amp.GradScaler` to use.
    """

    def __init__(
        self, precision: Union[str, int], device: str, scaler: Optional[torch.cuda.amp.GradScaler] = None
    ) -> None:
        super().__init__(precision, device, scaler)
        dtype = None
        # MixedPrecisionPlugin class in PTL >= 2.0 takes only "16-mixed" or "bf16-mixed" for precision arg
        if precision == "16-mixed":
            dtype = torch.float16
        elif precision == "bf16-mixed":
            dtype = torch.bfloat16

        torch.set_autocast_gpu_dtype(dtype)

    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        model: Union["pl.LightningModule", torch.nn.Module],
        closure: Callable[[], Any],
        **kwargs: Any,
    ) -> None:
        assert isinstance(
            optimizer, MainParamsOptimizerWrapper
        ), "MegatronHalfPrecisionPlugin supports only the optimizer with master parameters"

        if self.scaler is None:
            assert optimizer.fp32_grad_accumulation, "BF16 uses FP32 grad accumulation"
            _ = closure()
            self._after_closure(model, optimizer)
            return optimizer.step(**kwargs)

        assert not optimizer.fp32_grad_accumulation, "FP16 uses FP16 grad accumulation"
        closure_result = closure()

        # TODO: Add an option for merged all-reduce

        # cast fp16 grads to fp32 and copy to main grads, which are used for unscale and param update
        optimizer.copy_model_grads_to_main_grads()
        # `unscale` after the closure is executed but before the `on_before_optimizer_step` hook.
        # unscale main (fp32) gradients
        self.scaler.unscale_(optimizer)
        self._after_closure(model, optimizer)
        skipped_backward = closure_result is None
        # in manual optimization, the closure does not return a value
        if not isinstance(model, pl.LightningModule) or not model.automatic_optimization or not skipped_backward:
            # note: the scaler will skip the `optimizer.step` if nonfinite gradients are found
            self.scaler.step(optimizer, **kwargs)
            self.scaler.update()

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """ No explicit precision casting. Inputs are supposed to be manually casted """
        try:
            yield
        finally:
            pass


class GlobalBatchDataFetcher(_DataFetcher):
    """ Overrides PTL DataFetcher. Used to fetch global batches."""

    def __init__(self, prefetch_batches: int = 0, store_on_device: bool = False) -> None:

        if not HAVE_APEX:
            logging.warning("Apex was not found. Using model parallel or megatron models will error out.")
        if not HAVE_MEGATRON_CORE:
            logging.warning("Megatron-core was not found. Using model parallel or megatron models will error out..")

        super().__init__(prefetch_batches=prefetch_batches, store_on_device=store_on_device)

    def _fetch_next_batch(self, iterator: Iterator) -> None:
        start_output = self.on_fetch_start()
        batch = [next(iterator) for _ in range(get_num_microbatches())]
        self.fetched += 1
        if not self.prefetch_batches and self._has_len:
            # when we don't prefetch but the dataloader is sized, we use the length for `done`
            dataloader = self.dataloader
            assert isinstance(dataloader, Sized)  # `_has_len` is True
            self.done = self.fetched >= len(dataloader)
        self.on_fetch_end(batch, start_output)


class CustomProgressBar(TQDMProgressBar):
    """
    Add CustomProgressBar to remove 's/it' and display progress per step instead of per microbatch
    for megatron models
    """

    def get_current_epoch_step(self, trainer):
        """
        Get the value of step within an epoch
        """
        return trainer.fit_loop.epoch_loop.automatic_optimization.optim_progress.optimizer.step.current.completed

    def init_train_tqdm(self):
        """
        Override bar_format to not have 's/it'
        """
        self.bar = super().init_train_tqdm()
        self.bar.bar_format = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]"
        return self.bar

    def on_train_epoch_start(self, trainer, *_):
        if trainer.max_steps > 0 and (trainer.ckpt_path is not None):
            # while resuming from a ckpt use trainer.max_steps as the total for progress bar as trainer.num_training_batches
            # is truncated to max_steps - step being resumed at
            num_training_batches = trainer.max_steps
        else:
            num_training_batches = trainer.num_training_batches
        self.train_progress_bar.reset(num_training_batches)
        self.train_progress_bar.initial = 0
        self.train_progress_bar.set_description(f"Epoch {trainer.current_epoch}")

    def on_train_batch_end(self, trainer, pl_module, *_, **__):
        """
        Override parent class on_train_batch_end to update progress bar per global batch instead of per microbatch
        """
        n = self.get_current_epoch_step(trainer)
        if self._should_update(n, self.train_progress_bar.total):
            _update_n(self.train_progress_bar, n)
            self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))
