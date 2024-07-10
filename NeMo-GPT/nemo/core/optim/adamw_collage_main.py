# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

import torch
from torch import Tensor
from .torch_optim_optimizer import (_use_grad_for_differentiable, _default_to_fused_or_foreach, ParamsT)
from typing import List, Dict, Optional, Tuple, Union
from .torch_utils_foreach_utils import _get_fused_kernels_supported_devices
from .adamw_collage_single_tensor import _single_tensor_adamw
from .monitor_metrics import complete_norm_metrics, complete_final_ip_metrics, get_final_ratio_metrics, get_final_projected_metrics

class AdamW_collage(torch.optim.Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        weight_decay_style: str = 'compact',
        Collage: str = 'none',
        monitor_metrics: bool = False,
        amsgrad: bool = False,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if isinstance(lr, Tensor) and foreach and not capturable:
            raise ValueError("lr as a Tensor is not supported for capturable=False and foreach=True")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            foreach=foreach,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )
        self.weight_decay_style = weight_decay_style
        self.Collage = Collage
        self.monitor_metrics = monitor_metrics
        super().__init__(params, defaults)

        if fused:
            if differentiable:
                raise RuntimeError("`fused` does not support `differentiable`")
            self._step_supports_amp_scaling = True
            # TODO(crcrpar): [low prec params & their higher prec copy]
            # Suppor AMP with FP16/BF16 model params which would need
            # higher prec copy of params to do update math in higher prec to
            # alleviate the loss of information.
            fused_supported_devices = _get_fused_kernels_supported_devices()
            if not all(
                p.device.type in fused_supported_devices and
                torch.is_floating_point(p)
                for pg in self.param_groups for p in pg['params']
            ):
                raise RuntimeError("`fused=True` requires all the params to be floating point Tensors of "
                                   f"supported devices: {fused_supported_devices}.")
            if foreach:
                raise RuntimeError("`fused` and `foreach` cannot be `True` together.")

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            group.setdefault("fused", None)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
            state_values[0]["step"]
        )
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]))

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        amsgrad,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        accounted_weights,
        accounted_exp_avg_sqs,
        state_steps,
    ):
        has_complex = False
        for p in group["params"]:
            if p.grad is None:
                continue
            has_complex |= torch.is_complex(p)
            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError("AdamW does not support sparse gradients")
            grads.append(p.grad)

            state = self.state[p]
        
            # State initialization
            if len(state) == 0:
                # note(crcrpar): Deliberately host `step` on CPU if both capturable and fused are off.
                # This is because kernel launches are costly on CUDA and XLA.
                state["step"] = (
                    torch.zeros((), dtype=torch.float, device=p.device)
                    if group["capturable"] or group["fused"]
                    else torch.tensor(0.0)
                )
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                # Exponential moving average of squared gradient values
                state["exp_avg_sq"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                if self.Collage is not None:
                    state["weight_accounted"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)
                    if self.Collage == 'plus':
                        state["exp_avg_sq_accounted"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format)

                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state["max_exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            if self.Collage is not 'none':
                accounted_weights.append(state["weight_accounted"])
                if self.Collage == 'plus':
                    accounted_exp_avg_sqs.append(state["exp_avg_sq_accounted"])

            if group['amsgrad']:
                max_exp_avg_sqs.append(state["max_exp_avg_sq"])
            if group['differentiable'] and state['step'].requires_grad:
                raise RuntimeError('`requires_grad` is not supported for `step` in differentiable mode')

            # Foreach without capturable does not support a tensor lr
            if group['foreach'] and isinstance(group['lr'], Tensor) and not group['capturable']:
                raise RuntimeError('lr as a Tensor is not supported for capturable=False and foreach=True')

            state_steps.append(state["step"])
        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self.monitor_metrics:
            partial_metrics = {
                'raw_update_norm':{}, 'effective_update_norm':{}, 'param_count':{}, 'imprecision_percentage':{},
                'effective_update_ip_current_raw_update':{}, 'projected_effective_update_norm_along_current_raw_update':{}, 'projected_effective_update_norm_ratio_current_raw_update':{}, 
                'weight_accounted_norm':{},
                'weight_accounted_ip_curent_raw_update':{}, 'projected_weight_accounted_norm_along_current_raw_update':{}, 'projected_weight_accounted_norm_ratio_current_raw_update':{}, 
            }
        else:
            partial_metrics = None
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            accounted_weights = []
            accounted_exp_avg_sqs = []
            state_steps = []
            amsgrad = group["amsgrad"]
            beta1, beta2 = group["betas"]

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                amsgrad,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                accounted_weights,
                accounted_exp_avg_sqs,
                state_steps,
            )

            partial_metrics_group = adamw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                accounted_weights,
                accounted_exp_avg_sqs,
                state_steps,
                amsgrad=amsgrad,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                weight_decay_style=self.weight_decay_style,
                Collage=self.Collage,
                monitor_metrics=self.monitor_metrics,
                eps=group["eps"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
                has_complex=has_complex,
            )
            if self.monitor_metrics:
                group_name = group['name']
                if group_name not in partial_metrics['raw_update_norm']:
                    for metric_name in partial_metrics_group:
                        partial_metrics[metric_name][group_name] = partial_metrics_group[metric_name]
                else:
                    for metric_name in partial_metrics_group:
                        partial_metrics[metric_name][group_name] += partial_metrics_group[metric_name]

        if self.monitor_metrics:
            complete_norm_metrics(partial_metrics['raw_update_norm'])
            complete_norm_metrics(partial_metrics['effective_update_norm'])
            complete_final_ip_metrics(partial_metrics['effective_update_ip_current_raw_update'])
            complete_final_ip_metrics(partial_metrics['imprecision_percentage'])
            complete_final_ip_metrics(partial_metrics['param_count'])
            partial_metrics['imprecision_percentage'] = get_final_ratio_metrics(partial_metrics['imprecision_percentage'], partial_metrics['param_count'])
            partial_metrics['projected_effective_update_norm_along_current_raw_update'] = get_final_projected_metrics(partial_metrics['effective_update_ip_current_raw_update'], partial_metrics['raw_update_norm'])
            partial_metrics['projected_effective_update_norm_ratio_current_raw_update'] = get_final_ratio_metrics(partial_metrics['projected_effective_update_norm_along_current_raw_update'], partial_metrics['raw_update_norm'])
            if self.Collage is not 'none':
                complete_norm_metrics(partial_metrics['weight_accounted_norm'])
                complete_final_ip_metrics(partial_metrics['weight_accounted_ip_curent_raw_update'])
                partial_metrics['projected_weight_accounted_norm_along_current_raw_update'] = get_final_projected_metrics(partial_metrics['weight_accounted_ip_curent_raw_update'], partial_metrics['raw_update_norm'])
                partial_metrics['projected_weight_accounted_norm_ratio_current_raw_update'] = get_final_ratio_metrics(partial_metrics['projected_weight_accounted_norm_along_current_raw_update'], partial_metrics['raw_update_norm'])
        return loss, partial_metrics

def adamw(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    accounted_weights: List[Tensor],
    accounted_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    foreach: Optional[bool] = None,
    capturable: bool = False,
    differentiable: bool = False,
    fused: Optional[bool] = None,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    has_complex: bool = False,
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: Union[float, Tensor],
    weight_decay: float,
    weight_decay_style: str,
    Collage: str,
    monitor_metrics: bool,
    eps: float,
    maximize: bool,
):
    r"""Functional API that performs AdamW algorithm computation.

    See :class:`~torch.optim.AdamW` for details.
    """

    if not torch._utils.is_compiling() and not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    # Respect when the user inputs False/True for foreach or fused. We only want to change
    # the default when neither have been user-specified. Note that we default to foreach
    # and pass False to use_fused. This is not a mistake--we want to give the fused impl
    # bake-in time before making it the default, even if it is typically faster.
    if fused is None and foreach is None:
        _, foreach = _default_to_fused_or_foreach(params, differentiable, use_fused=False)
        # Do not flip on foreach for the unsupported case where lr is a Tensor and capturable=False.
        if foreach and isinstance(lr, Tensor) and not capturable:
            foreach = False
    if fused is None:
        fused = False
    if foreach is None:
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")
    if fused and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with fused optimizers")

    if fused and not torch.jit.is_scripting():
        raise NotImplementedError("_fused_adamw is removed!")
        func = _fused_adamw
    elif foreach and not torch.jit.is_scripting():
        raise NotImplementedError("_multi_tensor_adamw is removed")
        func = _multi_tensor_adamw
    else:
        func = _single_tensor_adamw

    return func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        accounted_weights,
        accounted_exp_avg_sqs,
        state_steps,
        amsgrad=amsgrad,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        weight_decay_style=weight_decay_style,
        Collage=Collage,
        monitor_metrics=monitor_metrics,
        eps=eps,
        maximize=maximize,
        capturable=capturable,
        differentiable=differentiable,
        grad_scale=grad_scale,
        found_inf=found_inf,
        has_complex=has_complex,
    )