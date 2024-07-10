# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

from typing import Callable, Iterable, Tuple
import math
import torch
from torch import nn
import warnings
from transformers.utils.versions import require_version
from .monitor_metrics import (Grow_Exp, Grow_Exp_, Grow_Exp_addcmul_, MultMC_,
                     complete_norm_metrics, complete_final_ip_metrics, get_final_ratio_metrics, 
                     get_final_projected_metrics)

class AdamW_collage(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        weight_decay_style: str = 'compact',
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
        upcast_optim_states: bool = False,
        enable_master_weight: bool = False,
        Collage: str = 'none',
        monitor_metrics: bool = False,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "weight_decay_style": weight_decay_style, "correct_bias": correct_bias}
        self.weight_decay_style = weight_decay_style
        self.upcast_optim_states = upcast_optim_states
        self.enable_master_weight = enable_master_weight
        self.Collage = Collage
        self.monitor_metrics = monitor_metrics
        if enable_master_weight: assert Collage == 'none'
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        collage_is_not_none = self.Collage is not 'none'
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
            group_name = group['name']
            if self.monitor_metrics:
                for metric_name in partial_metrics:
                    partial_metrics[metric_name][group_name] = 0.0
            for p in group["params"]:
                if p.grad is None:
                    continue
                if self.upcast_optim_states:
                    grad = p.grad.data.float()
                else:
                    grad = p.grad.data

                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                    if self.enable_master_weight:
                        state["master_weight"] = p.clone().float()
                    if collage_is_not_none:
                        state["weight_accounted"] = torch.zeros_like(p)
                        if self.Collage == 'plus':
                            state["exp_avg_sq_accounted"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                if collage_is_not_none:
                    weight_accounted = state["weight_accounted"]
                else:
                    weight_accounted = None

                if group["weight_decay"] > 0.0 and group["weight_decay_style"] == 'torch':
                    # Perform stepweight decay
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                if self.Collage == 'plus':
                    rounded_beta2 = torch.tensor(beta2).to(exp_avg_sq)
                    beta2_correction = torch.tensor(beta2 - torch.tensor(beta2).item()).to(exp_avg_sq)
                    # multiple with MultMC
                    MultMC_(exp_avg_sq, state["exp_avg_sq_accounted"], rounded_beta2, beta2_correction)
                    Grow_Exp_addcmul_(exp_avg_sq, state["exp_avg_sq_accounted"], grad, grad, 1.0 - beta2)
                    # use low precision sqrt
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
                
                if group["weight_decay"] > 0.0 and group["weight_decay_style"] == 'compact':
                    if self.enable_master_weight:
                        raw_p_update = - step_size * exp_avg / denom - group["lr"] * group["weight_decay"] * state["master_weight"]
                    else:
                        raw_p_update = - step_size * exp_avg / denom - group["lr"] * group["weight_decay"] * p
                else:
                    raw_p_update = - step_size * exp_avg / denom
                if self.upcast_optim_states:
                    raw_p_update = raw_p_update.to(p.dtype)
                if not collage_is_not_none:
                    if self.enable_master_weight:
                        if self.monitor_metrics:
                            effective_update = (state["master_weight"] + raw_p_update) - state["master_weight"]
                            partial_metrics['imprecision_percentage'][group_name] += torch.eq(state["master_weight"], state["master_weight"] + raw_p_update).sum()
                        state["master_weight"].add_(raw_p_update)
                        p.copy_(state["master_weight"].data)
                    else:
                        if self.monitor_metrics:
                            effective_update = (p + raw_p_update) - p
                            partial_metrics['imprecision_percentage'][group_name] += torch.eq(p, p + raw_p_update).sum()
                        p.add_(raw_p_update)
                else:
                    if self.monitor_metrics:
                        tmp, weight_accounted = Grow_Exp(p, weight_accounted, raw_p_update)
                        effective_update = tmp - p
                        partial_metrics['imprecision_percentage'][group_name] += torch.eq(p, tmp).sum()
                        p.copy_(tmp)
                        state["weight_accounted"].copy_(weight_accounted)
                    else:
                        Grow_Exp_(p, weight_accounted, raw_p_update)
                
                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0 and group["weight_decay_style"] == 'HF':
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))
                    
                if self.monitor_metrics:
                    # track metrics
                    raw_p_update = raw_p_update.float()
                    effective_update = effective_update.float()
                    if collage_is_not_none:
                        weight_accounted = weight_accounted.float()
                        partial_metrics['weight_accounted_norm'][group_name] += torch.sum(weight_accounted ** 2)
                        partial_metrics['weight_accounted_ip_curent_raw_update'][group_name] += torch.sum(weight_accounted * raw_p_update)
                    partial_metrics['param_count'][group_name] += torch.numel(p)
                    partial_metrics['raw_update_norm'][group_name] += torch.sum(raw_p_update ** 2)
                    partial_metrics['effective_update_norm'][group_name] += torch.sum(effective_update ** 2)
                    partial_metrics['effective_update_ip_current_raw_update'][group_name] += torch.sum(effective_update * raw_p_update)
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
                complete_norm_metrics(partial_metrics['error_accounted_norm'])
                complete_final_ip_metrics(partial_metrics['error_accounted_ip_curent_raw_update'])
                partial_metrics['projected_error_accounted_norm_along_current_raw_update'] = get_final_projected_metrics(partial_metrics['error_accounted_ip_curent_raw_update'], partial_metrics['raw_update_norm'])
                partial_metrics['projected_error_accounted_norm_ratio_current_raw_update'] = get_final_ratio_metrics(partial_metrics['projected_error_accounted_norm_along_current_raw_update'], partial_metrics['raw_update_norm'])
        return loss, partial_metrics