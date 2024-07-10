# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

import torch
from torch import Tensor
from .torch_optim_optimizer import (_get_value, _dispatch_sqrt)
from typing import List, Dict, Optional, Union
from .monitor_metrics import Grow_Exp, Grow_Exp_, Grow_Exp_addcmul_, MultMC_

def _single_tensor_adamw(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    accounted_weights: List[Tensor],
    accounted_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: Union[Tensor, float],
    weight_decay: float,
    weight_decay_style: str,
    Collage: str,
    monitor_metrics: bool,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
    has_complex: bool,
):

    assert grad_scale is None and found_inf is None

    if torch.jit.is_scripting():
        # this assert is due to JIT being dumb and not realizing that the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float since most people using JIT are using floats
        assert isinstance(lr, float)
    
    collage_is_not_none = Collage is not 'none'
    if monitor_metrics:
        partial_metrics_group = {
                'raw_update_norm':0.0, 'effective_update_norm':0.0, 
                'effective_update_ip_current_raw_update':0.0,
                'imprecision_percentage':0.0,
                'param_count':0.0,
            }
        if collage_is_not_none:
            partial_metrics_group.update({'weight_accounted_norm':0.0, 
                                        'weight_accounted_ip_curent_raw_update':0.0})
    else:
        partial_metrics_group = None
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]
        if collage_is_not_none:
            weight_accounted = accounted_weights[i]
            if Collage == 'plus':
                exp_avg_sq_accounted = accounted_exp_avg_sqs[i]

        # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
        if not torch._utils.is_compiling() and capturable:
            assert (
                (param.is_cuda and step_t.is_cuda) or (param.is_xla and step_t.is_xla)
            ), "If capturable=True, params and state_steps must be CUDA or XLA tensors."

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            if collage_is_not_none:
                weight_accounted = torch.view_as_real(weight_accounted)
                if Collage == 'plus':
                    exp_avg_sq_accounted = torch.view_as_real(exp_avg_sq_accounted)
            if amsgrad:
                max_exp_avg_sqs[i] = torch.view_as_real(max_exp_avg_sqs[i])
            param = torch.view_as_real(param)

        # update step
        step_t += 1

        if weight_decay > 0.0 and weight_decay_style == 'torch':
            # Perform stepweight decay
            param.mul_(1 - lr * weight_decay)
        # Decay the first and second moment running average coefficient
        exp_avg.lerp_(grad, 1 - beta1)
        if Collage == 'plus':
            rounded_beta2 = torch.tensor(beta2).to(exp_avg_sq)
            beta2_correction = torch.tensor(beta2 - torch.tensor(beta2).item()).to(exp_avg_sq)
            # multiple with MultMC
            MultMC_(exp_avg_sq, exp_avg_sq_accounted, rounded_beta2, beta2_correction)
            Grow_Exp_addcmul_(exp_avg_sq, exp_avg_sq_accounted, grad, grad, 1.0 - beta2)
            # accounted_exp_avg_sqs[i].copy_(exp_avg_sq_accounted)
        else:
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        if capturable or differentiable:
            step = step_t
            # since step is float, the following terms (till ###) all in fp32, should be numerically safe, 
            # otherwise problematic in low precision, bf16
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            step_size = lr / bias_correction1
            step_size_neg = step_size.neg()
            bias_correction2_sqrt = bias_correction2.sqrt()
            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                if differentiable:
                    max_exp_avg_sq = max_exp_avg_sqs[i].clone()
                else:
                    max_exp_avg_sq = max_exp_avg_sqs[i]

                max_exp_avg_sqs[i].copy_(torch.maximum(max_exp_avg_sq, exp_avg_sq))

                # Uses the max. for normalizing running avg. of gradient
                # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write
                # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)
                denom = (
                    max_exp_avg_sqs[i].sqrt() / (bias_correction2_sqrt * step_size_neg)
                ).add_(eps / step_size_neg)
            else:
                denom = (
                    exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)
                ).add_(eps / step_size_neg)
            if weight_decay > 0.0 and weight_decay_style == 'compact':
                raw_p_update = exp_avg / denom - lr * weight_decay * param
            else:
                raw_p_update = exp_avg / denom
        else:
            step = _get_value(step_t)

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            step_size = lr / bias_correction1

            bias_correction2_sqrt = _dispatch_sqrt(bias_correction2)

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])

                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

            if weight_decay > 0.0 and weight_decay_style == 'compact':
                raw_p_update = -step_size * exp_avg / denom - lr * weight_decay * param
            else:
                raw_p_update = -step_size * exp_avg / denom
        if collage_is_not_none:
            #### standard pipeline
            if monitor_metrics:
                tmp, weight_accounted = Grow_Exp(param, weight_accounted, raw_p_update)
                effective_update = tmp - param
                partial_metrics_group['imprecision_percentage'] += torch.eq(param, tmp).sum()
                param.copy_(tmp)
                accounted_weights[i].copy_(weight_accounted)
            else:
                Grow_Exp_(param, weight_accounted, raw_p_update)
        else:
            if monitor_metrics:
                effective_update = (param + raw_p_update) - param
                partial_metrics_group['imprecision_percentage'] += torch.eq(param, param + raw_p_update).sum()
            param.add_(raw_p_update)

        if weight_decay > 0.0 and weight_decay_style == 'HF':
            param.add_(param, alpha=(-lr * weight_decay))
        
        if monitor_metrics:
            # track metrics
            raw_p_update = raw_p_update.float()
            effective_update = effective_update.float()
            if collage_is_not_none:
                weight_accounted = weight_accounted.float()
                partial_metrics_group['weight_accounted_norm'] += torch.sum(weight_accounted ** 2)
                partial_metrics_group['weight_accounted_ip_curent_raw_update'] += torch.sum(weight_accounted * raw_p_update)
            partial_metrics_group['param_count'] += torch.numel(param)
            partial_metrics_group['raw_update_norm'] += torch.sum(raw_p_update ** 2)
            partial_metrics_group['effective_update_norm'] += torch.sum(effective_update ** 2)
            partial_metrics_group['effective_update_ip_current_raw_update'] += torch.sum(effective_update * raw_p_update)
        
        # Lastly, switch back to complex view
        if amsgrad and torch.is_complex(params[i]):
            max_exp_avg_sqs[i] = torch.view_as_complex(max_exp_avg_sqs[i])
    return partial_metrics_group