# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

from typing import Optional
import torch
"""
MC operations
"""
def Grow_Exp_(tensor_m, tensor_r, val):
    # tensor_m, tensor_r can be modified in-place
    result_m = tensor_m + val
    tensor_r.add_(val - (result_m - tensor_m))
    tensor_m.copy_(result_m + tensor_r)
    tensor_r.sub_(tensor_m - result_m)

def Grow_Exp(tensor_m, tensor_r, val):
    result_m = tensor_m + val
    tensor_r = tensor_r + (val - (result_m - tensor_m))
    tensor_m = result_m + tensor_r
    tensor_r = tensor_r - (tensor_m - result_m)
    return tensor_m, tensor_r

def Grow_Exp_addcmul_(tensor_m, tensor_r, t1, t2, v):
    # val = t1*t2*v
    # tensor_m, tensor_r can be modified in-place
    value = torch.addcmul(torch.zeros_like(t1), t1, t2, value=v)
    result_m = tensor_m + value
    value.add_(tensor_m - result_m).add_(tensor_r) # negative of (result_m - tensor_m)
    tensor_m.copy_(result_m + value)
    tensor_r.copy_(value - (tensor_m - result_m))
    # return tensor_m, tensor_r

def MultMC_(tensor_m, tensor_r, val_m, val_r):
    # tensor_m, tensor_r can be modified in-place
    # val_m, val_r are scalars
    result_m = tensor_m.mul_(val_m)
    value = torch.addcmul(-result_m, tensor_m, val_m)
    tensor_r.mul_(val_m).addcmul_(tensor_m, val_r).add_(value)
    tensor_m.copy_(result_m + tensor_r)
    tensor_r.sub_(tensor_m - result_m)
    # return tensor_m, tensor_r

"""
compute param and grad norm
"""
def get_param_norm(
    model,
    device='cuda',
    norm_type=2.0,
    report_per_block=False,
):

    norm_type = float(norm_type)
    total_param_norm = torch.tensor(0.0).double().to(device)
    if report_per_block:
        block_param_norm = {}
        embedding_param_norm = torch.tensor(0.0).double().to(device)
        try:
            model_typed = model.bert
        except:
            model_typed = model.roberta
        for sub_embedding_parameter in model_typed.embeddings.parameters():
            if norm_type == 2.0:
                embedding_param_norm += torch.square(sub_embedding_parameter.detach().double()).sum()
            else:
                embedding_param_norm += torch.norm(sub_embedding_parameter.detach().double(), norm_type) ** norm_type
        block_param_norm['model.embeddings'] = embedding_param_norm ** (1.0 / norm_type)
        total_param_norm += embedding_param_norm

        bert_encoder_param_norm = torch.tensor(0.0).double().to(device)
        for bert_layer_index, bert_layer in model_typed.encoder.layer.named_children():
            bert_layer_param_norm = torch.tensor(0.0).double().to(device)
            for sub_layer_parameter in bert_layer.parameters():
                if norm_type == 2.0:
                    bert_layer_param_norm += torch.square(sub_layer_parameter.detach().double()).sum()
                else:
                    bert_layer_param_norm += torch.norm(sub_layer_parameter.detach().double(), norm_type) ** norm_type
            block_param_norm[f'model.encoder.layer{bert_layer_index}'] = bert_layer_param_norm ** (1.0 / norm_type)
            bert_encoder_param_norm += bert_layer_param_norm
        block_param_norm['model.encoder'] = bert_encoder_param_norm ** (1.0 / norm_type)
        block_param_norm['model_typed'] = (embedding_param_norm + bert_encoder_param_norm) ** (1.0 / norm_type)
        total_param_norm += bert_encoder_param_norm
        
        cls_param_norm = torch.tensor(0.0).double().to(device)
        try:
            model_typed = model.cls
        except:
            model_typed = model.lm_head
        for sub_cls_parameter in model_typed.parameters():
            if norm_type == 2.0:
                cls_param_norm += torch.square(sub_cls_parameter.detach().double()).sum()
            else:
                cls_param_norm += torch.norm(sub_cls_parameter.detach().double(), norm_type) ** norm_type
        block_param_norm['last_layer'] = cls_param_norm ** (1.0 / norm_type)
        total_param_norm += cls_param_norm
        block_param_norm['total_param'] = total_param_norm ** (1.0 / norm_type)
        return block_param_norm
    else:
        for layer_param in model.parameters():
            if norm_type == 2.0:
                total_param_norm += torch.square(layer_param.detach().double()).sum()
            else:
                total_param_norm += torch.norm(layer_param.detach().double(), norm_type) ** norm_type
        total_param_norm = total_param_norm ** (1.0 / norm_type)
        return total_param_norm

def get_grad_norm(
    model,
    device='cuda',
    norm_type=2.0,
    report_per_block=False,
):
    
    norm_type = float(norm_type)
    total_grad_norm = torch.tensor(0.0).double().to(device)
    if report_per_block:
        block_grad_norm = {}
        embedding_grad_norm = torch.tensor(0.0).double().to(device)
        try:
            model_typed = model.bert
        except:
            model_typed = model.roberta
        for sub_embedding_parameter_ in model_typed.embeddings.parameters():
            grad_not_none = sub_embedding_parameter_.grad is not None
            if grad_not_none:
                sub_embedding_parameter_grad = sub_embedding_parameter_.grad.detach().double()
                if norm_type == 2.0:
                    embedding_grad_norm += torch.square(sub_embedding_parameter_grad).sum()
                else:
                    embedding_grad_norm += torch.norm(sub_embedding_parameter_grad, norm_type) ** norm_type
        block_grad_norm['model.embeddings'] = embedding_grad_norm ** (1.0 / norm_type)
        total_grad_norm += embedding_grad_norm

        bert_encoder_grad_norm = torch.tensor(0.0).double().to(device)
        for bert_layer_index, bert_layer in model_typed.encoder.layer.named_children():
            bert_layer_grad_norm = torch.tensor(0.0).double().to(device)
            for sub_layer_parameter_ in bert_layer.parameters():
                grad_not_none = sub_layer_parameter_.grad is not None
                if grad_not_none:
                    sub_layer_parameter_grad = sub_layer_parameter_.grad.detach().double()
                    if norm_type == 2.0:
                        bert_layer_grad_norm += torch.square(sub_layer_parameter_grad).sum()
                    else:
                        bert_layer_grad_norm += torch.norm(sub_layer_parameter_grad, norm_type) ** norm_type
            block_grad_norm[f'model.encoder.layer{bert_layer_index}'] = bert_layer_grad_norm ** (1.0 / norm_type)
            bert_encoder_grad_norm += bert_layer_grad_norm
        block_grad_norm['model.encoder'] = bert_encoder_grad_norm ** (1.0 / norm_type)
        block_grad_norm['model_typed'] = (embedding_grad_norm + bert_encoder_grad_norm) ** (1.0 / norm_type)
        total_grad_norm += bert_encoder_grad_norm
        
        cls_grad_norm = torch.tensor(0.0).double().to(device)
        try:
            model_typed = model.cls
        except:
            model_typed = model.lm_head
        for sub_cls_parameter_ in model_typed.parameters():
            grad_not_none = sub_layer_parameter_.grad is not None
            if grad_not_none:
                sub_cls_parameter_grad = sub_cls_parameter_.grad.detach().double()
                if norm_type == 2.0:
                    cls_grad_norm += torch.square(sub_cls_parameter_grad).sum()
                else:
                    cls_grad_norm += torch.norm(sub_cls_parameter_grad, norm_type) ** norm_type
        block_grad_norm['last_layer'] = cls_grad_norm ** (1.0 / norm_type)
        total_grad_norm += cls_grad_norm
        block_grad_norm['total_grad'] = total_grad_norm ** (1.0 / norm_type)
        return block_grad_norm
    else:
        for layer_param_ in model.parameters():
            grad_not_none = layer_param_.grad is not None
            if grad_not_none:
                layer_param_grad = layer_param_.grad.detach().double()
                if norm_type == 2.0:
                    total_grad_norm += torch.square(layer_param_grad).sum()
                else:
                    total_grad_norm += torch.norm(layer_param_grad, norm_type) ** norm_type
        total_grad_norm = total_grad_norm ** (1.0 / norm_type)
        return total_grad_norm

"""
funcs to monitor and complete metric_dict
"""
def complete_norm_metrics(norm0):
    encoder_sq_norm = 0.0
    for layer_key in norm0.keys():
        if 'model.encoder' in layer_key:
            encoder_sq_norm += norm0[layer_key]
        norm0[layer_key] = torch.sqrt(norm0[layer_key])
    norm0['model'] = norm0['model.last']**2 + norm0['model.embeddings']**2 + encoder_sq_norm
    norm0['model.encoder'] = torch.sqrt(encoder_sq_norm)
    norm0['model'] = torch.sqrt(norm0['model'])

def complete_final_ip_metrics(ip0):
    encoder_ip = 0.0
    for layer_key in ip0.keys():
        if 'model.encoder' in layer_key:
            encoder_ip += ip0[layer_key]
    ip0['model.encoder'] = encoder_ip
    ip0['model'] = ip0['model.last'] + ip0['model.embeddings'] + ip0['model.encoder']

def get_final_ratio_metrics(norm0, base_norm):
    ratio = {}
    for layer_key in norm0.keys():
        ratio[layer_key] = norm0[layer_key] / base_norm[layer_key]
    return ratio

def get_final_projected_metrics(ip0, base_norm):
    projected_norm = {}
    for layer_key in ip0.keys():
        projected_norm[layer_key] = ip0[layer_key] / base_norm[layer_key]
    return projected_norm

def training_metrics_closure(logger_metrics, epoch, global_step, metrics_dict):
    for metric_name in metrics_dict:
        if metric_name not in ['step_loss', 'learning_rate', 'throughput']:
            if isinstance(metrics_dict[metric_name], dict):
                for layer_keys in metrics_dict[metric_name].keys():
                    metrics_dict[metric_name][layer_keys] = metrics_dict[metric_name][layer_keys].detach().to('cpu').tolist()
            else:
                metrics_dict[metric_name] = metrics_dict[metric_name].detach().to('cpu').item()
    if logger_metrics != None:
        logger_metrics.log(epoch, global_step, metrics_dict)