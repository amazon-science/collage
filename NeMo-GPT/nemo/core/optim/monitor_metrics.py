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