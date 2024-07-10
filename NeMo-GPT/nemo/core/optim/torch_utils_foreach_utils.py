# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

from typing import List
import torch

def _get_foreach_kernels_supported_devices() -> List[str]:
    r"""
    Return the device type list that supports foreach kernels.
    """
    return ["cuda", "xpu", torch._C._get_privateuse1_backend_name()]

def _get_fused_kernels_supported_devices() -> List[str]:
    r"""
    Return the device type list that supports fused kernels in optimizer.
    """
    return ["cuda", "xpu", torch._C._get_privateuse1_backend_name()]