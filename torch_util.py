"""
This module provides utility functions for working with PyTorch tensors and modules.

Functions:
    get_total_size(tensor: torch.Tensor) -> int:
        Computes the total number of elements in a given tensor.

    get_param_count(module: nn.Module) -> int:
        Computes the total number of parameters in a given PyTorch module.
"""

import torch 
import torch.nn as nn 

def get_total_size(tensor: torch.Tensor) -> int:
    """
    Calculate the total number of elements in a given PyTorch tensor.
    Args:
        tensor (torch.Tensor): The input tensor.
    Returns:
        int: The total number of elements in the tensor.
    """
    return torch.prod(torch.tensor(tensor.shape)).item()

def get_param_count(module: nn.Module) -> int:
    """
    Calculate the total number of parameters in a given PyTorch module.

    Args:
        module (nn.Module): The PyTorch module for which to count the parameters.

    Returns:
        int: The total number of parameters in the module.
    """
    return sum([ get_total_size(p) for p in module.parameters()])

def hash_tensor(tensor: torch.Tensor) -> int:
    """
    Compute the hash of a PyTorch tensor.

    Args:
        tensor (torch.Tensor): The input tensor.

    Returns:
        int: The hash of the tensor.
    """
    return hash(tensor.numpy().tobytes())