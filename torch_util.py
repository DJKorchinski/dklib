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
        tensor (torch.Tensor): The input tensor. (might break if tensor is on gpu)

    Returns:
        int: The hash of the tensor.
    """
    return hash(tensor.numpy().tobytes())

def unique_tensors(tensors: list[torch.Tensor],initial_hashes:set|None = None,return_hashes:bool=False) -> list[torch.Tensor]|tuple[list[torch.Tensor],set]:
    """
    Find the unique elements in a list of PyTorch tensors.

    Args:
        tensors (list[torch.Tensor]): The list of input tensors.
        initial_hashes (set | None, optional): A set of initial hashes to disallow. Defaults to None. Useful if adding additional tensors to an existing set.
        return_hashes (bool, optional): If True, returns the set of unique hashes along with the unique tensors. Defaults to False.

    Returns:
        list[torch.Tensor]: The unique tensors in the input list.
        tuple[list[torch.Tensor], set]: If return_hashes is True, returns a tuple containing the unique tensors and the set of unique hashes.
    """
    unique_tensors = []
    if(initial_hashes is not None):
        unique_hashes = initial_hashes
    else:
        unique_hashes = set()
    for tensor in tensors:
        tensor_hash = hash_tensor(tensor)
        if tensor_hash not in unique_hashes:
            unique_hashes.add(tensor_hash)
            unique_tensors.append(tensor)
    if(return_hashes):
        return unique_tensors, unique_hashes
    else:
        return unique_tensors

def print_device_info(device: torch.device|str):
    device = torch.device(device)
    print("Using device:", device, "\n")
    # Additional Info when using cuda
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
        print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")
