import numpy as np 

def get_param_array_from_dictionaries(list_of_dictionaries, param_name):
    return np.unique(sorted([dic[param_name] for dic in list_of_dictionaries]))

def dict_inverse(dic : dict) -> dict:
    return {v: k for k, v in dic.items()} 

import bidict
from typing import List
import bidict
def auto_bidict(arr: List) -> bidict.bidict:
    """
    Creates a bidirectional dictionary from a given array.

    Parameters:
    arr (list): The array from which to create the bidirectional dictionary.

    Returns:
    bidict.bidict: The bidirectional dictionary created from the array.
    """
    return bidict.bidict({elem: i for i, elem in enumerate(arr)})    