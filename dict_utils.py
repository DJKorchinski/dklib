import numpy as np 

def get_param_array_from_dictionaries(list_of_dictionaries, param_name):
    """
    Returns an array of unique values for a given parameter name from a list of dictionaries.

    Parameters:
    - list_of_dictionaries (list): A list of dictionaries.
    - param_name (str): The name of the parameter to extract from the dictionaries.

    Returns:
    - array: An array of unique values for the given parameter name.
    """
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

import pandas as pd  
def build_invertible_map(df: pd.DataFrame, column_name: str, column_key: str) -> bidict.bidict:
    """
    Builds an invertible map for a specified column in a DataFrame.
    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the column.
    - column_name (str): The name of the column to build the map for.
    - column_key (str): The name of the column to store the mapped values.
    Returns:
    - element_map (bidict.bidict): The invertible map where keys are elements from the column, and values are the unique labels for them.
    """
    unique_elements = df[column_name].unique()
    element_map = auto_bidict(unique_elements)
    def apply_map(row):
        return element_map[row[column_name]]
    df[column_key] = df.apply(apply_map,axis=1)
    return element_map