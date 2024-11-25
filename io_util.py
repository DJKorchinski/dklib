from typing import Iterable, List
import pickle


def check_existence(fname: str) -> bool:
    """
    Check if a file exists at the given file path.
    Parameters:
        fname (str): The file path to check.
    Returns:
        bool: True if the file exists, False otherwise.
    """
    import pathlib

    return pathlib.Path(fname).is_file()


def load_series(
    fname_function: function, iterator: Iterable, print_skips: bool = False
) -> List:
    """
    Load a series of files from a given iterator.
    Parameters:
        fname_function (function): A function that takes an element from the iterator and returns the file path to load.
        iterator (Iterable): An iterator that yields elements to load.
    Returns:
        List: A list of the loaded data.
    """
    data = []
    for rep in iterator:
        fname = fname_function(rep)
        if not check_existence(fname):
            if print_skips:
                print("Skpping file, does not exist: ", fname)
            continue
        with open(fname, "rb") as fh:
            data.append(pickle.load(fh))
    return data
