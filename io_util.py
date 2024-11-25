from typing import Callable, Iterable, List, TypeVar
T = TypeVar('T')
import pickle
import pathlib

def check_existence(fname: str) -> bool:
    """
    Check if a file exists at the given file path.
    Parameters:
        fname (str): The file path to check.
    Returns:
        bool: True if the file exists, False otherwise.
    """
    return pathlib.Path(fname).is_file()


def load_series(
    fname_function: Callable[[T], str], iterator: Iterable[T], print_skips: bool = False
) -> List:
    """
    Load a series of files from a given iterator.
    Parameters:
        fname_function (Callable[[T], str]): A function that takes an element from the iterator and returns the file path to load.
        iterator (Iterable[T]): An iterator that yields elements to load.
    Returns:
        List: A list of the loaded data.
    """
    data = []
    for rep in iterator:
        fname = fname_function(rep)
        if not check_existence(fname):
            if print_skips:
                print("Skipping file, does not exist: ", fname)
            continue
        with open(fname, "rb") as fh:
            data.append(pickle.load(fh))
    return data
