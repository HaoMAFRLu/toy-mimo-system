"""Collection of useful functions.
"""
import os
from pathlib import Path
import pickle

def mkdir(path: Path) -> None:
    """Check if the folder exists and create it
    if it does not exist.
    """
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def get_parent_path(lvl: int=0) -> Path:
    """Get the lvl-th parent path as root path.
    Return current file path when lvl is zero.
    Must be called under the same folder.
    """
    path = os.path.dirname(os.path.abspath(__file__))
    if lvl > 0:
        for _ in range(lvl):
            path = os.path.abspath(os.path.join(path, os.pardir))
    return path

def load_file(file_nmae: str) -> dict:
    """Load data from file.
    """
    root = get_parent_path(lvl=1)
    path_file = os.path.join(root, 'data', 'excitation_signals', file_nmae)
    with open(path_file, 'rb') as file:
        data = pickle.load(file)
    return data

def save_data(data: dict,
              file_name: str) -> None:
    """Save the data.
    """ 
    root = get_parent_path(lvl=1)
    path_file = os.path.join(root, 'data', 'response_signals', file_name)
    with open(path_file, 'wb') as file:
        pickle.dump(data, file)

