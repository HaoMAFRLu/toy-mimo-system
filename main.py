"""This script is used to excite the toy mimo system,
and get the response signals.
"""
import numpy as np

from src.mimo_system import ToyMIMO
from src.utils import *

def get_data(file_name):
    """Read the data from file.
    """
    data = load_file(file_name)
    return data['us'], data['t_stamps']

def get_applied_u(u: np.ndarray) -> np.ndarray:
    return 2.0*u

def main(nr_outputs: int,
         signal_name: str):
    us, t_stamp = get_data(signal_name)
    us_applied = get_applied_u(us)

    nr_inputs = us_applied.shape[0]
    
    sys = ToyMIMO(nr_inputs, nr_outputs)
    sys.initialization()
    sys.build_system()
    y = sys.excite_system(us_applied, t_stamp)    
    
    data = {
        'system': sys.sys_name,     # which system to excite
        'signal': signal_name,      # which signal to load
        'u': us_applied,            # signals applied to the system
        'y': y,                     # outputs
        't_stamp_input': t_stamp,   # time stamp of inputs
        't_stamp_output': t_stamp,  # time stamp of outputs
    }

    save_data(data, 'test')

if __name__ == '__main__':
    main(nr_outputs=2,
         signal_name='test')