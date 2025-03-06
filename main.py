"""This script is used to excite the toy mimo system,
and get the response signals.
"""
from src.mimo_system import ToyMIMO
from src.utils import *

def get_data(file_name):
    """Read the data from file.
    """
    data = load_file(file_name)
    return data['us'], data['t_stamps']

def main(nr_inputs,
         nr_outputs,
         signal_name: str='test'):
    
    sys = ToyMIMO(nr_inputs, nr_outputs)
    sys.initialization()
    sys.build_system()

    us, t_stamp = get_data(signal_name)
    y = sys.excite_system(us[0:3, :], t_stamp[:10000])    
    
    data = {
        'system': sys.sys_name,  # which system to excite
        'signal': signal_name,  # which signal to apply
        'u': us,  # inputs
        'y': y,  # outputs
        't_stamp_input': t_stamp,  # time stamp of inputs
        't_stamp_output': t_stamp,  # time stamp of outputs
    }

if __name__ == '__main__':
    main(nr_inputs=3,
         nr_outputs=2)