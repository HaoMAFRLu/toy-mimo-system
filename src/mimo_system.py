"""Classes for a toy linear mimo system
"""
import numpy as np
import scipy.signal as signal
import pickle
from datetime import datetime

from src.utils import *

np.random.seed(42)

class ToyMIMO():
    """Generate a mimo system. And excite the system
    with designed signals to get reponse signals.
    """
    def __init__(self, nr_inputs: int, nr_outputs: int) -> None:
        """Initialize a instance
        
        Args:
            nr_inputs: the number of inputs of the system
            nr_outputs: the number of outputs of the system
        """
        self.nr_inputs = nr_inputs
        self.nr_outputs = nr_outputs
        self.root = get_parent_path(lvl=1)
    
    def initialization(self):
        """Initialize the parameters for generating the 
        transfer functions.
        """
        self.params = [[None] * self.nr_inputs for _ in range(self.nr_outputs)]
        for row in range(self.nr_outputs):
            for col in range(self.nr_inputs):
                order_num, order_den, num, den = self.get_transfer_function_parameters()
                self.params[row][col] = {
                    'order_num': order_num, 
                    'order_den': order_den,
                    'num': num,
                    'den': den
                }

        self.sys_name = self.get_time()
        self.save_params(self.params, self.sys_name)
    
    def build_system(self) -> None:
        """Build the system.
        """
        self.H = self._built_system(self.params)
        self.save_tf(self.params, self.sys_name)
    
    @staticmethod
    def get_format(a: np.ndarray) -> str:
        """Format the array.

        Args:
            a: the given array.
        
        Returns:
            str: the formatted string
        """
        terms = []
        order = len(a) - 1  # the highest order

        for i, coef in enumerate(a):
            exponent = order - i  # descent
            if coef == 0:
                continue  # skip if coefficient is zero

            if exponent == 0:
                term = f"{abs(coef):.5f}"  # only show the cofficient
            elif exponent == 1:
                term = f"{abs(coef):.5f}s"  # s^1 = s
            else:
                term = f"{abs(coef):.5f}s^{exponent}"  # s^n

            if coef < 0:
                terms.append(f"- {term}")  # add "-" symbol
            else:
                if terms:  # not the first term, add "+"
                    terms.append(f"+ {term}")
                else:
                    terms.append(term)  # first term, no "+"

        return " ".join(terms)

    def save_tf(self, params: dict, 
                file_name: str) -> None:
        """Save the transfer functions as .txt file.
        """
        path_file = os.path.join(self.root, 'data', 'systems', file_name, 'transfer_function.txt')
        
        nr_outputs = len(params)
        nr_inputs = len(params[0])

        with open(path_file, "w") as file:
            for row in range(nr_outputs):
                for col in range(nr_inputs):
                    
                    str_num = self.get_format(params[row][col]['num'])
                    str_den = self.get_format(params[row][col]['den'])

                    file.write(f"H_{row}{col}(s) = \n")
                    file.write("\n")
                    file.write(f"{str_num}\n")
                    file.write("-" * 60 + "\n")
                    file.write(f"{str_den}\n")
                    file.write("\n")
                    file.write("=" * 60 + "\n")
                    file.write("\n")
                    file.write("\n")

    @staticmethod
    def get_num(order: int) -> np.ndarray:
        """Get the parameters for the numerator.

        Args:
            order: the order of the numerator

        Returns:
            array: the parameters of the numerator
        """
        return np.random.uniform(-1, 1, order + 1)

    def get_den(self, order: int) -> np.ndarray:
        """Generate the parameters for the denominator and
        ensure that all poles are negative.

        Args:
            order: the order of the denominator
        
        Returns:
            array: the parameters of the denominator
        """
        poles = -np.random.uniform(0.1, 5.0, order)
        return np.poly(poles)

    @staticmethod
    def get_transfer_function(num: np.ndarray, 
                              den: np.ndarray):
        """Generate the transfer function.

        Args:
            num: the parameters of the numerator
            den: the parameters of the denominator

        Returns:
            H: the transfer function
        """
        return signal.TransferFunction(num, den)

    def get_transfer_function_parameters(self) -> tuple[int, 
                                                        int, 
                                                        np.ndarray, 
                                                        np.ndarray]:
        """Generate the parameters for a transfer function, which ensure
        that the generated transfer function is stable, namely, all poles
        are negative.

        Returns:
            order_num: the order of the nominator
            order_den: the order of the denominator
            num: parameters of the nominator
            den: parameters of the denominator
        """
        order_num, order_den = self.get_orders()
        num = self.get_num(order_num)
        den = self.get_den(order_den)
        return order_num, order_den, num, den
        
    @staticmethod
    def get_random_order(order_min: int, order_max: int) -> int:
        """Generate a random number in [order_min, order_max].

        Args:
            order_min: the minimum order
            order_max: the maximum order

        Returns:
            order: the random order
        """
        return np.random.randint(order_min, order_max+1)

    def get_orders(self, order_max: int=3) -> tuple[int, int]:
        """Generate the orders of numerator and denominator,
        which ensure the causality.

        Args:
            order_max: the highest order
        
        Returns:
            order_num: the order of the numerator
            order_den: the order of the denominator
        """
        order_num = self.get_random_order(1, order_max)  # 1 <= order_num <= order_max
        order_den = self.get_random_order(order_num, order_max+1)  # order_num <= order_den <= order_max + 1
        return order_num, order_den

    def _built_system(self, params: dict) -> np.ndarray:
        """Build the linear system H.

        Args:
            params: parameters of the transfer functions
        
        Returns:
            H: the matrix of the transfer functions
        """
        nr_outputs = len(params)
        nr_inputs = len(params[0])
        H = np.empty((nr_outputs, nr_inputs), dtype=object)

        for row in range(nr_outputs):
            for col in range(nr_inputs):
                num = params[row][col]['num']
                den = params[row][col]['den']
                H[row, col] = self.get_transfer_function(num, den)

        return H
    
    @staticmethod
    def time2freq(signal: np.ndarray) -> complex:
        """Convert time signal to frequency signal.

        Args:
            signal (n x N): the signal in the time domain
        
        Returns:
            Y (n x N): the singal in the frequency domain
        """
        return np.fft.fft(signal, axis=1)

    @staticmethod
    def freq2time(SIGNAL: complex) -> np.ndarray:
        """Convert frequency signal to time signal.

        Args:
            SIGNAL (n x N): the signal in the frequency domain

        Returns:
            y (n x N): the signal in the frequency domain
        """
        return np.fft.ifft(SIGNAL, axis=1).real 
    
    def save_params(self, params: list, 
                    file_name: str) -> None:
        """Save the data of the transfer functions.
        """
        path_folder = os.path.join(self.root, 'data', 'systems', file_name)
        mkdir(path_folder)
        path_file = os.path.join(path_folder, 'params')
        with open(path_file, 'wb') as file:
            pickle.dump(params, file)
    
    @staticmethod
    def get_time():
        """Get the current time as file name.
        """
        current_time = datetime.now()
        return current_time.strftime("%Y-%m-%d_%H-%M-%S")

    def excite_channel(self, H, 
                       u: np.ndarray, 
                       t_stamp: np.ndarray) -> np.ndarray:
        """
        """
        t_out, y_out, _ = signal.lsim(H, U=u, T=t_stamp)
        return y_out

    def excite_dof(self, H: np.ndarray, 
                   u: np.ndarray, 
                   t_stamp: np.ndarray) -> np.ndarray:
        """
        """
        m, N = u.shape
        y = np.zeros((N,))
        for i in range(m):
            y += self.excite_channel(H[i], u[i, :], t_stamp)
        return y
    
    def excite_system(self, u: np.ndarray, 
                      t_stamp: np.ndarray) -> np.ndarray:
        """Excite the system with designed time signals u.

        Args:
            u (nr_inputs x N): the designed input signals
            t_stamp: time stamp for all signals
        
        Returns:
            y (nr_outpus x N): response signals
        """
        _, N = u.shape
        y = np.zeros((self.nr_outputs, N))

        for i in range(self.nr_outputs):
            y[i, :] = self.excite_dof(self.H[i, :], u, t_stamp)

        return y
