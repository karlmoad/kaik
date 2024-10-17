import random as rn
import numpy as np
import numpy.random as np_rn


def is_int_in_range(x, variable_name, min_val=-np.inf, max_val=np.inf):
    """
    A function to verify that a variable is an integer in a specified closed range.
    Args:
        x: the variable to check
        variable_name (str): the name of the variable to print out in error messages
        min_val: the minimum value that `x` can attain
        max_val: the maximum value that `x` can attain
    """

    if not isinstance(x, int):
        raise TypeError(f"{variable_name}: expected int, found {type(x).__name__}")

    if x < min_val or x > max_val:

        if min_val == -np.inf:
            region = f"<= {max_val}"
        elif max_val == np.inf:
            region = f">= {min_val}"
        else:
            region = f"in the range [{min_val}, {max_val}]"

        raise ValueError(f"{variable_name}: expected integer {region}, found {x}")

    return True


class RandomState:
    """
        manages an instanced set of randomization objects aligned with an optional seed value
    """

    def __init__(self, seed):
        self._r = rn.Random(seed) if seed is not None else rn
        self._n = np_rn.RandomState(seed) if seed is not None else np_rn

    @property
    def state(self):
        return self._r, self._n

    @property
    def random(self):
        return self._r

    @property
    def numpy(self):
        return self._n
