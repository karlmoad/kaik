from pathlib import Path
import numpy as np
import pandas as pd

def load_test_file(fname):
    pth = Path(__file__).parent.parent.parent.joinpath('ztest',fname)
    with open(str(pth), 'r') as f:
        return f.read()
    
def array_equals(sets):
    r = []
    for i in range(len(sets)):
        a, b = sets[i]
        r.append(np.array_equal(a, b))
    return r

def load_test_dataframe_from_csv(fname, **kwargs):
    pth = Path(__file__).parent.parent.parent.joinpath('ztest', fname)
    return pd.read_csv(pth, **kwargs)
