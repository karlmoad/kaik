import json
from common.utils.test_utils import load_test_file

sparse_tst_vals=None

sparse_tst_actual=None

dense_tst_vals=None

dense_tst_actual=None

def load_test_data():
    global sparse_tst_vals
    global sparse_tst_actual
    global dense_tst_vals
    global dense_tst_actual
    
    data = json.loads(load_test_file('features.json'))
    sparse_tst_vals=data['sparse']['test']
    sparse_tst_actual=data['sparse']['actual']
    dense_tst_vals=data['dense']['test']
    dense_tst_actual=data['dense']['actual']
    


load_test_data()
