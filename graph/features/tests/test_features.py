import numpy as np
from graph.features.feature import Feature
from graph.features.tests import dense_tst_vals, sparse_tst_vals, sparse_tst_actual, dense_tst_actual

def init_test_data():
    rez_d = []
    for val in dense_tst_vals:
        rez_d.append(Feature.from_string(val))
    
    rez_s = []
    for val in sparse_tst_vals:
        rez_s.append(Feature.from_string(val))
    
    rez_d_a = [np.array(v, dtype=np.float64) for v in dense_tst_actual]
    rez_s_a = [np.array(v, dtype=np.float64) for v in sparse_tst_actual]
  
    return rez_d, rez_s, rez_d_a, rez_s_a

def feature_batch_to_numpy(features):
    return [f.to_numpy(dtype=np.float64) for f in features]

def array_to_numpy(l):
    return [np.array(v, dtype=np.float64) for v in l]

def array_equals(sets):
    r=[]
    for i in range(len(sets)):
        a,b = sets[i]
        r.append(np.array_equal(a,b))
    return r
    
def test_feature_from_string():
    rez_d, rez_s, rez_d_a, rez_s_a = init_test_data()
        
    dense_obj = [r is not None for r in rez_d]
    sparse_obj = [r is not None for r in rez_s]

    assert all(dense_obj)
    assert all(sparse_obj)
    
    rez_d_r = feature_batch_to_numpy(rez_d)
    assert all(array_equals([(a,b) for a,b in zip(rez_d_r,rez_d_a)]))

    rez_s_r = feature_batch_to_numpy(rez_s)
    assert all(array_equals([(a,b) for a,b in zip(rez_s_r,rez_s_a)]))
        
        
def test_feature_dimension():
    rez_d, rez_s, rez_d_a, rez_s_a = init_test_data()
    
    rez_d_r = feature_batch_to_numpy(rez_d)
    rez_s_r = feature_batch_to_numpy(rez_s)
    
    assert all([len(a) == len(b) for a,b in zip(rez_d_r,rez_d_a)])
    assert all([len(a) == len(b) for a,b in zip(rez_s_r,rez_s_a)])
    
    
    
    
    
        
    
        
    
    
    
    
    

