import numpy as np
from graph import GraphObjectType
from graph.features import Feature, FeatureStore
from graph.features.tests import dense_tst_vals, sparse_tst_vals, sparse_tst_actual, dense_tst_actual, array_equals


def test_feature_store_fill():
    
    fs = FeatureStore(len(dense_tst_vals), len(sparse_tst_vals))
    assert len(fs) == len(dense_tst_vals) + len(sparse_tst_vals)
    
    ctr = 0
    for tv in dense_tst_vals:
        fs.add_feature(ctr, GraphObjectType.NODE, Feature.from_string(tv))
        ctr += 1
        
    ctr = 0
    for tv in sparse_tst_vals:
        fs.add_feature(ctr, GraphObjectType.EDGE, Feature.from_string(tv))
        ctr += 1
    
    
    #test feature index same size as actual feature list
    assert len(fs) == fs.num_features
    
    # value test not feature validation
    setN = fs.get_features([1,2,3], GraphObjectType.NODE, False).astype(np.float64)
    setE = fs.get_features([1,2,3], GraphObjectType.EDGE, False).astype(np.float64)

    setN_actual = np.array(dense_tst_actual[1:4], dtype=np.float64)
    setE_actual = np.array(sparse_tst_actual[1:4], dtype=np.float64)
    
    assert all(array_equals([(a,b) for a,b in zip(setN, setN_actual)]))
    assert all(array_equals([(a,b) for a,b in zip(setE, setE_actual)]))
    
    
    
    


