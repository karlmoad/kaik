import json
import numpy as np
from common.utils.test_utils import load_test_file, array_equals
from graph import GraphObjectType
from graph.features import Feature, FeatureStore
from common.utils.test_utils import array_equals

class TestFeatureStore:
    def __init__(self):
        data = json.loads(load_test_file('features.json'))
        self.sparse_tst_vals = data['sparse']['test']
        self.sparse_tst_actual = data['sparse']['actual']
        self.dense_tst_vals = data['dense']['test']
        self.dense_tst_actual = data['dense']['actual']
        
        self.rez_d = []
        for val in self.dense_tst_vals:
            self.rez_d.append(Feature.from_string(val))
        
        self.rez_s = []
        for val in self.sparse_tst_vals:
            self.rez_s.append(Feature.from_string(val))
        
        self.rez_d_a = [np.array(v, dtype=np.float64) for v in self.dense_tst_actual]
        self.rez_s_a = [np.array(v, dtype=np.float64) for v in self.sparse_tst_actual]
        
    def test_feature_store_fill(self):
        fs = FeatureStore(len(self.dense_tst_vals), len(self.sparse_tst_vals))
        assert len(fs) == len(self.dense_tst_vals) + len(self.sparse_tst_vals)
        
        ctr = 0
        for tv in self.dense_tst_vals:
            fs.add_feature(ctr, GraphObjectType.NODE, Feature.from_string(tv))
            ctr += 1
        
        ctr = 0
        for tv in self.sparse_tst_vals:
            fs.add_feature(ctr, GraphObjectType.EDGE, Feature.from_string(tv))
            ctr += 1
        
        # test feature index same size as actual feature list
        assert len(fs) == fs.num_features
        
        # value test not feature validation
        setN = fs.get_features([1, 2, 3], GraphObjectType.NODE, False)[:, 1:]
        setE = fs.get_features([2, 3, 4], GraphObjectType.EDGE, False)[:, 1:]
        
        setN_actual = np.array(self.dense_tst_actual[1:4], dtype=np.float64)
        setE_actual = np.array(self.sparse_tst_actual[2:5], dtype=np.float64)
        
        assert all(array_equals([(a, b) for a, b in zip(setN, setN_actual)]))
        assert all(array_equals([(a, b) for a, b in zip(setE, setE_actual)]))


    def test_feature_store_verify(self):
        fs = FeatureStore(2, 2)
        
        # sparse [0,1],  dense [7,8]
        fs.add_feature(0, GraphObjectType.NODE, Feature.from_string(self.sparse_tst_vals[0]))
        fs.add_feature(1, GraphObjectType.NODE, Feature.from_string(self.sparse_tst_vals[1]))
        fs.add_feature(0, GraphObjectType.EDGE, Feature.from_string(self.dense_tst_vals[7]))
        fs.add_feature(1, GraphObjectType.EDGE, Feature.from_string(self.dense_tst_vals[8]))
        
        with pytest.raises(ValueError):
            x = fs.get_features([0, 1], GraphObjectType.NODE, True)
        
        with pytest.raises(ValueError):
            y = fs.get_features([0, 1], GraphObjectType.EDGE, True)