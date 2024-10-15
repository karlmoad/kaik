import json
import numpy as np
from common.utils.test_utils import load_test_file, array_equals
from graph.features import Feature


class TestFeature:
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
    
    @staticmethod
    def feature_batch_to_numpy(features):
        return [f.to_numpy(dtype=np.float64) for f in features]
    
    @staticmethod
    def array_to_numpy(l):
        return [np.array(v, dtype=np.float64) for v in l]
    
    
    def test_feature_from_string(self):
        
        
        dense_obj = [r is not None for r in self.rez_d]
        sparse_obj = [r is not None for r in self.rez_s]
        
        assert all(dense_obj)
        assert all(sparse_obj)
        
        rez_d_r = TestFeature.feature_batch_to_numpy(self.rez_d)
        assert all(array_equals([(a, b) for a, b in zip(rez_d_r, self.rez_d_a)]))
        
        rez_s_r = TestFeature.feature_batch_to_numpy(self.rez_s)
        assert all(array_equals([(a, b) for a, b in zip(rez_s_r, self.rez_s_a)]))
    
    
    def test_feature_dimension(self):
        
        rez_d_r = TestFeature.feature_batch_to_numpy(self.rez_d)
        rez_s_r = TestFeature.feature_batch_to_numpy(self.rez_s)
        
        assert all([len(a) == len(b) for a, b in zip(rez_d_r, self.rez_d_a)])
        assert all([len(a) == len(b) for a, b in zip(rez_s_r, self.rez_s_a)])