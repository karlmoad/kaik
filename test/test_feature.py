import json
import numpy as np
from kaik.common.utils.test_utils import load_test_file, array_equals
from kaik.common.utils.serialization_utils import serialize, deserialize
from kaik.graph.features import Feature
import logging


class TestFeature:
    
    def test_feature(self, caplog):
        with caplog.at_level(logging.INFO):
            wrap = FeatureTestWrapper()
            wrap.t_feature_from_string()
            wrap.t_feature_dimension()
            wrap.t_feature_serialization()

        
class FeatureTestWrapper:
    def __init__(self):
        data = json.loads(load_test_file('features.json'))
        self.sparse_tst_vals = data['sparse']['test']
        self.sparse_tst_actual = data['sparse']['actual']
        self.dense_tst_vals = data['dense']['test']
        self.dense_tst_actual = data['dense']['actual']
        self.log = logging.getLogger(__name__)
        
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
    
    
    def t_feature_from_string(self):
        
        
        dense_obj = [r is not None for r in self.rez_d]
        sparse_obj = [r is not None for r in self.rez_s]
        
        assert all(dense_obj)
        assert all(sparse_obj)
        
        rez_d_r = FeatureTestWrapper.feature_batch_to_numpy(self.rez_d)
        assert all(array_equals([(a, b) for a, b in zip(rez_d_r, self.rez_d_a)]))
        
        rez_s_r = FeatureTestWrapper.feature_batch_to_numpy(self.rez_s)
        assert all(array_equals([(a, b) for a, b in zip(rez_s_r, self.rez_s_a)]))
    
        self.log.info("Feature from string test [SUCCESS]")
        
    def t_feature_dimension(self):
        
        rez_d_r = FeatureTestWrapper.feature_batch_to_numpy(self.rez_d)
        rez_s_r = FeatureTestWrapper.feature_batch_to_numpy(self.rez_s)
        
        assert all([len(a) == len(b) for a, b in zip(rez_d_r, self.rez_d_a)])
        assert all([len(a) == len(b) for a, b in zip(rez_s_r, self.rez_s_a)])
        
        self.log.info("Feature dimension test [SUCCESS]")
        
        
    def t_feature_serialization(self):
        s_rez_d =  serialize(self.rez_d)
        s_rez_s = serialize(self.rez_s)
        
        d_rez_d = FeatureTestWrapper.feature_batch_to_numpy(deserialize(s_rez_d))
        d_rez_s = FeatureTestWrapper.feature_batch_to_numpy(deserialize(s_rez_s))
        
        assert all([len(a) == len(b) for a, b in zip(d_rez_d, self.rez_d_a)])
        assert all([len(a) == len(b) for a, b in zip(d_rez_s, self.rez_s_a)])
        assert all(array_equals([(a, b) for a, b in zip(d_rez_d, self.rez_d_a)]))
        assert all(array_equals([(a, b) for a, b in zip(d_rez_s, self.rez_s_a)]))
        
        self.log.info("Feature serialization test [SUCCESS]")
        


        
        