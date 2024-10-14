from typing import Union
import numpy as np
from numpy import ndarray
from graph import GraphObjectType
from threading import Lock
from graph.features import Feature

class FeatureStore(object):
    __slots__ = ['_features','_idx', '_free','_lock']

    def __init__(self, node_count:int, edge_count:int):
        self._features = []
        self._idx = np.full((node_count+edge_count, 4), None, dtype=int)
        self._free = 0
        self._lock = Lock()
        
    def __getstate__(self):
        return {
            'features': self._features,
            'index': self._idx.tolist(),
            "free": self._free,
        }

    def __setstate__(self, state):
        self._features = state['features']
        self._idx = np.array(state['index'], dtype=int)
        self._free = state['free']
        
    def add_feature(self, id:int, otype:GraphObjectType, feature:Feature):
        try:
            def __advance(n=1):
                self._free += n
            
            self._lock.acquire()
            
            self._idx[self._free] = [id, int(otype), len(feature), self._free]
            self._features.append(feature)
            __advance()
        finally:
            self._lock.release()
            
    def get_features(self, ids:list[int], otype:GraphObjectType, verify:bool=True) -> np.array:
        if not isinstance(ids, list):
            ids = [ids]
        features =  self._idx[np.isin(self._idx[0], ids)]
        features = features[np.where(features[:,1] == int(otype))]
        
        if verify:
            # all features must be same size dim
            if features[0,2] != np.sum(features[:, 2]) / features.shape[0]:
                raise ValueError("Features dimensions are not equal")
            
        buffer = []
        for f in features:
            buffer.append([f[0], self._features[f[3]].to_numpy(dtype=object)])
        
        
        return np.array(buffer, dtype=object)

        
            
        
        
            
        
        
        
        
        
        

        


        
        
        
        
        
    
    
    
