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
            
    def get_feature(self, id:Union[list[int],int], otype:GraphObjectType) -> np.array:
        pass #TODO
    
    
    
