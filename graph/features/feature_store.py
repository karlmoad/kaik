from typing import Union
import numpy as np
from numpy import ndarray
from graph import GraphObjectType


class EncodedFeatureArray(object):
    __slots__ = ['_dim', '_dtype', '_type', '_def_val', '_data']

    def __init__(self, dim_size: int, dtype: np.dtype = np.int64, otype: str = "0", def_val: int = -1):
        self._dim = dim_size
        self._dtype = dtype
        self._type = otype
        self._def_val = def_val
        self._data = {}

    def __contains__(self, item):
        return item in self._data

    def __getitem__(self, item):
        arr = [self._def_val] * self._dim
        features = self._data[item]
        for feature in features:
            val, pos = feature
            for i in range(len(pos)):
                arr[pos[i]] = val
        return arr

    def __setitem__(self, key, value):
        self._data[key] = value

    def __getstate__(self):
        return {
            "dim": self._dim,
            "dtype": self._dtype,
            "def_val": self._def_val,
            "type": self._type,
            "data": self._data,
        }

    def __setstate__(self, state):
        self._dim = state["dim"]
        self._dtype = state["dtype"]
        self._def_val = state["def_val"]
        self._data = state["data"]
        self._type = state["type"]

    def is_type(self, otype: int):
        return self._type == otype

    def all(self, /, dtype: np.dtype = None):
        buffer = []
        for key in list(self._data.keys()):
            arr = [key]
            arr.extend(self[key])
            buffer.append(arr)
        return np.array(buffer, dtype=dtype if dtype is not None else self._dtype)


class FeatureStore(object):
    __slots__ = '_features'

    def __init__(self):
        self._features = {}

    def __getstate__(self):
        return {
            'features': self._features,
        }

    def __setstate__(self, state):
        self._features = state['features']

    def add_feature_spec(self, graph_obj_type: GraphObjectType, sub_obj_type: int, dim: int,
                         /, dtype: np.dtype = int, def_val: int = -1):

        if str(graph_obj_type) not in self._features:
            self._features[str(graph_obj_type)] = {}

        if str(sub_obj_type) not in self._features[str(graph_obj_type)]:
            self._features[str(graph_obj_type)][str(sub_obj_type)] = EncodedFeatureArray(dim, dtype, sub_obj_type,
                                                                                         def_val)

    def __verify_feature(self, graph_obj_type: GraphObjectType, sub_obj_type: int):
        if str(graph_obj_type) not in self._features or str(sub_obj_type) not in self._features[str(graph_obj_type)]:
            raise IndexError(f'feature set for type [{sub_obj_type}] has no definition')

    def add_feature(self, graph_obj_type: GraphObjectType, sub_obj_type: int, key, value):
        self.__verify_feature(graph_obj_type, sub_obj_type)

        self._features[str(graph_obj_type)][str(sub_obj_type)][key] = value

    def get_feature(self, graph_obj_type: GraphObjectType, sub_obj_type: int, key: str):
        self.__verify_feature(graph_obj_type, sub_obj_type)

        return self._features[str(graph_obj_type)][str(sub_obj_type)][key]

    def get_feature_set(self, graph_obj_type: GraphObjectType, sub_obj_type: int, /, dtype: np.dtype = None):
        self.__verify_feature(graph_obj_type, sub_obj_type)

        return self._features[str(graph_obj_type)][str(sub_obj_type)].all(dtype=dtype)
