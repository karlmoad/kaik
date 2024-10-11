import re
import json
import pandas as pd
import numpy as np
from collections import abc
from typing import Union
from enum import Enum
from common.exceptions import FormatException
from common.utils.numeric import *


class Feature(object):
    __slots__ = ('_data', '_size', '_default', '_ftype')

    def __init__(self):
        self._data = None
        self._size = 0
        self._default = 0
        self._ftype = Feature.FeatureType.UNKNOWN

    class FeatureType(Enum):
        UNKNOWN = 0
        SPARSE = 1
        DENSE = 2

    def __getstate__(self):
        return {'data': self._data, 'size': self._size, 'default': self._default, 'ftype': self._ftype}

    def __setstate__(self, state):
        self._data = state['data']
        self._size = state['size']
        self._default = state['default']
        self._ftype = state['ftype']

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        if isinstance(self._data, dict):
            return self._data[str(idx)]
        elif isinstance(self._data, list):
            return self._data[int(idx)]
        else:
            IndexError("index out of range or invalid")

        return self._data[idx]

    def __iter__(self):
        return iter(self.__get_as_list())

    def __get_as_list(self) -> list:
        if self._ftype == Feature.FeatureType.DENSE:
            return self._data
        else:
            arr = [self._default] * self._size
            for k, v in self._data.items():
                for i in v:
                    arr[i] = parse_numeric(k)
            return arr

    def to_numpy(self, dtype: np.dtype) -> np.array:
        return np.array(self.__get_as_list(), dtype=dtype)

    def type(self) -> FeatureType:
        return self._ftype

    @classmethod
    def from_string(cls, string: str) -> Union['Feature', None]:
        string = WHITE_SPACE.sub('', string)  # remove whitespace
        inst = cls()
        if SPARSE_PATTERN.fullmatch(string) is not None:
            header = SPARSE_HEADER.search(string)
            defs = SPARSE_DEFINITION.findall(string)
            data = {}
            for d in defs:
                def_m_val = VALUE.search(d)
                def_m_arr = ARRAY.search(d)
                data[def_m_val.group('value')] = json.loads(def_m_arr.group('array'))

            inst._data = data
            inst._ftype = cls.FeatureType.SPARSE
            inst._size = int(header.group('dim'))
            inst._default = parse_numeric(header.group('def'))

            return inst

        elif DENSE_PATTERN.fullmatch(string) is not None:
            inst._data = json.loads(ARRAY.search(string).group('array'))
            inst._ftype = cls.FeatureType.DENSE
            inst._size = len(inst._data)
            return inst
        else:
            return None