from abc import ABC, abstractmethod

class BaseTransform(ABC):
    __slots__ = ['_name','_desc']

    def __init__(self, name:str, desc:str):
        self._name = name
        self._desc = desc

    @abstractmethod
    def __call__(self, data):
        pass

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._desc
