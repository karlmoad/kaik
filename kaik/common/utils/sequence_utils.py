from abc import ABC, abstractmethod
from typing import Union

class Sequence(ABC):
    @abstractmethod
    def __call__(self)->Union[int, float, str]:
        pass
    
class IntSequence(Sequence):
    __slots__ = ['_seq']
    def __init__(self, start:int=0):
        self._seq = start
    
    def __call__(self):
        hold = self._seq
        self._seq += 1
        return hold
    
        