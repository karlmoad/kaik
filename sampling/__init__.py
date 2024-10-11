from abc import ABC, abstractmethod
from common.utils.randomization import RandomState
from enum import Enum


class Subset(Enum):
    TRAIN = 1
    TEST = 2
    VALIDATION = 3


class BaseSampler(ABC):
    __slots__ = '_rs', '_dataset'

    def __init__(self, dataset: GraphDataset, seed: int = None):
        self._rs = RandomState(seed)
        self._dataset = dataset

    @abstractmethod
    def generate(self, **kwargs):
        pass

    def __call__(self, **kwargs):
        return self.generate(**kwargs)

    @abstractmethod
    def sample(self, **kwargs):
        pass

    @abstractmethod
    def loader(self, subset: Subset, **kwargs):
        pass



