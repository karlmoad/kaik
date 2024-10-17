from enum import Enum

class GraphObjectType(Enum):
    NODE = 1
    EDGE = 2
    
from .graph import Graph

__all__ = [Graph, GraphObjectType]