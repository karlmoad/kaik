from kaik.graph import Graph
import numpy as np
import pandas as pd
from kaik.common.utils.test_utils import load_test_dataframe_from_csv
from kaik.common.utils.serialization_utils import serialize, deserialize
from kaik.graph.transforms import *
import logging

class TestGraphObject:
    def test_graph_object(self, caplog):
        gtw = GraphTestWrapper()
        gtw.t_build_graph()
        gtw.t_serialization()
    
class GraphTestWrapper:
    def __init__(self):
        self.data = {
            "nodes": load_test_dataframe_from_csv('test_nodes.csv', sep="|"),
            "edges": load_test_dataframe_from_csv('test_edges.csv', sep="|")
        }
        
        self.field_map = {
            'edges': {'source': 'ORIGIN', 'target': 'TARGET', 'graph_id': 'GRAPH', 'edge_type': 'TYPE',
                      'class': 'CLASS',
                      'features': 'FEATURES'},
            'nodes': {'label': 'LABEL', 'node_type': 'TYPE', 'features': 'FEATURES', "class": "CLASS"},
        }
        
        self.defaults = {
            'nodes': {"node_type": "NODE", "features": "-NONE-", "class": "NODE"},
            'edges': {"edge_type": "EDGE", "graph_id": 0, "features": "-NONE-", "class": "EDGE"},
        }
        self.graph = None
        self.log = logging.getLogger(__name__)
        
        
    def t_build_graph(self):
        
        transforms = [
            InputMappingTransform(self.field_map, self.defaults),
            AlignReferences(),
            EncodingTransform('class', 'nodes'),
            EncodingTransform('class', 'edges'),
            EncodingTransform('node_type', 'nodes'),
            EncodingTransform('edge_type', 'edges')]
        
        
        for tf in transforms:
            tf(self.data)
            
            
        g = Graph()
        assert g.nodes is None
        
        g.build(self.data)
        assert g.nodes is not None and g.nodes.shape[0] > 0
        
        self.graph = g
        self.log.info("Build graph test : [SUCCESS]")
        
    def t_graph_contents(self):
        GraphTestWrapper._content_eval(self.graph)
        self.log.info("Graph content test : [SUCCESS]")
        
    @staticmethod
    def _content_eval(graph):
        assert graph.nodes.shape[0] == 9
        assert graph.adjacency_matrix.shape[0] == 1
        assert graph.adjacency_matrix.shape[1] == 4
        assert graph.adjacency_matrix.shape[2] == 9 and graph.adjacency_matrix.shape[3] == 9
        
        assert graph.heterogeneous
        assert not graph.weighted
        assert not graph.undirected
        
        assert len(graph.features) == 9
    
    def t_serialization(self):
        g_s = serialize(self.graph)
        d_g = deserialize(g_s)
        
        assert np.array_equal(self.graph.adjacency_matrix, d_g.adjacency_matrix)
        GraphTestWrapper._content_eval(d_g)
        self.log.info("Graph serialization test : [SUCCESS]")
