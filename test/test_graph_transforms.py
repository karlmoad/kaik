from graph.transforms.core import InputMappingTransform, ForceBidirectional
from common.utils.test_utils import load_test_dataframe_from_csv


class TestGraphTransforms:
    def __init__(self):
        self.data = None
        self.mapping = None
        self.defaults = None
        
    def _init_self(self):
        self.data = {
            "nodes": load_test_dataframe_from_csv('test_nodes.csv', sep="|"),
            "edges": load_test_dataframe_from_csv('test_edges.csv', sep="|")
        }
        
        self.mapping = {
            'edges': {'source': 'ORIGIN', 'target': 'TARGET', 'graph_id': 'GRAPH', 'edge_type': 'TYPE', 'class': 'CLASS',
                      'features': 'FEATURES'},
            'nodes': {'label': 'LABEL', 'node_type': 'TYPE', 'features': 'FEATURES', "class": "CLASS"},
        }
        
        self.defaults = {
            'nodes': {"node_type": "NODE", "features": "-NONE-", "class": "NODE"},
            'edges': {"edge_type": "EDGE", "graph_id": 0, "features": "-NONE-", "class": "EDGE"},
        }


    def test_transforms(self):
        self.t_input_mapping()
        self.t_input_mapping_content()
        self.t_force_bidirectional()

    def t_input_mapping(self):
        self._init_self()
        # orig copy
        orig_nodes = self.data['nodes'].copy(deep=True)
        orig_edges = self.data['edges'].copy(deep=True)
        
        imt = InputMappingTransform(mapping=self.mapping, defaults=self.defaults)
        imt(self.data)
        
        edge_cols_r = ['source', 'target', 'graph_id', 'edge_type', 'class', 'features']
        node_cols_r = ['label', 'node_type', 'features', 'class']
        
        # assert columns required are present
        for col in edge_cols_r:
            assert col in self.data['edges'].columns
        
        for col in node_cols_r:
            assert col in self.data['nodes'].columns
        
        # assert data size is same in vs out
        assert self.data['edges'].shape[0] == orig_edges.shape[0]
        assert self.data['nodes'].shape[0] == orig_nodes.shape[0]


    def t_input_mapping_content(self):
        self._init_self()
        
        # orig copy
        orig_nodes = self.data['nodes'].copy(deep=True)
        orig_edges = self.data['edges'].copy(deep=True)
        
        imt = InputMappingTransform(mapping=self.mapping, defaults=self.defaults)
        imt(self.data)
        
        for row_o, row_t in zip(orig_edges.iterrows(), self.data['edges'].iterrows()):
            assert (
                    str(row_o[1]['ORIGIN']).upper() == row_t[1]['source'] and
                    str(row_o[1]['TARGET']).upper() == row_t[1]['target'] and
                    str(row_o[1]['TYPE']).upper() == row_t[1]['edge_type'] and
                    row_t[1]['features'] == "-NONE-" and
                    row_t[1]['class'] == "EDGE" and
                    row_t[1]['graph_id'] == 0
            )
        
        for row_o, row_t in zip(orig_nodes.iterrows(), self.data['nodes'].iterrows()):
            assert (
                    str(row_o[1]['LABEL']).upper() == row_t[1]['label'] and
                    str(row_o[1]['TYPE']).upper() == row_t[1]['node_type'] and
                    str(row_o[1]['FEATURES']).upper() == row_t[1]['features'] and
                    str(row_t[1]['class']).upper() == 'NODE'
            )

    def t_force_bidirectional(self):
        orig_edges = self.data['edges'].copy(deep=True)
        
        transform = ForceBidirectional()
        transform(self.data)
        
        et = self.data['edges']
        for _, row in orig_edges.iterrows():
            dsr = et[(et['source'] == row['TARGET']) & (et['target'] == row['ORIGIN'])]
            assert dsr.shape[0] >= 1