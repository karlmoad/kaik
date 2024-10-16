from graph.transforms.core import InputMappingTransform, ForceBidirectional
from common.utils.test_utils import load_test_dataframe_from_csv

# transforms = [InputMappingTransform(self._mapping_spec, self._field_defaults), ForceBidirectional(),PurgeSelfLoops(), PurgeIsolatedNodes()]


class TestGraphTransforms:
    @staticmethod
    def _initialize_data():
        data = {
            "nodes": load_test_dataframe_from_csv('test_nodes.csv', sep="|"),
            "edges": load_test_dataframe_from_csv('test_edges.csv', sep="|")
        }
        
        field_map= {
            'edges': {'source': 'ORIGIN', 'target': 'TARGET', 'graph_id': 'GRAPH', 'edge_type': 'TYPE', 'class': 'CLASS',
                      'features': 'FEATURES'},
            'nodes': {'label': 'LABEL', 'node_type': 'TYPE', 'features': 'FEATURES', "class": "CLASS"},
        }
        
        defaults = {
            'nodes': {"node_type": "NODE", "features": "-NONE-", "class": "NODE"},
            'edges': {"edge_type": "EDGE", "graph_id": 0, "features": "-NONE-", "class": "EDGE"},
        }
        
        return data, field_map, defaults
        
    def test_graph_transforms(self):
        data, mapping, defaults = TestGraphTransforms._initialize_data()
        self.t_input_mapping(data, mapping, defaults)
        self.t_force_bidirectional(data, mapping, defaults)
        
        
    def t_input_mapping(self, data, mapping, defaults):
        # orig copy
        orig_nodes = data['nodes'].copy(deep=True)
        orig_edges = data['edges'].copy(deep=True)
        
        imt = InputMappingTransform(mapping=mapping, defaults=defaults)
        imt(data)
        
        edge_cols_r = ['source', 'target', 'graph_id', 'edge_type', 'class', 'features']
        node_cols_r = ['label', 'node_type', 'features', 'class']
        
        # assert columns required are present
        for col in edge_cols_r:
            assert col in data['edges'].columns
        
        for col in node_cols_r:
            assert col in data['nodes'].columns
        
        # assert data size is same in vs out
        assert data['edges'].shape[0] == orig_edges.shape[0]
        assert data['nodes'].shape[0] == orig_nodes.shape[0]
        
        for row_o, row_t in zip(orig_edges.iterrows(), data['edges'].iterrows()):
            batch = [bool(str(row_o[1]['ORIGIN']).upper() == row_t[1]['source']),
                     bool(str(row_o[1]['TARGET']).upper() == row_t[1]['target']),
                     bool(str(row_o[1]['TYPE']).upper() == row_t[1]['edge_type']),
                     bool(row_t[1]['features'] == "-NONE-"),
                     bool(row_t[1]['class'] == "EDGE"),
                     bool(row_t[1]['graph_id'] == 0)]
            
            assert all(batch)
        
        for row_o, row_t in zip(orig_nodes.iterrows(), data['nodes'].iterrows()):
            batch = [
                    bool(str(row_o[1]['LABEL']).upper() == row_t[1]['label']),
                    bool(str(row_o[1]['TYPE']).upper() == row_t[1]['node_type']),
                    bool(str(row_o[1]['FEATURES']).upper() == row_t[1]['features']),
                    bool(str(row_o[1]['CLASS']).upper() == row_t[1]['class'])
            ]
            
            assert all(batch)
        
    def t_force_bidirectional(self, data, mapping=None, defaults=None):
        orig_edges = data['edges'].copy(deep=True)
        
        transform = ForceBidirectional()
        transform(data)
        
        et = data['edges']
        for _, row in orig_edges.iterrows():
            dsr = et[(et['source'] == row['target']) & (et['target'] == row['source'])]
            assert dsr.shape[0] >= 1