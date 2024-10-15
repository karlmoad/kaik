from graph.transforms.core import InputMappingTransform, ForceBidirectional
from common.utils.test_utils import load_test_dataframe_from_csv

def _init_test_data():
    data = {
        "nodes": load_test_dataframe_from_csv('nodes.csv', sep="|"),
        "edges": load_test_dataframe_from_csv('edges.csv', sep="|")
    }
    
    mapping = {
        'edges': {'source': 'ORIGIN', 'target': 'TARGET', 'graph_id': 'GRAPH', 'edge_type': 'TYPE', 'class': 'CLASS',
                  'features': 'FEATURES'},
        'nodes': {'label': 'LABEL', 'node_type': 'TYPE', 'features': 'FEATURES', "class": "CLASS"},
    }
    
    defaults = {
        'nodes': {"node_type": "NODE", "features": "-NONE-", "class": "NODE"},
        'edges': {"edge_type": "EDGE", "graph_id": 0, "features": "-NONE-", "class": "EDGE"},
    }
    
    return data, mapping, defaults

def test_input_mapping():
    data, mapping,defaults = _init_test_data()
    
    # orig copy
    orig_nodes = data['nodes'].copy(deep=True)
    orig_edges = data['edges'].copy(deep=True)
    
    imt = InputMappingTransform(mapping=mapping, defaults=defaults)
    imt(data)
    
    edge_cols_r = ['source','target','graph_id','edge_type','class','features']
    node_cols_r = ['label','node_type','features','class']
    
    # assert columns required are present
    for col in edge_cols_r:
        assert col in data['edges'].columns

    for col in node_cols_r:
        assert col in data['nodes'].columns
        
    # assert data size is same in vs out
    assert data['edges'].shape[0] == orig_edges.shape[0]
    assert data['nodes'].shape[0] == orig_nodes.shape[0]

    
def test_input_mapping_content():
    
    data, mapping, defaults = _init_test_data()
    
    #orig copy
    orig_nodes = data['nodes'].copy(deep=True)
    orig_edges = data['edges'].copy(deep=True)
    
    imt = InputMappingTransform(mapping=mapping, defaults=defaults)
    imt(data)
    
    for row_o, row_t in zip(orig_edges.iterrows(), data['edges'].iterrows()):
        assert (
            str(row_o[1]['ORIGIN']).upper() == row_t[1]['source'] and
            str(row_o[1]['TARGET']).upper() == row_t[1]['target'] and
            str(row_o[1]['TYPE']).upper() == row_t[1]['edge_type'] and
            row_t[1]['features'] == "-NONE-" and
            row_t[1]['class'] == "EDGE" and
            row_t[1]['graph_id']== 0
        )
        
    for row_o, row_t in zip(orig_nodes.iterrows(), data['nodes'].iterrows()):
        assert (
            str(row_o[1]['LABEL']).upper() == row_t[1]['label'] and
            str(row_o[1]['TYPE']).upper() == row_t[1]['node_type'] and
            str(row_o[1]['FEATURES']).upper() == row_t[1]['features'] and
            str(row_t[1]['class']).upper() =='NODE'
        )
        
    
def test_force_bidirectional():
    data, _, _ = _init_test_data()
    orig_edges = data['edges'].copy(deep=True)
    
    transform = ForceBidirectional()
    transform(data)

    et = data['edges']
    for _, row in orig_edges.iterrows():
        dsr = et[(et['source'] == row['target'])&(et['target'] == row['source'])]
        assert dsr.shape[0] >=1
        
    