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