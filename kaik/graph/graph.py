import pandas as pd
import numpy as np
from kaik.common.utils.pandas_utils import row_get_or_default, default_if_none
from kaik.graph import GraphObjectType
from kaik.graph.features.feature_store import FeatureStore, Feature
from kaik.common.utils.sequence_utils import IntSequence

class Graph(object):
    __slots__ = ('_nodes', '_adj', '_meta_paths', '_graphs_idx', '_features', '_data')

    def __init__(self):
        self._nodes = None
        self._adj = None
        self._meta_paths = None
        self._graphs_idx = None
        self._features = None
        self._data = None

    def __getstate__(self):
        state = {
            'nodes': self._nodes.tolist(),
            'adj': self._adj.tolist(),
            'meta_paths': self._meta_paths,
            'graphs_idx': self._graphs_idx.tolist(),
            'features': self._features,
            'data': self._data
        }
        return state
        
    def __setstate__(self, state):
        self._nodes = np.array(state['nodes'], dtype=object)
        self._adj = np.array(state['adj'], dtype=object)
        self._meta_paths = state['meta_paths']
        self._graphs_idx = np.array(state['graphs_idx'], dtype=object)
        self._features = state['features']
        self._data = state['data']
        

    def build(self, data):
        self._data = data
        nodes = self._data['nodes']
        edges = self._data['edges']
        
        assert isinstance(nodes, pd.DataFrame), 'Invalid input data type for node data'
        assert isinstance(edges, pd.DataFrame), 'Invalid input data type for edge data'

        self._meta_paths = None
        self._features = FeatureStore(nodes.shape[0], edges.shape[0])

        self._graphs_idx = edges['graph_id'].value_counts().keys().to_numpy(object, True, None)

        # add node features to feature store
        for i, row in nodes.iterrows():
            self.__set_feature(int(row['id']), GraphObjectType.NODE, row['features'])
        
        #set final nodes list
        self._nodes = nodes[['id','node_type_encoded','class_encoded']].to_numpy(dtype=np.int64)
        
        #multi dim adj info matrix shape(a,b,c,c), a)num graphs, b)info slice, c) num nodes
        #information slice dim = 4 , 0)weights, 1)types 2)features 3)classes
        self._adj = np.full((self._graphs_idx.shape[0], 4, self._nodes.shape[0], self._nodes.shape[0]), -1.0,
                            dtype=object)

        edges.sort_values(['graph_id', 'source_id', 'target_id'], inplace=True)
        sequencer = IntSequence()
        for g in range(self._graphs_idx.shape[0]):
            graph_edges = edges[edges['graph_id'] == self._graphs_idx[g]]
            for _, edg in graph_edges.iterrows():
                m, n = int(default_if_none(edg['source_id'], -1)), int(default_if_none(edg['target_id'], -1))
                wtfc = row_get_or_default(edg, ['weight', 'edge_type_encoded','features','class_encoded'], -1)
                if m != -1 and n != -1:
                    feature_ref = self.__set_feature(sequencer(), GraphObjectType.EDGE, wtfc[2]) if wtfc[2] != -1 else None
                    self._adj[g, 0, m, n] = wtfc[0] + 1 if wtfc[0] > 0 else 1  # default all weights to at least 1, if weight present shift
                    self._adj[g, 1, m, n] = wtfc[1]  #edge type
                    self._adj[g, 2, m, n] = feature_ref if feature_ref is not None else -1
                    self._adj[g, 3, m, n] = wtfc[3]  #edge classes
                else:
                    self._adj[g, 0, m, n] = 0
                    self._adj[g, 1, m, n] = -1
                    self._adj[g, 2, m, n] = -1
                    self._adj[g, 3, m, n] = -1

        # meta paths identification
        et = {r['edge_type_encoded']: r['edge_type'] for _,r in self.edge_types.iterrows()}
        nt = {r['node_type_encoded']: r['node_type'] for _,r in self.node_types.iterrows()}
        meta_paths = []

        for g in range(self._adj.shape[0]):
            etm = self._adj[g, 1, :, :]
            for i in range(nodes.shape[0]):
                for j in range(nodes.shape[0]):
                    etype = etm[i, j]
                    if etype >= 0:
                        meta_paths.append({"source": nt[self._nodes[i, 1]], "edge": et[etype], "target": nt[self._nodes[j, 1]]})

        self._meta_paths = pd.DataFrame(meta_paths).drop_duplicates(ignore_index=True).to_numpy(dtype=object, copy=True)


    def __call__(self):
        pass

    def __set_feature(self, iden:int, otype:GraphObjectType, value:str):
        if len(value.strip()) > 0 and value != '-NONE-':
            self._features.add_feature(iden, otype, Feature.from_string(value))
            return iden
        else:
            return None
        
    @property
    def adjacency_matrix(self):
        return self._adj

    @property
    def nodes(self):
        return self._nodes

    @property
    def weighted(self):
        return len(np.where(self._adj[:, 0, :, :] != 1)[0]) > 0

    @property
    def undirected(self):
        return all([np.array_equal(self._adj[i, 0, :, :], self._adj[i, 0, :, :].transpose()) for i in
                                range(self._adj.shape[0])])

    @property
    def meta_paths(self):
        return self._meta_paths

    @property
    def node_types(self):
        return self._data['nodes'][['node_type','node_type_encoded']].value_counts().to_frame().reset_index()
    
    @property
    def edge_types(self):
        return self._data['edges'][['edge_type','edge_type_encoded']].value_counts().to_frame().reset_index()
    
    @property
    def node_classes(self):
        return self.__get_class_list('nodes')

    @property
    def edge_classes(self):
        return self.__get_class_list('edges')

    def __get_class_list(self, t:str):
        if t in self._data:
            return self._data[t][['class', 'class_encoded']].value_counts().to_frame().reset_index()
        
    @property
    def features(self):
        return self._features

    @property
    def heterogeneous(self):
        return len(np.where(self._adj[:, 1, :, :] != -1)[0]) > 0

    def __repr__(self):
        return self.info()

    def __str__(self):
        return self.info()

    def info(self) -> str:
        s = ""

        s += "\nGraph count: {}\n".format(self._graphs_idx.shape[0])

        if self.undirected:
            s += "\nUndirected graph"
        else:
            s += "\nDirected graph"

        s += f"\nWeighted: {self.weighted}"
        s += f"\nNode Count:{self._data['nodes'].shape[0]}"
        s += f"\nEdge Count:{self._data['edges'].shape[0]}"

        s += f"\n{''.ljust(25, '-')}"

        s += "\nNode Types:"
        for _,r in self.node_types.iterrows():
            s += f"\n\t{r['node_type']}: {r['count']}"

        s += "\nNode Classes:"
        for _,r in self.node_classes.iterrows():
            s += f"\n\t{r['class']}: {r['count']}"

        s += "\n\nEdge Types:"
        for _,r in self.edge_types.iterrows():
            s += f"\n\t{r['edge_type']}: {r['count']}"

        s += "\nEdge Classes:"
        for _,r in self.edge_classes.iterrows():
            s += f"\n\t{r['class']}: {r['count']}"

        s += "\n\nMeta paths:"

        l_d = ' -- '
        if self.undirected:
            l_d = ' <- '

        for i in range(self._meta_paths.shape[0]):
            s += f"\n\t{self._meta_paths[i, 0]} {l_d} {self._meta_paths[i, 1]} -> {self._meta_paths[i, 2]}"
        return s

    def has_edge(self, origin, target, /, graph_idx=0, edge_types=None):
        if target is None:
            return False

        if edge_types is not None:
            etypes = self._map_edge_types(edge_types)
            return self._adj[graph_idx, 1, origin, target] in etypes
        else:
            return self._adj[graph_idx, 0, origin, target] >= 0

    def neighbors(self, node: int, /, graph_idx: int = 0, edge_types:list[int]=None, node_types:list[int]=None) -> np.ndarray:
        assert self._adj is not None, "Adjacency matrix can not be equal to None"
        assert self._adj.shape[2] == self._adj.shape[3], "Adjacency matrix must be square matrix"
        assert node is not None and node >= 0, "invalid node index"
        assert graph_idx is not None and graph_idx >= 0, "invalid graph index"

        # default mask to existing edges
        mask = np.isin(self._adj[graph_idx, 0, node, :], self._adj[graph_idx, 0, node, :] > 0)

        if node_types is not None:
            assert isinstance(node_types, list), "node types must be a list of integers, encoded node types"
            mask &= np.isin(self._nodes[:, 2], node_types)

        if edge_types is not None:
            assert isinstance(edge_types, list), "edge types must be a list of integers, encoded edge types"
            mask &= np.isin(self._adj[graph_idx, 1, node, :], edge_types)

        n_idx = np.argwhere(mask).flatten()
        n_wght = self._adj[graph_idx, 0, node, n_idx]

        fin = np.ndarray((2, n_idx.shape[0]), dtype=object)
        fin[0] = n_idx
        fin[1] = n_wght

        return fin

