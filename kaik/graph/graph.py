import pandas as pd
import numpy as np
import json
from pathlib import Path
from kaik.common.utils.pandas_utils import row_get_or_default, default_if_none
from kaik.graph.features.feature_store import FeatureStore
from kaik.common.utils.serialization_utils import serialize, deserialize

class Graph(object):
    __slots__ = ('_nodes', '_adj', '_weighted', '_heterogeneous',
                 '_undirected', '_type_indexes',
                 '_meta_paths', '_graphs_idx', '_counts', '_features')

    # mask to define which attr are std vs numpy serialization, True = numpy
    __mask__ = (True, True, False, False, False, False, True, True, False, False)

    def __init__(self):
        self._nodes = None
        self._adj = None
        self._weighted = False
        self._heterogeneous = False
        self._undirected = False
        self._type_indexes = {'nodes': {}, 'edges': {}, 'node_classes': {}, 'edge_classes': {}}
        self._meta_paths = None
        self._graphs_idx = None
        self._counts = (0, 0)
        self._features = FeatureStore()

    def save(self, path: str):
        g_file = f'{path}.g'
        npz_file = f'{path}.npz'

        metadata = {}
        for ms in Graph._get_serialization_fields('std'):
            metadata[ms] = self.__getattribute__(ms)

        if self._features is not None:
            metadata['features'] = serialize(self._features)

        # add current npz fields for deserialization
        metadata['npz'] = Graph._get_serialization_fields('numpy').tolist()

        with open(g_file, 'w') as meta_file:
            json.dump(metadata, meta_file)

        npz_objects = {}
        for npo in metadata['npz']:
            npz_objects[npo] = self.__getattribute__(npo)

        np.savez_compressed(npz_file, **npz_objects)

    def load(self, path: str):
        g_file = f'{path}.g'
        npz_file = f'{path}.npz'
        if not Path(g_file).exists() or not Path(npz_file).exists():
            raise FileNotFoundError()

        with open(g_file, 'r') as meta_file:
            meta = json.load(meta_file)

        npz = np.load(npz_file, allow_pickle=True)

        for ms in Graph._get_serialization_fields('std'):
            if ms in meta:
                self.__setattr__(ms, meta[ms])

        if 'features' in meta:
            self._features = deserialize(meta['features'])

        # it has to be there let it error if not, will set the attrs according to serialization spec at time of creation
        npz_objects = meta['npz']
        for npz_obj in npz_objects:
            if npz_obj in npz:
                self.__setattr__(npz_obj, npz[npz_obj])

    @staticmethod
    def _get_serialization_fields(type_of_attr):
        attr = np.array(Graph.__slots__, dtype=object)
        mask = np.array(Graph.__mask__, dtype=bool)

        match str(type_of_attr):
            case 'std':
                return attr[~mask]
            case 'numpy':
                return attr[mask]
            case _:
                return None

    def build(self, nodes, edges):
        assert isinstance(nodes, pd.DataFrame), 'Invalid input data type for node data'
        assert isinstance(edges, pd.DataFrame), 'Invalid input data type for edge data'
        self._weighted = False
        self._heterogeneous = False
        self._undirected = False
        self._counts = (nodes.shape[0], edges.shape[0])
        self._meta_paths = None

        self._graphs_idx = edges['graph_id'].value_counts().keys().to_numpy(object, True, None)
        self._type_indexes = {'nodes': {}, 'edges': {}, 'node_classes': {}, 'edge_classes': {}}

        for idx, row in edges['edge_type'].value_counts().to_frame().reset_index().iterrows():
            self._type_indexes['edges'][row['edge_type']] = (idx, row['count'])

        for idx, row in edges['class'].value_counts().to_frame().reset_index().iterrows():
            self._type_indexes['edge_classes'][row['class']] = (idx, row['count'])

        for idx, row in nodes['node_type'].value_counts().to_frame().reset_index().iterrows():
            self._type_indexes['nodes'][row['node_type']] = (idx, row['count'])

        for idx, row in nodes['class'].value_counts().to_frame().reset_index().iterrows():
            self._type_indexes['node_classes'][row['class']] = (idx, row['count'])

        def __encode_node_attr(row):
            if 'node_type' in row and not pd.isna(row['node_type']):
                row['type_encoded'] = self._type_indexes['nodes'][row['node_type']] if row['node_type'] in \
                                                                                       self._type_indexes[
                                                                                           'nodes'] else -1
            else:
                row['type_encoded'] = -1

            if 'edge_type' in row and not pd.isna(row['edge_type']):
                row['class_encoded'] = self._type_indexes['node_classes'][row['class']] if row['class'] in \
                                                                                           self._type_indexes[
                                                                                               'node_classes'] else -1
            else:
                row['class_encoded'] = -1

            return row

        tn = nodes[['label', 'id', 'node_type', 'features', 'class']].drop_duplicates()
        tn = tn.apply(__encode_node_attr, axis=1)

        #multi dim adj info matrix shape(a,b,c,c), a)num graphs, b)info slice, c) num nodes
        #information slice dim = 4 , 0)weights, 1)types 2)features 3)classes
        self._adj = np.full((self._graphs_idx.shape[0], 4, self._nodes.shape[0], self._nodes.shape[0]), -1.0,
                            dtype=object)

        edges.sort_values(['graph_id', 'source_id', 'target_id'], inplace=True)
        for g in range(self._graphs_idx.shape[0]):
            graph_edges = edges[edges['graph_id'] == self._graphs_idx[g]]
            for _, edg in graph_edges.iterrows():
                m, n = int(default_if_none(edg['source_id'], -1)), int(default_if_none(edg['target_id'], -1))
                wtfc = row_get_or_default(edg, ['weight', 'edge_type', 'features', 'class'], -1)
                if m != -1 and n != -1:
                    self._adj[g, 0, m, n] = wtfc[0] + 1 if wtfc[
                                                               0] > 0 else 1  # default all weights to at least 1, if weight present shift
                    self._adj[g, 1, m, n] = self._type_indexes['edges'][wtfc[1]][0] if wtfc[1] in self._type_indexes[
                        'edges'] else -1  #edge type
                    self._adj[g, 2, m, n] = wtfc[2]  #edge features
                    self._adj[g, 3, m, n] = wtfc[3]  #edge classes
                else:
                    self._adj[g, 0, m, n] = 0
                    self._adj[g, 1, m, n] = -1
                    self._adj[g, 2, m, n] = -1
                    self._adj[g, 3, m, n] = -1

        # meta paths identification
        et = {v[0]: k for k, v in self._type_indexes['edges'].items()}
        meta_paths = []

        for g in range(self._adj.shape[0]):
            etm = self._adj[g, 1, :, :]
            for i in range(etm.shape[0]):
                for j in range(etm.shape[1]):
                    etype = etm[i, j]
                    if etype >= 0:
                        meta_paths.append({"source": self._nodes[i, 2], "edge": et[etype], "target": self._nodes[j, 2]})

        self._meta_paths = pd.DataFrame(meta_paths).drop_duplicates(ignore_index=True).to_numpy(dtype=object, copy=True)
        self.__assess()

    def __call__(self):
        self.__assess()

    def __assess(self):
        self._weighted = len(np.where(self._adj[:, 0, :, :] != 1)[0]) > 0
        self._heterogeneous = len(np.where(self._adj[:, 1, :, :] != -1)[0]) > 0
        self._undirected = all([np.array_equal(self._adj[i, 0, :, :], self._adj[i, 0, :, :].transpose()) for i in
                                range(self._adj.shape[0])])

    @property
    def adjacency_matrix(self):
        return self._adj

    @property
    def nodes(self):
        return self._nodes

    @property
    def weighted(self):
        return self._weighted

    @property
    def meta_paths(self):
        return self._meta_paths

    @property
    def node_types(self):
        return self._type_indexes['nodes']

    def edge_types(self):
        return self._type_indexes['edges']

    @property
    def heterogeneous(self):
        return self._heterogeneous

    def __repr__(self):
        return self.info()

    def __str__(self):
        return self.info()

    def info(self) -> str:
        s = ""

        s += "\nGraph count: {}\n".format(self._graphs_idx.shape[0])

        if self._undirected:
            s += "\nUndirected graph"
        else:
            s += "\nDirected graph"

        s += f"\nWeighted: {self._weighted}"
        s += f"\nNode Count:{self._counts[0]}"
        s += f"\nEdge Count:{self._counts[1]}"

        s += f"\n{''.ljust(25, '-')}"

        s += "\nNode Types:"
        for k, v in self._type_indexes['nodes'].items():
            s += f"\n\t{k}: {v[1]}"

        s += "\nNode Classes:"
        for k, v in self._type_indexes['node_classes'].items():
            s += f"\n\t{k}: {v[1]}"

        s += "\n\nEdge Types:"
        for k, v in self._type_indexes['edges'].items():
            s += f"\n\t{k}: {v[1]}"

        s += "\nEdge Classes:"
        for k, v in self._type_indexes['edge_classes'].items():
            s += f"\n\t{k}: {v[1]}"

        s += "\n\nMeta paths:"

        l_d = ' -- '
        if self._undirected:
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

    def neighbors(self, node: int, /, graph_idx: int = 0, edge_types=None, node_types=None) -> np.ndarray:
        assert self._adj is not None, "Adjacency matrix can not be equal to None"
        assert self._adj.shape[2] == self._adj.shape[3], "Adjacency matrix must be square matrix"
        assert node is not None and node >= 0, "invalid node index"
        assert graph_idx is not None and graph_idx >= 0, "invalid graph index"

        # default mask to existing edges
        mask = np.isin(self._adj[graph_idx, 0, node, :], self._adj[graph_idx, 0, node, :] > 0)

        if node_types is not None:
            mask &= np.isin(self._nodes[:, 2], node_types)

        if edge_types is not None:
            etypes = self._map_edge_types(edge_types)  #convert edge types to integer idx
            mask &= np.isin(self._adj[graph_idx, 1, node, :], etypes)

        n_idx = np.argwhere(mask).flatten()
        n_wght = self._adj[graph_idx, 0, node, n_idx]

        fin = np.ndarray((2, n_idx.shape[0]), dtype=object)
        fin[0] = n_idx
        fin[1] = n_wght

        return fin

    def _map_edge_types(self, edge_types):
        return [v[0] for k, v in self._type_indexes['edges'].items() if k in edge_types]
