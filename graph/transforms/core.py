import numpy as np
import pandas as pd
from transforms import BaseTransform
from common.utils.pandas_utils import map_dataframe


class InputMappingTransform(BaseTransform):
    __slots__ = ('_mapping', '_defaults')

    def __init__(self, mapping: dict, defaults: dict):
        super().__init__('InputMappingTransform', 'Applying input map transform')
        self._mapping = mapping
        self._defaults = defaults

    def __call__(self, data):
        _assert_state(data, edges=True)
        for t in ['nodes', 'edges']:
            if t in data and t in self._mapping:
                data[t] = map_dataframe(data[t], self._mapping[t],
                                        self._defaults[t] if t in self._defaults else None,
                                        apply_func=(lambda x: str(x).upper()))


class ForceBidirectional(BaseTransform):
    def __init__(self):
        super().__init__('ForceBidirectionalTransform', 'Applying force bidirectional transform')
        pass

    def __call__(self, data):
        _assert_state(data, edges=True)
        rev = data['edges'].copy(deep=True)
        rev.rename(columns={'source': 'target', 'target': 'source', 'source_id': 'target_id', 'target_id': 'source_id'},
                   inplace=True)
        new_edges = pd.concat([data['edges'], rev], ignore_index=True, axis=0)
        data['edges'] = new_edges.drop_duplicates(ignore_index=True)


class AlignReferences(BaseTransform):
    __slots__ = ['_context']

    def __init__(self):
        super().__init__('RealignReferencesTransform', 'Applying realignment of references transform')
        self._context = None

    def __call__(self, data):
        # encode an integer identifier value for nodes
        data['nodes'].reset_index(drop=True, inplace=True)
        data['nodes']['id'] = data['nodes'].index

        #realign edge source and target values to the newly identified node set
        nodes = data['nodes'].filter(['id', 'label'], axis=1)
        e = data['edges'].filter(['source', 'target', 'edge_type', 'features', 'weight', 'graph_id', 'class'], axis=1)
        e = e.merge(nodes, how="left", left_on="source", right_on="label")
        e = e.rename(columns={'id': 'source_id', 'label': 'source_label'})
        e = e.merge(nodes, how="left", left_on="target", right_on="label")
        e = e.rename(columns={'id': 'target_id', 'label': 'target_label'})
        data['edges'] = e.filter(['source', 'source_id', 'target', 'target_id',
                                  'edge_type', 'features', 'weight', 'graph_id', 'class'], axis=1).copy(deep=True)


class PurgeIsolatedNodes(BaseTransform):
    def __init__(self):
        super().__init__('PurgeIsolatedNodesTransform', 'Applying purge isolated nodes transform')

    def __call__(self, data):
        _assert_state(data, nodes=True, edges=True)
        nodes = np.array([row['label'] for _, row in data['nodes'].iterrows()])
        edges = np.unique(np.concatenate((
            np.array([row['source'] for _, row in data['edges'].iterrows()]),
            np.array([row['target'] for _, row in data['edges'].iterrows()])), axis=0))
        edges.sort(axis=0)

        diff = np.setdiff1d(nodes, edges, assume_unique=True)
        for i in range(diff.shape[0]):
            data['nodes'] = data['nodes'][data['nodes']['label'] != diff[i]]


class PurgeSelfLoops(BaseTransform):
    def __init__(self):
        super().__init__('PurgeSelfLoopsTransform', 'Applying purge self loops transform')

    def __call__(self, data):
        _assert_state(data, edges=True)
        data['edges'] = data['edges'][data['edges']['source'] != data['edges']['target']].copy(deep=True)


class InferNodesFromEdges(BaseTransform):
    def __init__(self):
        super().__init__('InferNodesFromEdgesTransform', 'Applying infer nodes from edges transform')
        pass

    def __call__(self, data):
        _assert_state(data, edges=True)
        if data['edges'].shape[0] > 0:
            n = data['edges'][['source', 'target']].values.to_numpy(object, True, None)
            n = np.concatenate((n[:, 0], n[:, 1]), axis=0)
            n = np.unique(n)
            n.sort()
            data['nodes'] = pd.DataFrame(
                [{'label': n, 'node_type': 'NODE', 'features': '-', 'class': 'NODE'} for i, n in enumerate(n)])


class FeaturesTransform(BaseTransform):
    def __init__(self):
        super().__init__('FeaturesTransform', 'Applying features transform')

    def __call__(self, data):
        _assert_state(data, edges=True, nodes=True)
        node_types = data['nodes']['node_type'].value_counts().to_frame().reset_index()
        for i, row in node_types.iterrows():
            ntrows = data['nodes'][data['nodes']['node_type'] == row['node_type']]
