class SemanticNetworkDataset(GraphDataset):
    def __init__(self, root_dir: str,/, **kwargs):
        super().__init__(root_dir, **kwargs)
        self._files = {'nodes':'base/sn_nodes.csv',
                       'edges':'base/sn_edges.csv',
                       'graph':'final/semantic_network'}

        # default mapping
        self._mapping = {
            'edges': {'source': 'ORIG', 'target': 'TARG', 'graph_id': 'GRAPH', 'edge_type': 'TYPE', 'class': 'CLASS',
                      'features': 'FEATURES'},
            'nodes': {'label': 'LABEL', 'node_type': 'TYPE', 'features': 'FEATURES', 'graph_id': 'GRAPH', "class": "CLASS"},
        }

        # see if minimally edges file present if not alert download
        file_meta = {'graph': self._files['graph']}
        transforms = [InputMappingTransform(self._mapping_spec, self._field_defaults), ForceBidirectional(),PurgeSelfLoops(), PurgeIsolatedNodes()]
        if Path(f"{self._root_dir}/{self._files['edges']}").exists():
            file_meta['edges'] = f"{self._root_dir}/{self._files['edges']}"
        else:
            raise FileNotFoundError(f"{self._root_dir}/{self._files['edges']} does not exist, please download or place file in directory path")

        if Path(f"{self._root_dir}/{self._files['nodes']}").exists():
            file_meta['nodes'] = f"{self._root_dir}/{self._files['nodes']}"
        else:
            transforms.append(InferNodesFromEdges())

        transforms.append(AlignReferences())

        self._evaluate(file_meta, transforms=transforms, **kwargs)

    def __len__(self):
        pass  # not implemented at this time

    def __getitem__(self, idx):
        pass #not implemented at this time
