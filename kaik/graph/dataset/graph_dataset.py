import collections
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from kaik.common.utils.pandas_utils import row_get_or_default, default_if_none
from tqdm import tqdm
from kaik.graph import Graph
from kaik.graph.transforms import EncodingTransform

class GraphDataset(Dataset):
    __slots__ =('_root_dir','_graph','_mapping','_field_defaults','_force_rebuild')
    def __init__(self, root_dir: str, **kwargs):
        self._root_dir = Path(root_dir)
        self._graph = Graph()
        self._mapping = None
        self._force_rebuild = kwargs.get('force_rebuild', False)
        self._field_defaults = {
            'nodes':{"node_type":"NODE", "features":"-NONE-","class":"NODE"},
            'edges':{"edge_type":"EDGE", "graph_id":0, "features":"-NONE-", "class":"EDGE"},
        }

        if 'mapping' in kwargs:
            self._mapping = kwargs.get('mapping', None)

    def _evaluate(self,file_metadata:dict, **kwargs):
        if self._force_rebuild:
            #go straight to build
            self._build(file_metadata, **kwargs)
        else:
            #identify if graph file(s) exist and if so load them
            #if file error run build
            try:
                with tqdm(total=1, position=0, leave=True, desc="Loading graph dataset from file") as pbar:
                    self._graph.load(f"{self._root_dir}/{file_metadata['graph']}")
                    pbar.update(1)
            except FileNotFoundError:
                self._build(file_metadata, **kwargs)

    def _build(self,file_metadata:dict,/,
               transforms:list=None, **kwargs):
        if 'edges' not in file_metadata:
            raise ValueError('minimally edge metadata must be provided')

        if self._mapping is None or not isinstance(self._mapping, dict):
            raise ValueError('Mapping must be an instance of dict')
        
        # add encoding steps to end of transforms list
        transforms.extend([EncodingTransform('class','nodes'),
                          EncodingTransform('class','edges'),
                          EncodingTransform('node_type','nodes'),
                          EncodingTransform('edge_type','edges')])
        
        with tqdm(total=len(transforms)+3, position=0, desc="Preparing Dataset") as pbar:

            pbar.set_description("Loading files")
            pbar.refresh()

            if self._root_dir.is_dir() is None or not self._root_dir.exists():
                raise FileNotFoundError(f'{self._root_dir} is not a directory or does not exist')

            if 'nodes' in file_metadata:
                if not self._root_dir.joinpath(file_metadata['nodes']).exists():
                    raise FileNotFoundError("nodes does not exist")

            if not self._root_dir.joinpath(file_metadata['edges']).exists():
                raise FileNotFoundError("edges does not exist")

            data = collections.defaultdict(None)

            data['nodes'] = pd.read_csv(self._root_dir.joinpath(file_metadata['nodes']))
            data['edges'] = pd.read_csv(self._root_dir.joinpath(file_metadata['edges']))

            pbar.update(1)

            for transform in transforms:
                pbar.set_description(transform.description)
                pbar.refresh()
                transform(data)
                pbar.update(1)

            pbar.set_description('Building Graph')
            pbar.refresh()
            self._graph.build(data['nodes'], data['edges'])
            pbar.update(1)
            pbar.set_description('Saving Graph')
            pbar.refresh()
            self._graph.save(f"{self._root_dir}/{file_metadata['graph']}")
            pbar.update(1)

    @property
    def graph(self):
        return self._graph

    @property
    def _mapping_spec(self):
        return self._mapping

    def __len__(self):
        pass  # not implemented at this time

    def __getitem__(self, idx):
        pass  # not implemented at this time







