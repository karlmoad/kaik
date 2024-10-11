import numpy as np
import torch
import warnings
from collections import defaultdict, deque
import pandas as pd
from abc import ABC, abstractmethod
from torch.cuda import graph
from tqdm import tqdm
from graph import Graph
from common.utils.randomization import RandomState, is_int_in_range
from common.utils.iteration import is_iterable_collection
from common.utils.strings import comma_sep
from graph import GraphDataset
from sampling import BaseSampler
from typing import List, Tuple, Union
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


class BiasedRandomWalk(BaseSampler, Dataset):
    __slots__ = ('num_walks', 'walk_length', 'p', 'q', '_neg_samples', '_neg_mult', '_walks', '_context', '_idx',
                 '_loader', '_device', '_subset_mask')

    def __init__(self, dataset: GraphDataset, n=None, length=None, context=None,
                 p=1.0, q=1.0, neg_samples=True, neg_multiplier=1, device=None, seed=None):
        super().__init__(dataset, seed=seed)
        self.num_walks = n
        self.walk_length = length
        self.p = p
        self.q = q
        self._neg_samples = neg_samples
        self._walks = None
        self._subset_mask = None
        self._label_mask = None
        self._context = context
        self._idx = None
        self._loader = None
        self._neg_mult = neg_multiplier
        self._device = device if device is not None else torch.device('cpu')

        BiasedRandomWalk._validate_walk_params(self._dataset.graph.nodes.shape[0], n, length)

    @staticmethod
    def _validate_walk_params(nodes, n, length):
        is_int_in_range(nodes, "nodes", min_val=1)
        is_int_in_range(n, "n", min_val=1)
        is_int_in_range(length, "length", min_val=1)

    def __getitem__(self, index):
        return self._idx[index].item()

    def __len__(self):
        return self._idx.size(0)

    def __call__(self, **kwargs):
        self.generate(**kwargs)
        return self

    def _next_iter(self, prev, cur, etypes=None, ntypes=None):
        neighbors = self._dataset.graph.neighbors(cur, edge_types=etypes)
        weights = []

        for i in range(neighbors.shape[1]):
            if self._dataset.graph.weighted and neighbors[1, i] <= 0:
                raise Exception("Edge weight invalid")

            if neighbors[0, i] == prev:
                weights.append(neighbors[1, i] / self.p)
            elif self._dataset.graph.has_edge(neighbors[0, i], prev, edge_types=etypes):
                weights.append(neighbors[1, i])
            else:
                weights.append(neighbors[1, i] / self.q)

        ws = sum(weights)
        prob = [weight / ws for weight in weights]
        return self._rs.numpy.choice(neighbors[0, :], size=1, p=prob)[0]

    def generate(self, *, n=None, length=None, context=None, p=None, q=None, seed=None,
                 edge_types=None, node_types=None, neg_samples=None, neg_mult=None, **kwargs):
        assert self._dataset is not None, "Dataset object is None"
        assert self._dataset.graph is not None, "Dataset graph object is None"
        assert isinstance(self._dataset.graph.nodes, np.ndarray), "Graph nodes collection is of invalid type"
        assert self._dataset.graph.nodes.shape[0] > 0, "length of node collection is 0"

        n = n if n is not None else self.num_walks
        length = length if length is not None else self.walk_length
        self.p = p if p is not None else self.p
        self.q = q if q is not None else self.q
        self._context = context if context is not None else self._context
        self._neg_samples = neg_samples if neg_samples is not None else self._neg_samples
        self._neg_mult = neg_mult if neg_mult is not None else self._neg_mult

        assert n is not None, "n (or number of iterations) cannot be None"
        assert length is not None, "length of random walk cannot be None"

        # check the negative multiplier if less than n default to n
        # in order to balance the dist of pos and neg samples
        # Note: multiplier can be greater,  additional neg samples will be generated
        #    but no more than n will be used in training/testing of the model due to batch def functionality
        #    greater number of neg samples will simply provide bigger pool to randomly select from
        if self._neg_mult < n and self._neg_samples:
            self._neg_mult = n

        if self._context is None:
            self._context = length

        if node_types is not None:
            nodes = (self._dataset.graph.nodes[np.isin(self._dataset.graph.nodes[:, 2], node_types)])[:, 1]
        else:
            nodes = self._dataset.graph.nodes[:, 1]

        self._idx = torch.arange(0, n * nodes.shape[0], dtype=torch.int64)

        if self._neg_samples:
            self._walks = torch.cat((self._gen_pos(n, length, nodes, **kwargs),
                                     self._gen_neg(n, length, nodes, **kwargs)), dim=0)
        else:
            self._walks = self._gen_pos(n, length, nodes, **kwargs)

        self._label_mask = torch.zeros((self._walks.size(0), 1), dtype=torch.int32)
        self._subset_mask = torch.zeros((self._walks.size(0), 1), dtype=torch.int32)

        print('Samples shape : ', self._walks.shape)  # TODO remove debug only

    def _gen_neg(self, n, length, nodes, **kwargs):
        neg_walks = []

        with tqdm(total=self._neg_mult * nodes.shape[0], position=0, leave=True) as bar:
            for i in range(self._neg_mult):
                bar.set_description(f'Generating negative sequences {i + 1} of {n}', True)
                self._rs.numpy.shuffle(nodes)
                for i2 in range(len(nodes)):
                    neg_walk = [nodes[i2]]
                    neg_walk.extend(self._rs.numpy.randint(0, nodes.shape[0], size=length - 1).tolist())
                    neg_walk.append(0)  # label indicator for negative sample
                    neg_walks.append(neg_walk)
                    bar.update(1)

        print('Num Neg Samples: ', len(neg_walks))  # TODO remove debug only
        return torch.tensor(neg_walks, dtype=torch.int64).to(self._device)

    def _gen_pos(self, n, length, nodes, **kwargs):
        walks = []

        with tqdm(total=n * nodes.shape[0], position=0, leave=True) as bar:
            for itr in range(n):
                bar.set_description(f'Node walk sequence {itr + 1} of {n}', True)
                self._rs.numpy.shuffle(nodes)
                for node in nodes:
                    walk = [node]
                    while len(walk) < length:
                        cur = walk[-1]
                        prev = walk[-2] if len(walk) > 1 else None
                        nxt = self._next_iter(prev, cur)
                        walk.append(nxt)
                    walk.append(1)  # label indicator positive
                    walks.append(walk)
                    bar.update(1)

        print('Num Pos Samples: ', len(walks))  # TODO remove debug only
        return torch.tensor(walks, dtype=torch.int64).to(self._device)

    def sample(self, batch: Union[List[int], Tensor]) -> Tuple[Tensor, Tensor]:
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)

        print('Batch shape : ', batch.size())
        #neg_rand_sample = torch.randint(0,self._neg.size(0), batch.size())

        #return self._pos[batch, : ], self._neg[neg_rand_sample, : ]
        return torch.full((batch.size(0), self._walks.size(1)), 1, dtype=torch.int32), torch.full(
            (batch.size(0), self._walks.size(1)), 0, dtype=torch.int32)

    def _init_loader(self, **kwargs):
        self._loader = DataLoader(self, collate_fn=self.sample, **kwargs)

    def loader(self, **kwargs):
        if self._idx is not None and self._loader is None:
            self._init_loader(**kwargs)
        return self._loader
