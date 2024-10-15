import pandas as pd
import numpy as np
from pathlib import Path
from dataset import SemanticNetworkDataset
from sampling import BaseSampler
from random_walk import BiasedRandomWalk

#p = Path(root_dir)
#p2 = p.joinpath('base/sn_nodes.csv')
#print(p2.exists())
#df = pd.read_csv(p2)
dataset = SemanticNetworkDataset(root_dir)   #, force_rebuild=True)
print(dataset.graph.info())
#%%
sampler = BiasedRandomWalk(dataset,5, length=10, p=100.0, q=1.0, neg_samples=True, neg_multiplier=5)()
loader = sampler.loader(batch_size=10, shuffle=True, num_workers=0)