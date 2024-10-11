import numpy as np
import pandas as pd


def _assert_state(data, /, nodes=False, edges=False):
    if edges:
        assert 'edges'  in data, "edge data not found"
        assert isinstance(data['edges'], pd.DataFrame),"edges must be transformed to dataset prior to this step"

    if nodes:
        assert 'nodes'  in data, "node data not found"
        assert isinstance(data['nodes'], pd.DataFrame),"nodes must be transformed to dataset prior to this step"
