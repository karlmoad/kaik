from collections import abc
import copy
import pandas as pd
import numpy as np


def row_get_or_default(row: pd.Series, fields: list, default: object):
    out = np.full(len(fields), default, dtype=object)
    for i, field in enumerate(fields):
        out[i] = row[field] if field in row and row[field] is not None else default
    return out


def default_if_none(value, default):
    return value if value is not None else default


def map_dataframe(data: pd.DataFrame, mapping: dict, defaults: dict, /, apply_func=None) -> pd.DataFrame:
    mapped = []
    apply = (lambda x: x) if apply_func is None else apply_func

    for _, row in data.iterrows():
        buffer = copy.deepcopy(defaults)
        for key, field in mapping.items():
            if isinstance(field, list) and field:
                buffer[key] = {}
                for f in field:
                    if f in data:
                        buffer[key][f] = apply(row[f])
            elif isinstance(field, dict) and field:
                buffer[key] = {}
                for k, v in field.items():
                    if v in data:
                        buffer[key][k] = apply(row[v])
            else:
                if field in data:
                    buffer[key] = apply(row[field] if row[field] is not None else buffer[key])
        mapped.append(buffer)

    return pd.DataFrame(mapped)
