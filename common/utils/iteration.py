from collections import abc


def is_iterable_collection(x):
    return isinstance(x, abc.Iterable) and not isinstance(x, (str, bytes))
