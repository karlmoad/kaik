import re
from .feature import Feature

SPARSE_PATTERN = re.compile(r"<\([0-9]*,-?[0-9.]*\):(?:\{-?[0-9.]*:\[(?:[0-9]*,?)*]},?)*>")
DENSE_PATTERN = re.compile("<\[(?:-?[0-9.]*,?)*]>")
SPARSE_HEADER = re.compile(r"\((?P<dim>[0-9.]*),(?P<def>-?[0-9.]*)\){1}")
SPARSE_DEFINITION = re.compile(r"(?P<definition>\{-?[0-9.]*:\[(?:[0-9]*,?)*]})")
VALUE = re.compile(r"\{(?P<value>-?[0-9.]*):")
ARRAY = re.compile(r"(?P<array>\[(-?[0-9.]*,?)*\])")
WHITE_SPACE = re.compile(r"\s")

