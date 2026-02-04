"""Tabular synthetic data generation package."""

from synthgen.compat.sklearn import apply_sklearn_compat

# 必须在 synthcity 首次导入前执行，解决 sklearn OneHotEncoder sparse/sparse_output 兼容
apply_sklearn_compat()

__version__ = "0.1.0"
