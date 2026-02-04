"""数据加载与 schema 模块。"""

from synthgen.data.load import load_csv
from synthgen.data.schema import Schema, infer_schema

__all__ = ["load_csv", "Schema", "infer_schema"]
