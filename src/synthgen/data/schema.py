"""数据 Schema 推断与序列化。"""

import json
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, asdict
import pandas as pd


@dataclass
class ColumnSchema:
    """单列 schema。"""

    name: str
    is_categorical: bool
    unique_count: Optional[int] = None
    null_count: int = 0


class Schema:
    """数据表 schema：列类型、统计等。"""

    def __init__(
        self,
        columns: List[ColumnSchema],
        total_rows: int = 0,
        total_columns: int = 0,
    ):
        self.columns = columns
        self._total_rows = total_rows
        self._total_columns = total_columns or len(columns)
        self._categorical_columns: List[str] = [c.name for c in columns if c.is_categorical]
        self._continuous_columns: List[str] = [c.name for c in columns if not c.is_categorical]

    @property
    def total_rows(self) -> int:
        return self._total_rows

    @property
    def total_columns(self) -> int:
        return self._total_columns

    @property
    def categorical_columns(self) -> List[str]:
        return self._categorical_columns

    @property
    def continuous_columns(self) -> List[str]:
        return self._continuous_columns

    def save(self, file_path: str) -> None:
        """保存 schema 到 JSON。"""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "total_rows": self._total_rows,
            "total_columns": self._total_columns,
            "categorical_columns": self._categorical_columns,
            "continuous_columns": self._continuous_columns,
            "columns": [asdict(c) for c in self.columns],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, file_path: str) -> "Schema":
        """从 JSON 加载 schema。"""
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        columns = [ColumnSchema(**c) for c in data["columns"]]
        return cls(
            columns=columns,
            total_rows=data.get("total_rows", 0),
            total_columns=data.get("total_columns", len(columns)),
        )


def infer_schema(df: pd.DataFrame, categorical_threshold: Optional[int] = None) -> Schema:
    """
    从 DataFrame 推断 schema（列类型、唯一值数、缺失值数）。

    Args:
        df: 输入数据
        categorical_threshold: 唯一值数量小于等于此值的数值列视为分类；默认 min(50, 行数*0.5)

    Returns:
        Schema 实例
    """
    n = len(df)
    if categorical_threshold is None:
        categorical_threshold = min(50, max(1, int(n * 0.5)))

    columns: List[ColumnSchema] = []
    for col in df.columns:
        s = df[col]
        unique_count = int(s.nunique())
        null_count = int(s.isna().sum())
        dtype = s.dtype

        # 对象/字符串/类别 或 唯一值较少 -> 分类
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            is_categorical = True
        elif pd.api.types.is_categorical_dtype(dtype):
            is_categorical = True
        elif pd.api.types.is_bool_dtype(dtype):
            is_categorical = True
        elif pd.api.types.is_numeric_dtype(dtype):
            is_categorical = unique_count <= categorical_threshold
        else:
            is_categorical = unique_count <= categorical_threshold

        columns.append(
            ColumnSchema(
                name=col,
                is_categorical=is_categorical,
                unique_count=unique_count,
                null_count=null_count,
            )
        )

    return Schema(columns=columns, total_rows=n, total_columns=len(columns))
