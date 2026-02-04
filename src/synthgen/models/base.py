"""Base model interface for synthetic data generation."""

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
from synthgen.data.schema import Schema


class BaseModel(ABC):
    """生成模型的基类接口。"""

    @abstractmethod
    def fit(self, data: pd.DataFrame, schema: Schema) -> None:
        """
        训练模型。

        Args:
            data: 训练数据
            schema: 数据 schema
        """
        pass

    @abstractmethod
    def sample(self, num_rows: int) -> pd.DataFrame:
        """
        生成合成数据。

        Args:
            num_rows: 生成的行数

        Returns:
            合成的 DataFrame
        """
        pass

    @abstractmethod
    def save(self, file_path: str) -> None:
        """
        保存模型。

        Args:
            file_path: 保存路径
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, file_path: str) -> "BaseModel":
        """
        加载模型。

        Args:
            file_path: 模型文件路径

        Returns:
            加载的模型实例
        """
        pass
