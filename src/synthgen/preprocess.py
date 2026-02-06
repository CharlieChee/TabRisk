"""可插拔的预处理模块：训练前 transform，采样后 inverse_transform。"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class PreprocessorBase(ABC):
    """预处理器基类。"""

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "PreprocessorBase":
        """基于数据拟合参数。"""
        ...

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """对数据进行变换。"""
        ...

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """拟合并变换。"""
        return self.fit(df).transform(df)

    @abstractmethod
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """将变换后的数据还原到原始尺度。"""
        ...

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """返回可序列化的状态，便于复现。"""
        ...


class IdentityPreprocessor(PreprocessorBase):
    """恒等预处理器：不进行任何变换。"""

    def fit(self, df: pd.DataFrame) -> "IdentityPreprocessor":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def state_dict(self) -> Dict[str, Any]:
        return {"name": "identity", "params": {}}


class MonotonicPreprocessor(PreprocessorBase):
    """单调变换预处理器：支持 log1p 与 zero_hurdle。"""

    def __init__(
        self,
        log1p_cols: Optional[List[str]] = None,
        zero_hurdle_cols: Optional[List[str]] = None,
        indicator_suffix: str = "__is_zero",
    ):
        self.log1p_cols = list(log1p_cols or [])
        self.zero_hurdle_cols = list(zero_hurdle_cols or [])
        self.indicator_suffix = indicator_suffix

        # fit 时记录的列（实际存在于 df 中的）
        self._log1p_applied: List[str] = []
        self._zero_hurdle_applied: List[str] = []
        self._indicator_columns: List[str] = []  # 新增的 __is_zero 列名

    def fit(self, df: pd.DataFrame) -> "MonotonicPreprocessor":
        self._log1p_applied = []
        self._zero_hurdle_applied = []
        self._indicator_columns = []

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in self.zero_hurdle_cols:
            if col in df.columns and col in numeric_cols:
                self._zero_hurdle_applied.append(col)
                ind_name = f"{col}{self.indicator_suffix}"
                self._indicator_columns.append(ind_name)

        for col in self.log1p_cols:
            if col in df.columns and col in numeric_cols and col not in self._zero_hurdle_applied:
                # 校验非负
                if (df[col] < 0).any():
                    raise ValueError(f"列 {col} 包含负值，无法应用 log1p（仅支持非负数值列）")
                self._log1p_applied.append(col)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # zero_hurdle：先处理，会新增 __is_zero 列
        for col in self._zero_hurdle_applied:
            ind_name = f"{col}{self.indicator_suffix}"
            x = out[col].values
            is_zero = (x == 0).astype(np.float64)
            out[ind_name] = is_zero
            # 非零部分 log1p，0 保持 0
            out[col] = np.where(x == 0, 0.0, np.log1p(x))

        # log1p
        for col in self._log1p_applied:
            out[col] = np.log1p(np.clip(out[col].values, 0, None))

        return out

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # zero_hurdle 逆变换：先还原
        for col in self._zero_hurdle_applied:
            ind_name = f"{col}{self.indicator_suffix}"
            if ind_name not in out.columns:
                raise ValueError(f"逆变换需要列 {ind_name}，但数据中不存在")
            x_trans = out[col].values
            is_zero = out[ind_name].values
            x_orig = np.where(is_zero >= 0.5, 0.0, np.expm1(x_trans))
            out[col] = x_orig
            out = out.drop(columns=[ind_name])

        # log1p 逆变换
        for col in self._log1p_applied:
            if col in out.columns:
                out[col] = np.expm1(out[col].values)

        return out

    def state_dict(self) -> Dict[str, Any]:
        return {
            "name": "monotonic",
            "params": {
                "log1p_cols": self._log1p_applied,
                "zero_hurdle_cols": self._zero_hurdle_applied,
                "indicator_suffix": self.indicator_suffix,
            },
        }


_PREPROCESSOR_REGISTRY: Dict[str, type] = {
    "identity": IdentityPreprocessor,
    "none": IdentityPreprocessor,
    "monotonic": MonotonicPreprocessor,
}


def get_preprocessor(name: str, params: Optional[Dict[str, Any]] = None) -> PreprocessorBase:
    """根据名称和参数创建预处理器。"""
    name = str(name).lower()
    if name not in _PREPROCESSOR_REGISTRY:
        raise ValueError(f"未知的预处理器: {name}，支持: {list(_PREPROCESSOR_REGISTRY.keys())}")
    cls = _PREPROCESSOR_REGISTRY[name]
    kwargs = dict(params or {})
    return cls(**kwargs)
