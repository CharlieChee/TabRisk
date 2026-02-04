"""
通用对象到 pandas.DataFrame 的转换适配。

SynthCity / fastai 等可能返回非标准类型，需统一转为 DataFrame。
"""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass


def to_dataframe(obj: Any) -> pd.DataFrame:
    """
    将多种类型强健地转换为 pandas.DataFrame。

    处理顺序：
    1. 已是 DataFrame → 直接返回
    2. .dataframe() 方法
    3. .to_pandas() 方法
    4. dict / list → pd.DataFrame
    5. numpy ndarray → pd.DataFrame
    6. torch.Tensor → numpy → pd.DataFrame
    7. 兜底：np.asarray → pd.DataFrame

    Args:
        obj: 待转换对象

    Returns:
        pandas.DataFrame

    Raises:
        ValueError: 无法转换时，附带 obj 类型与 repr 信息
    """
    if isinstance(obj, pd.DataFrame):
        return obj

    if hasattr(obj, "dataframe") and callable(getattr(obj, "dataframe")):
        try:
            return obj.dataframe()
        except Exception as e:
            raise ValueError(
                f"obj.dataframe() 失败 (type={type(obj).__name__}, repr={repr(obj)[:200]}): {e}"
            ) from e

    if hasattr(obj, "to_pandas") and callable(getattr(obj, "to_pandas")):
        try:
            return obj.to_pandas()
        except Exception as e:
            raise ValueError(
                f"obj.to_pandas() 失败 (type={type(obj).__name__}, repr={repr(obj)[:200]}): {e}"
            ) from e

    if isinstance(obj, dict):
        try:
            return pd.DataFrame(obj)
        except Exception as e:
            raise ValueError(
                f"pd.DataFrame(dict) 失败 (type={type(obj).__name__}, keys={list(obj.keys())[:10]}): {e}"
            ) from e

    if isinstance(obj, (list, tuple)):
        try:
            return pd.DataFrame(obj)
        except Exception as e:
            raise ValueError(
                f"pd.DataFrame(list) 失败 (type={type(obj).__name__}, len={len(obj)}): {e}"
            ) from e

    if isinstance(obj, np.ndarray):
        try:
            return pd.DataFrame(obj)
        except Exception as e:
            raise ValueError(
                f"pd.DataFrame(ndarray) 失败 (shape={obj.shape}, dtype={obj.dtype}): {e}"
            ) from e

    try:
        import torch
        if isinstance(obj, torch.Tensor):
            arr = obj.detach().cpu().numpy()
            return pd.DataFrame(arr)
    except ImportError:
        pass

    try:
        arr = np.asarray(obj)
        return pd.DataFrame(arr)
    except Exception as e:
        raise ValueError(
            f"无法转换为 DataFrame (type={type(obj).__name__}, repr={repr(obj)[:200]}): {e}"
        ) from e
