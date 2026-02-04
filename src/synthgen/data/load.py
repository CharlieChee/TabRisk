"""数据加载。"""

import pandas as pd


def load_csv(file_path: str, **kwargs) -> pd.DataFrame:
    """
    从 CSV 文件加载数据。

    Args:
        file_path: CSV 文件路径
        **kwargs: 传递给 pandas.read_csv 的额外参数（如 encoding）

    Returns:
        加载的 DataFrame
    """
    default_kwargs = {"encoding": "utf-8"}
    default_kwargs.update(kwargs)
    return pd.read_csv(file_path, **default_kwargs)
