"""数据加载：CSV 与标准数据集（OpenML / HF / SDV demo / sklearn）统一入口。"""

from pathlib import Path
from typing import Any, Optional

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


def _ensure_cache_dir(cache_dir: Optional[str]) -> Path:
    """解析并创建缓存目录。"""
    if not cache_dir:
        cache_dir = "data/cache"
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_dataframe(df: pd.DataFrame, dropna: bool) -> pd.DataFrame:
    """
    统一后处理：分类列转 category/string，可选 dropna。
    schema 推断模块能正确识别 object/string/category 为分类。
    """
    df = df.copy()
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            df[col] = s.astype("string")
        elif pd.api.types.is_categorical_dtype(s):
            continue
        elif pd.api.types.is_bool_dtype(s):
            df[col] = s.astype("string")
    if dropna:
        df = df.dropna()
    return df.reset_index(drop=True)


def _load_openml(
    name: str,
    openml_id: Optional[int] = None,
    split: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """从 OpenML 加载数据集，使用 data_home 缓存，重复运行不重复下载。"""
    try:
        from sklearn.datasets import fetch_openml
    except ImportError as e:
        raise ImportError("OpenML 数据源需要安装 scikit-learn: pip install scikit-learn") from e

    cache_path = _ensure_cache_dir(cache_dir)
    data_home = str(cache_path / "openml")

    if openml_id is not None:
        data = fetch_openml(data_id=openml_id, as_frame=True, data_home=data_home, **kwargs)
    else:
        data = fetch_openml(name=name, as_frame=True, data_home=data_home, **kwargs)

    # fetch_openml 返回 Bunch，.frame 为单表（含 target 列）或 (data, target)
    if hasattr(data, "frame") and data.frame is not None:
        df = data.frame
    elif hasattr(data, "data") and hasattr(data, "target"):
        df = data.data.copy()
        if data.target is not None:
            target_name = getattr(data, "target_name", "target")
            if hasattr(data.target, "name") and data.target.name:
                target_name = data.target.name
            df[target_name] = data.target
    else:
        raise ValueError("fetch_openml 返回格式无法解析为 DataFrame")

    return df


def _load_hf(
    name: str,
    split: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """从 HuggingFace Datasets 加载，转成 pandas，缓存到 cache_dir。"""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("HF 数据源需要安装 datasets: pip install datasets") from e

    cache_path = _ensure_cache_dir(cache_dir)
    hf_cache = str(cache_path / "hf")

    ds = load_dataset(name, cache_dir=hf_cache, **kwargs)
    if split:
        if split not in ds:
            raise ValueError(f"split '{split}' 不存在，可用: {list(ds.keys())}")
        ds = ds[split]
    else:
        # 取第一个 split 作为全量
        first_key = next(iter(ds.keys()))
        ds = ds[first_key]

    # Dataset 转 pandas
    return ds.to_pandas()


def _load_sdv_demo(
    name: str,
    cache_dir: Optional[str] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """从 SDV demo 下载并加载（需安装 sdv）。"""
    try:
        from sdv.datasets.demo import get_available_demos, load_demo
    except ImportError as e:
        raise ImportError("SDV demo 数据源需要安装 sdv: pip install sdv") from e

    _ensure_cache_dir(cache_dir)
    # load_demo 会下载到默认或指定路径，返回 metadata + tables
    out = load_demo(dataset=name, **kwargs)
    if isinstance(out, dict) and "tables" in out:
        tables = out["tables"]
        # 单表任务取第一张表
        first_table = next(iter(tables.values()))
        if hasattr(first_table, "to_pandas"):
            return first_table.to_pandas()
        return pd.DataFrame(first_table)
    if isinstance(out, pd.DataFrame):
        return out
    raise ValueError("SDV load_demo 返回格式无法解析为单表 DataFrame")


def _load_sklearn(
    name: str,
    **kwargs: Any,
) -> pd.DataFrame:
    """sklearn 内置数据集（如 iris、breast_cancer），无需网络。"""
    try:
        from sklearn.datasets import load_breast_cancer, load_iris
    except ImportError as e:
        raise ImportError("sklearn 数据源需要安装 scikit-learn: pip install scikit-learn") from e

    loaders = {
        "iris": load_iris,
        "breast_cancer": load_breast_cancer,
    }
    if name not in loaders:
        raise ValueError(f"不支持的 sklearn 数据集: {name}，可选: {list(loaders)}")

    data = loaders[name](return_X_y=False, as_frame=True)
    if hasattr(data, "frame") and data.frame is not None:
        df = data.frame
    else:
        df = pd.DataFrame(data.data, columns=data.feature_names)
        if data.target is not None:
            target_name = getattr(data, "target_names", ["target"])[0] if hasattr(data, "target_names") else "target"
            df[target_name] = data.target
    return df


def load_dataset(
    source: str,
    name: str,
    openml_id: Optional[int] = None,
    split: Optional[str] = None,
    target: Optional[str] = None,
    cache_dir: Optional[str] = None,
    dropna: bool = True,
    project_root: Optional[str] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    统一数据入口：根据 source 从 OpenML / HF / SDV demo / sklearn 加载，返回单表 DataFrame。

    - source=openml: sklearn.datasets.fetch_openml，缓存到 cache_dir/openml
    - source=hf: datasets.load_dataset，缓存到 cache_dir/hf
    - source=sdv_demo: sdv.datasets.demo.load_demo
    - source=sklearn: load_iris / load_breast_cancer 等（无需网络）

    自动将分类列转为 string/category，支持 dropna；返回的 df 可直接用于 schema 推断与训练。
    """
    if project_root and cache_dir and not Path(cache_dir).is_absolute():
        cache_dir = str(Path(project_root) / cache_dir)

    if source == "openml":
        df = _load_openml(name=name, openml_id=openml_id, split=split, cache_dir=cache_dir, **kwargs)
    elif source == "hf":
        df = _load_hf(name=name, split=split, cache_dir=cache_dir, **kwargs)
    elif source == "sdv_demo":
        df = _load_sdv_demo(name=name, cache_dir=cache_dir, **kwargs)
    elif source == "sklearn":
        df = _load_sklearn(name=name, **kwargs)
    else:
        raise ValueError(f"不支持的 source: {source}，可选: openml | hf | sdv_demo | sklearn")

    return _normalize_dataframe(df, dropna=dropna)
