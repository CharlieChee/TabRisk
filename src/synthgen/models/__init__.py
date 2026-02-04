"""Model interfaces and implementations."""

import pickle
from pathlib import Path
from typing import Union

from synthgen.models.base import BaseModel as BaseGeneratorModel
from synthgen.models.sdv_ctgan import SDVCTGANModel
from synthgen.models.synthcity_models import (
    SynthCityCTGANModel,
    SynthCityTVAEModel,
    SynthCityPATEGANModel,
    load_synthcity_model,
)
from synthgen.synthetic_backend import SYNTHCITY_GENERATORS


def load_model(file_path: str) -> Union[BaseGeneratorModel]:
    """
    根据保存文件内容自动选择加载方式：SynthCity (ctgan/tvae/pategan) 或旧版 SDV CTGAN。
    保证随机种子与数据划分方式与原实验一致（由训练时保存的配置决定）。
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"模型文件不存在: {file_path}")
    with open(path, "rb") as f:
        d = pickle.load(f)
    backend_name = d.get("backend_name")
    if backend_name in SYNTHCITY_GENERATORS:
        return load_synthcity_model(str(path))
    # 兼容旧版 SDV CTGAN 模型
    return SDVCTGANModel.load(str(path))


__all__ = [
    "BaseGeneratorModel",
    "SDVCTGANModel",
    "SynthCityCTGANModel",
    "SynthCityTVAEModel",
    "SynthCityPATEGANModel",
    "load_synthcity_model",
    "load_model",
]
