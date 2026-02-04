"""Model interfaces and implementations（仅使用 SynthCity 后端）。"""

import pickle
from pathlib import Path
from typing import Union

from synthgen.models.base import BaseModel as BaseGeneratorModel
from synthgen.models.synthcity_models import (
    SynthCityCTGANModel,
    SynthCityTVAEModel,
    SynthCityPATEGANModel,
    load_synthcity_model,
)
from synthgen.synthetic_backend import SYNTHCITY_GENERATORS


def load_model(file_path: str) -> Union[BaseGeneratorModel]:
    """
    根据保存文件中的 backend_name 加载 SynthCity 模型（ctgan / tvae / pategan）。
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"模型文件不存在: {file_path}")
    with open(path, "rb") as f:
        d = pickle.load(f)
    backend_name = d.get("backend_name")
    if backend_name in SYNTHCITY_GENERATORS:
        return load_synthcity_model(str(path))
    raise ValueError(
        f"不支持的模型类型: {backend_name}。本项目仅使用 SynthCity 后端（ctgan / tvae / pategan）。"
    )


__all__ = [
    "BaseGeneratorModel",
    "SynthCityCTGANModel",
    "SynthCityTVAEModel",
    "SynthCityPATEGANModel",
    "load_synthcity_model",
    "load_model",
]
