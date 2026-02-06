"""Model interfaces and implementations（仅使用 SynthCity 后端）。"""

from pathlib import Path
from typing import Union

from synthgen.models.base import BaseModel as BaseGeneratorModel
from synthgen.models.synthcity_models import (
    SynthCityCTGANModel,
    SynthCityTVAEModel,
    SynthCityPATEGANModel,
    SynthCityTabDDPMModel,
    SynthCityAIMModel,
    SynthCityPrivBayesModel,
    SynthCityDPGANModel,
    SynthCityARFModel,
    SynthCityMarginalDistributionsModel,
    SynthCityUniformSamplerModel,
    load_synthcity_model,
)
from synthgen.synthetic_backend import SYNTHCITY_GENERATORS


def load_model(file_path: str) -> Union[BaseGeneratorModel]:
    """
    加载 SynthCity 模型。已弃用：SynthCity plugin 不可 pickle 反序列化。
    合成数据请于训练时通过 synthetic_rows 生成。
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"模型文件不存在: {file_path}。本项目默认不保存 model.pkl，"
            "请在训练时通过 synthetic_rows 生成合成数据。"
        )
    raise NotImplementedError(
        "SynthCity plugin 不可 pickle 反序列化。本项目不序列化生成器对象，"
        "研究关注 synthetic data 与 privacy，而非模型复用。"
        "请在训练时通过 synthetic_rows 生成合成数据。"
    )


__all__ = [
    "BaseGeneratorModel",
    "SynthCityCTGANModel",
    "SynthCityTVAEModel",
    "SynthCityPATEGANModel",
    "SynthCityTabDDPMModel",
    "SynthCityAIMModel",
    "SynthCityPrivBayesModel",
    "SynthCityDPGANModel",
    "SynthCityARFModel",
    "SynthCityMarginalDistributionsModel",
    "SynthCityUniformSamplerModel",
    "load_synthcity_model",
    "load_model",
]
