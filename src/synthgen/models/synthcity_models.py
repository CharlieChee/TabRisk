"""
SynthCity-based model implementations (CTGAN, TVAE, PATEGAN).

These models implement BaseModel and delegate generation to the SynthCity
backend, keeping the same fit/sample/save/load interface and schema usage.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from loguru import logger

from synthgen.data.schema import Schema
from synthgen.models.base import BaseModel
from synthgen.synthetic_backend import SYNTHCITY_GENERATORS, create_plugin


def _save_dict(
    plugin: Any,
    schema: Optional[Schema],
    backend_name: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """构建保存用的字典，包含 backend_name 以便 load_model 识别。"""
    return {
        "backend_name": backend_name,
        "plugin": plugin,
        "schema": schema,
        "config": config,
    }


def _load_common(file_path: str) -> Dict[str, Any]:
    """加载 pickle 并返回字典。"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"模型文件不存在: {file_path}")
    with open(path, "rb") as f:
        return pickle.load(f)


class _SynthCityModelBase(BaseModel):
    """SynthCity 模型基类：共用 fit/sample/save/load，子类仅指定 backend_name 与默认参数。"""

    BACKEND_NAME: str = ""

    def __init__(
        self,
        random_state: int = 0,
        n_iter: int = 300,
        batch_size: int = 500,
        verbose: bool = True,
        **kwargs: Any,
    ):
        if not self.BACKEND_NAME:
            raise ValueError("子类必须设置 BACKEND_NAME")
        self.random_state = random_state
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.verbose = verbose
        self.extra_kwargs = kwargs
        self._plugin: Optional[Any] = None
        self.schema: Optional[Schema] = None

    def _plugin_params(self) -> Dict[str, Any]:
        # 所有参数经 create_plugin -> compat.synthcity.filter_plugin_params 显式过滤后传入
        # 不在此直接传 verbose 等已知不兼容参数；extra_kwargs 也会被过滤
        return {
            "n_iter": self.n_iter,
            "batch_size": self.batch_size,
            **self.extra_kwargs,
        }

    def fit(self, data: pd.DataFrame, schema: Schema) -> None:
        self.schema = schema
        self._plugin = create_plugin(
            self.BACKEND_NAME,
            random_state=self.random_state,
            **self._plugin_params(),
        )
        logger.info(f"开始训练 SynthCity {self.BACKEND_NAME} 模型...")
        logger.info(f"训练数据形状: {data.shape}, n_iter={self.n_iter}, batch_size={self.batch_size}")
        self._plugin.fit(data)
        logger.info("模型训练完成")

    def sample(self, num_rows: int) -> pd.DataFrame:
        if self._plugin is None:
            raise ValueError("模型尚未训练，请先调用 fit()")
        logger.info(f"生成 {num_rows} 行合成数据 (SynthCity {self.BACKEND_NAME})...")
        out = self._plugin.generate(count=num_rows)
        # SynthCity 可能返回 numpy 或 DataFrame，统一为 DataFrame
        if not isinstance(out, pd.DataFrame):
            out = pd.DataFrame(out)
        logger.info(f"生成完成，形状: {out.shape}")
        return out

    def save(self, file_path: str) -> None:
        if self._plugin is None:
            raise ValueError("模型尚未训练，无法保存")
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            "random_state": self.random_state,
            "n_iter": self.n_iter,
            "batch_size": self.batch_size,
            "verbose": self.verbose,
            **self.extra_kwargs,
        }
        with open(path, "wb") as f:
            pickle.dump(
                _save_dict(self._plugin, self.schema, self.BACKEND_NAME, config),
                f,
            )
        logger.info(f"模型已保存到: {file_path}")

    @classmethod
    def load(cls, file_path: str) -> _SynthCityModelBase:
        d = _load_common(file_path)
        backend = d.get("backend_name")
        if backend != cls.BACKEND_NAME:
            raise ValueError(
                f"模型文件为 {backend}，当前类为 {cls.BACKEND_NAME}"
            )
        config = d.get("config", {})
        instance = cls(**config)
        instance._plugin = d["plugin"]
        instance.schema = d.get("schema")
        logger.info(f"已加载 SynthCity {cls.BACKEND_NAME} 模型: {file_path}")
        return instance


class SynthCityCTGANModel(_SynthCityModelBase):
    """基于 SynthCity 的 CTGAN 生成模型。"""

    BACKEND_NAME = "ctgan"

    def __init__(
        self,
        random_state: int = 0,
        n_iter: int = 300,
        batch_size: int = 500,
        verbose: bool = True,
        generator_n_layers_hidden: int = 2,
        generator_n_units_hidden: int = 256,
        discriminator_n_layers_hidden: int = 2,
        discriminator_n_units_hidden: int = 256,
        **kwargs: Any,
    ):
        super().__init__(
            random_state=random_state,
            n_iter=n_iter,
            batch_size=batch_size,
            verbose=verbose,
            generator_n_layers_hidden=generator_n_layers_hidden,
            generator_n_units_hidden=generator_n_units_hidden,
            discriminator_n_layers_hidden=discriminator_n_layers_hidden,
            discriminator_n_units_hidden=discriminator_n_units_hidden,
            **kwargs,
        )


class SynthCityTVAEModel(_SynthCityModelBase):
    """基于 SynthCity 的 TVAE 生成模型。"""

    BACKEND_NAME = "tvae"

    def __init__(
        self,
        random_state: int = 0,
        n_iter: int = 300,
        batch_size: int = 500,
        verbose: bool = True,
        n_layers_hidden: int = 2,
        n_units_hidden: int = 256,
        **kwargs: Any,
    ):
        super().__init__(
            random_state=random_state,
            n_iter=n_iter,
            batch_size=batch_size,
            verbose=verbose,
            n_layers_hidden=n_layers_hidden,
            n_units_hidden=n_units_hidden,
            **kwargs,
        )


class SynthCityPATEGANModel(_SynthCityModelBase):
    """基于 SynthCity 的 PATEGAN 生成模型（隐私友好）。"""

    BACKEND_NAME = "pategan"

    def __init__(
        self,
        random_state: int = 0,
        n_iter: int = 300,
        batch_size: int = 500,
        verbose: bool = True,
        generator_n_layers_hidden: int = 2,
        generator_n_units_hidden: int = 256,
        discriminator_n_layers_hidden: int = 2,
        discriminator_n_units_hidden: int = 256,
        **kwargs: Any,
    ):
        super().__init__(
            random_state=random_state,
            n_iter=n_iter,
            batch_size=batch_size,
            verbose=verbose,
            generator_n_layers_hidden=generator_n_layers_hidden,
            generator_n_units_hidden=generator_n_units_hidden,
            discriminator_n_layers_hidden=discriminator_n_layers_hidden,
            discriminator_n_units_hidden=discriminator_n_units_hidden,
            **kwargs,
        )


def load_synthcity_model(file_path: str) -> BaseModel:
    """
    根据保存文件中的 backend_name 加载对应的 SynthCity 模型。

    Args:
        file_path: 模型文件路径（.pkl）

    Returns:
        对应的 SynthCity 模型实例（SynthCityCTGANModel / TVAE / PATEGAN）
    """
    d = _load_common(file_path)
    backend_name = d.get("backend_name")
    if backend_name not in SYNTHCITY_GENERATORS:
        raise ValueError(
            f"无法识别的 SynthCity 模型: {backend_name}，"
            f"支持: {SYNTHCITY_GENERATORS}"
        )
    if backend_name == "ctgan":
        return SynthCityCTGANModel.load(file_path)
    if backend_name == "tvae":
        return SynthCityTVAEModel.load(file_path)
    if backend_name == "pategan":
        return SynthCityPATEGANModel.load(file_path)
    raise ValueError(f"未实现的 backend: {backend_name}")
