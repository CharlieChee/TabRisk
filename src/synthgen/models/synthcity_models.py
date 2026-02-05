"""
SynthCity-based model implementations (CTGAN, TVAE, PATEGAN).

These models implement BaseModel and delegate generation to the SynthCity
backend, keeping the same fit/sample interface and schema usage.

注意：SynthCity plugin（TabularGAN 等）含 closure，不可 pickle。
本项目默认不序列化生成器对象，仅保存 schema / 配置 / 元数据。
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch
from loguru import logger

from synthgen.data.schema import Schema
from synthgen.models.base import BaseModel
from synthgen.synthetic_backend import SYNTHCITY_GENERATORS, create_plugin
from synthgen.utils.conversion import to_dataframe


def _get_synthcity_plugin_class(name: str) -> Optional[type]:
    """获取 SynthCity 插件类（用于 inspect 构造参数）。"""
    try:
        from synthcity.plugins import Plugins
        plugins = Plugins()
        if hasattr(plugins, "plugins") and isinstance(getattr(plugins, "plugins"), dict):
            return plugins.plugins.get(name)
        if hasattr(plugins, "_plugins") and isinstance(getattr(plugins, "_plugins"), dict):
            return plugins._plugins.get(name)
    except Exception:
        pass
    return None


def _move_torch_modules_to_cuda(plugin: Any) -> None:
    """在 plugin 内查找 torch.nn.Module（model/_model/net/generator/discriminator 等）并 .to('cuda')。"""
    if plugin is None:
        return
    candidates = ("model", "_model", "net", "_net", "generator", "discriminator", "_generator", "_discriminator")
    seen: set[int] = set()
    for attr_name in candidates:
        try:
            child = getattr(plugin, attr_name, None)
            if child is not None and id(child) not in seen and isinstance(child, torch.nn.Module):
                seen.add(id(child))
                child.to("cuda")
                logger.info("moved %s to cuda", attr_name)
        except Exception:
            pass
    try:
        for attr_name, child in vars(plugin).items():
            if child is not None and id(child) not in seen and isinstance(child, torch.nn.Module):
                seen.add(id(child))
                child.to("cuda")
                logger.info("moved %s to cuda", attr_name)
    except Exception:
        pass


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
        if self.BACKEND_NAME == "ctgan":
            logger.info(
                "CTGAN GPU: torch.cuda.is_available()=%s, CUDA_VISIBLE_DEVICES=%s, plugin.device=%s",
                torch.cuda.is_available(),
                os.environ.get("CUDA_VISIBLE_DEVICES", "not set"),
                getattr(self._plugin, "device", "unknown"),
            )
            try:
                for attr in ("model", "_model", "generator", "discriminator"):
                    m = getattr(self._plugin, attr, None)
                    if m is not None and isinstance(m, torch.nn.Module) and hasattr(m, "parameters"):
                        try:
                            dev = next(m.parameters(), None)
                            logger.info("CTGAN underlying %s device: %s", attr, dev.device if dev is not None else "N/A")
                        except Exception:
                            pass
                        break
            except Exception:
                pass
            _move_torch_modules_to_cuda(self._plugin)
            epochs = getattr(self._plugin, "epochs", self.n_iter)
            logger.info("CTGAN epochs: %s", epochs)
            logger.info("CTGAN training started...")
        self._plugin.fit(data)
        if self.BACKEND_NAME == "ctgan":
            logger.info("CTGAN training finished.")
        logger.info("模型训练完成")

    def sample(self, num_rows: int) -> pd.DataFrame:
        if self._plugin is None:
            raise ValueError("模型尚未训练，请先调用 fit()")
        logger.info(f"生成 {num_rows} 行合成数据 (SynthCity {self.BACKEND_NAME})...")
        out = self._plugin.generate(count=num_rows)
        df = to_dataframe(out)
        # 列名与训练 schema 对齐
        if self.schema is not None:
            expected_cols = [c.name for c in self.schema.columns]
            if set(expected_cols) <= set(df.columns):
                df = df[expected_cols]
            elif len(expected_cols) == len(df.columns):
                df.columns = expected_cols
        logger.info(f"生成完成，形状: {df.shape}")
        return df

    def save(self, file_path: str) -> None:
        """已弃用：SynthCity plugin 不可 pickle。请使用 save_metadata。"""
        raise NotImplementedError(
            "SynthCity plugin 含 closure，不可 pickle。"
            "请使用 save_metadata(output_dir) 保存元数据，或于训练时通过 synthetic_rows 生成数据。"
        )

    def save_metadata(self, output_dir: str) -> None:
        """保存 plugin name + params 元数据（不序列化 plugin 对象）。"""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        config = {
            "backend_name": self.BACKEND_NAME,
            "random_state": self.random_state,
            "n_iter": self.n_iter,
            "batch_size": self.batch_size,
            "verbose": self.verbose,
            **self.extra_kwargs,
        }
        meta_path = path / "model_metadata.yaml"
        meta_path.write_text(
            f"# 仅用于重建参考，不可用于反序列化\nbackend_name: {self.BACKEND_NAME}\nparams:\n"
            + "\n".join(f"  {k}: {v}" for k, v in config.items() if k != "backend_name"),
            encoding="utf-8",
        )
        logger.info(f"模型元数据已保存: {meta_path}")

    @classmethod
    def load(cls, file_path: str) -> _SynthCityModelBase:
        """已弃用：无法从 pickle 加载 SynthCity plugin。"""
        raise NotImplementedError(
            "SynthCity plugin 不可反序列化。本项目不保存/加载 plugin 对象，"
            "研究关注 synthetic data 与 privacy，而非模型复用。"
        )


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

    def _plugin_params(self) -> Dict[str, Any]:
        """根据 CTGAN 构造函数签名自动探测并传入 GPU 参数（device/cuda/use_cuda/gpu）。"""
        params = super()._plugin_params()
        plugin_cls = _get_synthcity_plugin_class("ctgan")
        if plugin_cls is not None:
            try:
                sig = inspect.signature(plugin_cls.__init__)
                allowed = set(sig.parameters.keys())
                if "device" in allowed:
                    params["device"] = "cuda:0"
                elif "cuda" in allowed:
                    params["cuda"] = True
                elif "use_cuda" in allowed:
                    params["use_cuda"] = True
                elif "gpu" in allowed:
                    params["gpu"] = True
            except Exception:
                pass
        return params


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
    """已弃用：SynthCity plugin 不可 pickle 反序列化。"""
    raise NotImplementedError(
        "SynthCity plugin 不可反序列化。本项目不保存/加载 plugin 对象，"
        "请于训练时通过 synthetic_rows 生成合成数据。"
    )
