"""
SynthCity-based model implementations (CTGAN, TVAE, PATEGAN).

These models implement BaseModel and delegate generation to the SynthCity
backend, keeping the same fit/sample interface and schema usage.

注意：SynthCity plugin（TabularGAN 等）含 closure，不可 pickle。
本项目默认不序列化生成器对象，仅保存 schema / 配置 / 元数据。
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from loguru import logger

from synthgen.data.schema import Schema
from synthgen.models.base import BaseModel
from synthgen.synthetic_backend import SYNTHCITY_GENERATORS, create_plugin
from synthgen.utils.conversion import to_dataframe


def _first_param_device(m: Any) -> Optional[str]:
    """返回 torch.nn.Module 第一个参数的 device 字符串。"""
    if not isinstance(m, torch.nn.Module):
        return None
    for p in m.parameters():
        return str(p.device)
    return "no-params"


def _find_torch_modules(obj: Any, prefix: str = "plugin", max_depth: int = 5, visited: Optional[set[int]] = None) -> List[Tuple[str, Any]]:
    """递归扫描 obj 找到所有 torch.nn.Module，返回 (path, module) 列表。"""
    if visited is None:
        visited = set()
    oid = id(obj)
    if oid in visited:
        return []
    visited.add(oid)
    found: List[Tuple[str, Any]] = []
    if isinstance(obj, torch.nn.Module):
        found.append((prefix, obj))
    if max_depth <= 0:
        return found
    if isinstance(obj, dict):
        for k, v in list(obj.items())[:200]:
            found.extend(_find_torch_modules(v, f"{prefix}[{repr(k)}]", max_depth - 1, visited))
        return found
    if isinstance(obj, (list, tuple, set)):
        for i, v in enumerate(list(obj)[:200]):
            found.extend(_find_torch_modules(v, f"{prefix}[{i}]", max_depth - 1, visited))
        return found
    for name in dir(obj):
        if name.startswith("__"):
            continue
        if name in ("__dict__", "__class__", "__weakref__"):
            continue
        try:
            val = getattr(obj, name)
        except Exception:
            continue
        if isinstance(val, (int, float, str, bytes, bool, type(None))):
            continue
        found.extend(_find_torch_modules(val, f"{prefix}.{name}", max_depth - 1, visited))
    return found


def _log_device_scan_later(plugin: Any, delay: int = 10) -> None:
    """后台线程：等待 delay 秒后递归扫描 plugin 内 torch.nn.Module 并打印 first_param_device。"""
    time.sleep(delay)
    try:
        mods = _find_torch_modules(plugin, "plugin", max_depth=5)
        logger.info("[deep-scan] found %s torch.nn.Module(s) after %ss", len(mods), delay)
        for path, m in mods[:30]:
            logger.info("[deep-scan] %s first_param_device=%s", path, _first_param_device(m))
    except Exception as e:
        logger.warning("[deep-scan] failed: %r", e)


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

        # 从 kwargs 中提前拿出仅用于环境控制的字段（不传给 SynthCity 插件）
        cuda_visible_devices = kwargs.pop("cuda_visible_devices", None)
        # 兼容旧配置中的 gpu_id，但只用于环境变量，不再透传到插件
        legacy_gpu_id = kwargs.pop("gpu_id", None)
        if cuda_visible_devices is not None:
            self.cuda_visible_devices: Optional[str] = str(cuda_visible_devices)
        elif legacy_gpu_id is not None:
            self.cuda_visible_devices = str(legacy_gpu_id)
        else:
            self.cuda_visible_devices = None

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
        params: Dict[str, Any] = {
            "n_iter": self.n_iter,
            "batch_size": self.batch_size,
            **self.extra_kwargs,
        }
        # 再次兜底：确保不会把环境相关字段透传给 SynthCity 插件
        params.pop("gpu_id", None)
        params.pop("cuda_visible_devices", None)
        return params

    def fit(self, data: pd.DataFrame, schema: Schema) -> None:
        self.schema = schema
        plugin_kwargs = self._plugin_params()

        # 在创建 SynthCity 插件前，根据 cuda_visible_devices 绑定 CUDA_VISIBLE_DEVICES
        cuda_vis = getattr(self, "cuda_visible_devices", None)
        if cuda_vis is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_vis)

        logger.info(f"开始训练 SynthCity {self.BACKEND_NAME} 模型...")
        logger.info(f"训练数据形状: {data.shape}, n_iter={self.n_iter}, batch_size={self.batch_size}")
        if self.BACKEND_NAME == "ctgan":
            logger.info(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
            logger.info(
                f"torch={torch.__version__}, "
                f"cuda_available={torch.cuda.is_available()}, "
                f"device_count={torch.cuda.device_count()}",
            )
            if torch.cuda.is_available():
                try:
                    logger.info(f"cuda_name0={torch.cuda.get_device_name(0)}")
                except Exception:
                    pass
            logger.info(f"ctgan params to synthcity: {plugin_kwargs}")
            logger.info(f"Creating SynthCity ctgan plugin with kwargs: {plugin_kwargs}")
        self._plugin = create_plugin(
            self.BACKEND_NAME,
            random_state=self.random_state,
            **plugin_kwargs,
        )
        if self.BACKEND_NAME == "ctgan":
            logger.info("CTGAN plugin created; starting fit() ...")
            threading.Thread(target=_log_device_scan_later, args=(self._plugin, 10), daemon=True).start()
        self._plugin.fit(data)
        if self.BACKEND_NAME == "ctgan":
            logger.info("CTGAN fit() finished.")
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
        """从配置读 device；当 device=cuda 且 torch.cuda.is_available() 时传 device='cuda' 给插件。"""
        params = super()._plugin_params()
        device = self.extra_kwargs.get("device", "cpu")
        if device == "cuda" and torch.cuda.is_available():
            params["device"] = "cuda"
        else:
            params["device"] = "cpu"
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
