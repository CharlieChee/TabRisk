"""
SynthCity backend for tabular synthetic data generation.

This module encapsulates all SynthCity usage as the unified tabular synthetic
data framework. It provides a single point of integration so that the system
can be described as: "We use SynthCity as our tabular synthetic data
generation backend."

Supported generators: CTGAN, TVAE, PATEGAN.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from loguru import logger

# Supported SynthCity generator names (used in config and load/save)
SYNTHCITY_GENERATORS = ("ctgan", "tvae", "pategan")


def create_plugin(
    name: str,
    random_state: int = 0,
    **kwargs: Any,
) -> Any:
    """
    创建 SynthCity 生成器插件。

    Args:
        name: 生成器名称，支持 "ctgan", "tvae", "pategan"
        random_state: 随机种子，保证可复现
        **kwargs: 传递给 SynthCity 插件的额外参数（如 n_iter, batch_size 等）

    Returns:
        SynthCity 插件实例

    Raises:
        ValueError: 不支持的 name
        ImportError: 未安装 synthcity
    """
    if name not in SYNTHCITY_GENERATORS:
        raise ValueError(
            f"不支持的生成器: {name}，可选: {SYNTHCITY_GENERATORS}"
        )

    try:
        from synthcity.plugins import Plugins
    except ImportError as e:
        raise ImportError(
            "未安装 SynthCity。请运行: pip install synthcity"
        ) from e

    # 统一传入 random_state 以保证可复现
    params: Dict[str, Any] = {"random_state": random_state, **kwargs}
    plugin = Plugins().get(name, **params)
    logger.debug(f"已创建 SynthCity 插件: {name}, random_state={random_state}")
    return plugin
