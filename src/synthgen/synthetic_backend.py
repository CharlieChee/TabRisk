"""
SynthCity backend for tabular synthetic data generation.

This module encapsulates all SynthCity usage as the unified tabular synthetic
data framework. It provides a single point of integration so that the system
can be described as: "We use SynthCity as our tabular synthetic data
generation backend."

Supported generators: CTGAN, TVAE, PATEGAN.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional

from loguru import logger

# Supported SynthCity generator names (used in config and load/save)
SYNTHCITY_GENERATORS = ("ctgan", "tvae", "pategan")

# 已知不兼容 SynthCity Plugin.__init__ 的参数（可随版本扩展）
_KNOWN_UNSUPPORTED = frozenset({"verbose"})


def _filter_plugin_params(name: str, params: Dict[str, Any], plugins: Any) -> Dict[str, Any]:
    """
    过滤 params，只保留插件构造函数支持的参数。
    至少移除 _KNOWN_UNSUPPORTED 中的键；
    若可获取插件类，则用 inspect.signature 进一步过滤。
    """
    filtered = {k: v for k, v in params.items() if k not in _KNOWN_UNSUPPORTED}
    try:
        plugin_cls = None
        if hasattr(plugins, "plugins") and isinstance(getattr(plugins, "plugins"), dict):
            plugin_cls = plugins.plugins.get(name)
        elif hasattr(plugins, "_plugins") and isinstance(getattr(plugins, "_plugins"), dict):
            plugin_cls = plugins._plugins.get(name)
        if plugin_cls is not None and isinstance(plugin_cls, type):
            sig = inspect.signature(plugin_cls.__init__)
            allowed = {
                p for p in sig.parameters
                if p not in ("self", "args", "kwargs")
                and sig.parameters[p].kind
                not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
            }
            filtered = {k: v for k, v in filtered.items() if k in allowed}
    except Exception:
        pass
    return filtered


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
            "未安装 SynthCity。请运行: pip install -e . 或 pip install -r requirements.txt"
        ) from e
    except AttributeError as e:
        if "RMSNorm" in str(e) or "rms_norm" in str(e).lower():
            raise ImportError(
                "opacus 与当前 torch 版本不兼容（nn.RMSNorm）。请运行: python scripts/check_env.py\n"
                "并按照 README 的 Dependency Compatibility Notes 安装: torch>=2.0,<2.1 与 opacus>=1.3,<1.5"
            ) from e
        raise

    # 统一传入 random_state 以保证可复现
    params: Dict[str, Any] = {"random_state": random_state, **kwargs}
    plugins = Plugins()
    params = _filter_plugin_params(name, params, plugins)
    plugin = plugins.get(name, **params)
    logger.debug(f"已创建 SynthCity 插件: {name}, random_state={random_state}")
    return plugin
