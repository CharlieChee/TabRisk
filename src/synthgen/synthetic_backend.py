"""
SynthCity backend for tabular synthetic data generation.

This module encapsulates all SynthCity usage as the unified tabular synthetic
data framework. It provides a single point of integration so that the system
can be described as: "We use SynthCity as our tabular synthetic data
generation backend."

Supported generators: CTGAN, TVAE, PATEGAN.

所有传入 SynthCity 的第三方参数必须经 compat.synthcity 显式过滤，禁止直接假设 API 稳定。
"""

from __future__ import annotations

from typing import Any, Dict

from loguru import logger

from synthgen.compat.synthcity import filter_plugin_params

# Supported SynthCity generator names (used in config and load/save)
SYNTHCITY_GENERATORS = ("ctgan", "tvae", "pategan")


def create_plugin(
    name: str,
    random_state: int = 0,
    **kwargs: Any,
) -> Any:
    """
    创建 SynthCity 生成器插件。

    所有 kwargs（verbose、encoder args 等）均经 compat.synthcity.filter_plugin_params
    显式过滤后传入，禁止直接透传第三方参数。

    Args:
        name: 生成器名称，支持 "ctgan", "tvae", "pategan"
        random_state: 随机种子，保证可复现
        **kwargs: 传递给 SynthCity 插件的额外参数（经过滤后仅保留插件支持的）

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

    params: Dict[str, Any] = {"random_state": random_state, **kwargs}
    plugins = Plugins()
    params = filter_plugin_params(name, params, plugins)

    # 拷贝一份参数并显式移除 gpu_id，禁止透传到 SynthCity 插件
    params = dict(params)
    gpu_id = params.pop("gpu_id", None)  # noqa: F841 - 保留以便后续如需使用

    plugin = plugins.get(name, **params)
    logger.debug(f"已创建 SynthCity 插件: {name}, random_state={random_state}")
    return plugin
