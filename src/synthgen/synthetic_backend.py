"""
SynthCity backend for tabular synthetic data generation.

This module encapsulates all SynthCity usage as the unified tabular synthetic
data framework. It provides a single point of integration so that the system
can be described as: "We use SynthCity as our tabular synthetic data
generation backend."

Supported generators: CTGAN, TVAE, PATEGAN, TabDDPM.

所有传入 SynthCity 的第三方参数必须经 compat.synthcity 显式过滤，禁止直接假设 API 稳定。
"""

from __future__ import annotations

from typing import Any, Dict

from loguru import logger

from synthgen.compat.synthcity import filter_plugin_params

# Supported SynthCity generator names (used in config and load/save)
SYNTHCITY_GENERATORS = ("ctgan", "tvae", "pategan", "tabddpm", "ddpm")


def _list_plugin_names(plugins: Any) -> list:
    """获取 SynthCity 已注册插件名列表，兼容不同版本 API。"""
    try:
        if hasattr(plugins, "list") and callable(plugins.list):
            out = plugins.list()
            if isinstance(out, (list, tuple)):
                return list(out)
            if hasattr(out, "__iter__") and not isinstance(out, (str, dict)):
                return list(out)
        if hasattr(plugins, "plugins") and isinstance(plugins.plugins, dict):
            return list(plugins.plugins.keys())
        if hasattr(plugins, "_plugins") and isinstance(plugins._plugins, dict):
            return list(plugins._plugins.keys())
    except Exception:
        pass
    return []


def _raise_plugin_not_found_friendly(
    plugins: Any,
    plugin_name: str,
    requested_name: str,
    original: Exception,
) -> None:
    """当插件不存在时，打印可用插件列表并给出升级/名称说明。"""
    available = _list_plugin_names(plugins)
    logger.info(
        "Plugin '%s' not found. Available SynthCity plugins: %s",
        plugin_name,
        available[:50] if len(available) > 50 else available,
    )
    if plugin_name == "ddpm" and plugin_name not in available:
        raise ImportError(
            "SynthCity 当前版本中未找到 'ddpm' 插件（TabRisk 将 model=tabddpm 映射为 SynthCity 的 'ddpm'）。"
            "请升级: pip install -U synthcity，或检查 README 的 Dependency Compatibility Notes。"
        ) from original
    raise ImportError(
        f"SynthCity 插件不可用: '{plugin_name}'（请求名: {requested_name}）。"
        "请确认已安装 synthcity 且版本支持该插件，或尝试: pip install -U synthcity。"
    ) from original


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
        name: 生成器名称，支持 "ctgan", "tvae", "pategan", "tabddpm", "ddpm"（tabddpm 会映射为 SynthCity 的 ddpm）
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

    # 别名映射：TabRisk 的 tabddpm 对应 SynthCity 的 ddpm；ddpm 保持不变
    plugin_name = "ddpm" if name == "tabddpm" else name
    if name == "tabddpm":
        logger.info("Mapping model 'tabddpm' -> synthcity plugin 'ddpm'")

    params: Dict[str, Any] = {"random_state": random_state, **kwargs}
    plugins = Plugins()
    # 使用实际插件名过滤参数（SynthCity 内注册名为 ddpm）
    params = filter_plugin_params(plugin_name, params, plugins)

    # 显式移除 gpu_id / cuda_visible_devices，禁止透传到 SynthCity 插件；device 仍然保留
    params = dict(params)
    params.pop("gpu_id", None)
    params.pop("cuda_visible_devices", None)

    try:
        plugin = plugins.get(plugin_name, **params)
    except KeyError as e:
        _raise_plugin_not_found_friendly(plugins, plugin_name, name, e)
    except Exception as e:
        err_msg = str(e).lower()
        if "doesn't exist" in err_msg or "not exist" in err_msg or "unknown" in err_msg:
            _raise_plugin_not_found_friendly(plugins, plugin_name, name, e)
        raise
    logger.debug(f"已创建 SynthCity 插件: {plugin_name} (requested name={name}), random_state={random_state}")
    return plugin
