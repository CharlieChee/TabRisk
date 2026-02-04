"""
SynthCity 兼容层。

对传入 Plugins().get(name, **params) 的 params 做显式过滤/适配，
禁止直接假设 synthcity 插件 API 稳定。
"""

from __future__ import annotations

import inspect
from typing import Any, Dict

# 已知不兼容 SynthCity Plugin.__init__ 的参数（可随版本扩展）
# verbose、encoder 相关等均需在此显式声明或通过 inspect 过滤
KNOWN_UNSUPPORTED = frozenset({"verbose"})


def filter_plugin_params(name: str, params: Dict[str, Any], plugins: Any) -> Dict[str, Any]:
    """
    过滤 params，只保留插件构造函数支持的参数。

    1. 移除 KNOWN_UNSUPPORTED 中的键
    2. 若可获取插件类，则用 inspect.signature 进一步过滤，只传支持的 keys
    """
    filtered = {k: v for k, v in params.items() if k not in KNOWN_UNSUPPORTED}
    try:
        plugin_cls = None
        if hasattr(plugins, "plugins") and isinstance(getattr(plugins, "plugins"), dict):
            plugin_cls = plugins.plugins.get(name)
        elif hasattr(plugins, "_plugins") and isinstance(getattr(plugins, "_plugins"), dict):
            plugin_cls = plugins._plugins.get(name)
        if plugin_cls is not None and isinstance(plugin_cls, type):
            sig = inspect.signature(plugin_cls.__init__)
            allowed = {
                p
                for p in sig.parameters
                if p not in ("self", "args", "kwargs")
                and sig.parameters[p].kind
                not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
            }
            filtered = {k: v for k, v in filtered.items() if k in allowed}
    except Exception:
        pass
    return filtered
