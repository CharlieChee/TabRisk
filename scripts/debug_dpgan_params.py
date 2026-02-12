#!/usr/bin/env python3
"""
打印 SynthCity DPGAN 插件 __init__ 支持的参数（含 noise multiplier、clipping 等）。

用法（在项目根目录）：
    python scripts/debug_dpgan_params.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    from synthcity.plugins import Plugins
    import inspect

    plugins = Plugins()
    plugin_name = "dpgan"
    try:
        plugin = plugins.get(plugin_name, random_state=42)
    except Exception as e:
        print(f"[ERROR] plugins.get('{plugin_name}') 失败: {e!r}")
        return

    sig = inspect.signature(plugin.__class__.__init__)
    allowed = [
        p
        for p in sig.parameters
        if p not in ("self", "args", "kwargs")
        and sig.parameters[p].kind
        not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    ]
    print("=" * 60)
    print("SynthCity DPGAN 插件 __init__ 支持的参数")
    print("=" * 60)
    for p in sorted(allowed):
        print(f"   {p}")
    # 突出 DP 相关
    dp_like = [x for x in allowed if "dp_" in x or "clip" in x.lower() or "noise" in x.lower() or "epsilon" in x or "delta" in x or "sigma" in x.lower()]
    if dp_like:
        print()
        print("与 DP 相关 (clipping / noise / epsilon):")
        for p in sorted(dp_like):
            print(f"   {p}")


if __name__ == "__main__":
    main()
