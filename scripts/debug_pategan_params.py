#!/usr/bin/env python3
"""
排查 PATEGAN 的 n_iter 为何不生效：打印 SynthCity 插件构造函数支持的参数，
以及 filter_plugin_params 过滤前后的参数字典。

用法（在项目根目录）：
    python scripts/debug_pategan_params.py
"""

import sys
from pathlib import Path

# 确保能 import 项目包
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main():
    from synthcity.plugins import Plugins
    from synthgen.compat.synthcity import filter_plugin_params

    plugins = Plugins()
    plugin_name = "pategan"

    # 1) 打印 PATEGAN 插件 __init__ 支持的参数名
    plugin_cls = None
    if hasattr(plugins, "_plugins") and isinstance(getattr(plugins, "_plugins"), dict):
        plugin_cls = plugins._plugins.get(plugin_name)
    if plugin_cls is None and hasattr(plugins, "plugins") and isinstance(getattr(plugins, "plugins"), dict):
        plugin_cls = plugins.plugins.get(plugin_name)

    if plugin_cls is None:
        print(f"[ERROR] 未找到 SynthCity 插件: {plugin_name}")
        return

    import inspect
    sig = inspect.signature(plugin_cls.__init__)
    allowed = [
        p for p in sig.parameters
        if p not in ("self", "args", "kwargs")
        and sig.parameters[p].kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    ]
    print("=" * 60)
    print("1) SynthCity PATEGAN 插件 __init__ 支持的参数 (allowed)")
    print("=" * 60)
    for p in sorted(allowed):
        print(f"   {p}")
    print()

    # 2) 模拟 TabRisk 传入的参数（与 train 时一致）
    raw_params = {
        "random_state": 42,
        "n_iter": 2,
        "batch_size": 32,
        "generator_n_layers_hidden": 2,
        "generator_n_units_hidden": 256,
        "discriminator_n_layers_hidden": 2,
        "discriminator_n_units_hidden": 256,
    }
    print("2) 过滤前 (raw params，TabRisk 传入)")
    print("-" * 60)
    for k, v in sorted(raw_params.items()):
        print(f"   {k}: {v}")
    print()

    # 3) 过滤后
    filtered = filter_plugin_params(plugin_name, raw_params, plugins)
    print("3) 过滤后 (filter_plugin_params 保留的参数)")
    print("-" * 60)
    for k, v in sorted(filtered.items()):
        print(f"   {k}: {v}")
    print()

    # 4) 结论
    if "n_iter" in raw_params and "n_iter" not in filtered:
        print("[结论] n_iter 被 filter_plugin_params 丢弃，因为 PATEGAN 插件 __init__ 不支持该参数名。")
        print("       所以无论 CLI 传 n_iter=2 还是 3000，训练用的都是插件内部默认值。")
    elif "n_iter" in filtered:
        print("[结论] n_iter 已保留在 filtered 中，若训练结果仍与 n_iter 无关，说明插件内部未使用该参数。")
    else:
        print("[结论] 请根据上面 allowed 列表和 filtered 结果自行判断。")

if __name__ == "__main__":
    main()
