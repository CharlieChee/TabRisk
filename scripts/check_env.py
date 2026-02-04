#!/usr/bin/env python3
"""
环境自检脚本：检查 torch / opacus / synthcity 版本与导入是否满足项目约束，
避免 opacus 与 torch 不兼容（nn.RMSNorm）导致的 import 崩溃。

用法:
  python scripts/check_env.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# 项目根目录（脚本在 scripts/ 下）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REQUIRED_TORCH_LO = "2.0"
REQUIRED_TORCH_HI = "2.1"
REQUIRED_OPACUS_LO = "1.3"
REQUIRED_OPACUS_HI = "1.5"
RMSNORM_HINT = (
    "若报错与 nn.RMSNorm 相关，请安装: torch>=2.0,<2.1 与 opacus>=1.3,<1.5\n"
    "详见 README 的 Dependency Compatibility Notes。"
)


def _version_tuple(s: str) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in s.strip().split(".")[:3])
    except (ValueError, AttributeError):
        return (0, 0, 0)


def check_torch() -> str | None:
    """检查 torch 版本是否在 2.0.x；若存在 nn.RMSNorm 也视为可接受（torch>=2.10）。"""
    try:
        import torch
    except ImportError:
        return "未安装 torch。请: pip install 'torch>=2.0,<2.1'"
    ver = getattr(torch, "__version__", "0.0.0")
    lo = _version_tuple(REQUIRED_TORCH_LO)
    hi = _version_tuple(REQUIRED_TORCH_HI)
    v = _version_tuple(ver)
    if lo <= v < hi:
        return None
    # 若 torch 已包含 nn.RMSNorm（2.10+），则 opacus 2.x 也可用，此处仅推荐 2.0.x 组合
    if hasattr(torch.nn, "RMSNorm"):
        return None
    return (
        f"当前 torch 版本 {ver} 不在推荐区间 [2.0, 2.1)。"
        f" 推荐: pip install 'torch>=2.0,<2.1'\n  {RMSNORM_HINT}"
    )


def check_opacus() -> str | None:
    """检查 opacus 版本是否在 [1.3, 1.5)，避免使用 nn.RMSNorm 的 1.5+。"""
    try:
        import opacus
    except ImportError:
        return None  # opacus 由 synthcity 间接依赖，未单独安装也可
    ver = getattr(opacus, "__version__", "0.0.0")
    lo = _version_tuple(REQUIRED_OPACUS_LO)
    hi = _version_tuple(REQUIRED_OPACUS_HI)
    v = _version_tuple(ver)
    if lo <= v < hi:
        return None
    if v >= hi:
        return (
            f"当前 opacus 版本 {ver} >= {REQUIRED_OPACUS_HI}，会使用 torch.nn.RMSNorm，"
            f"需 torch>=2.10。推荐使用 opacus<1.5: pip install 'opacus>=1.3,<1.5'\n  {RMSNORM_HINT}"
        )
    return (
        f"当前 opacus 版本 {ver} 低于推荐 {REQUIRED_OPACUS_LO}。"
        f" 推荐: pip install 'opacus>=1.3,<1.5'"
    )


def check_synthcity_import() -> str | None:
    """检查 synthcity 是否可正常 import（会间接加载 opacus）。"""
    try:
        from synthcity.plugins import Plugins
        Plugins()
    except ImportError as e:
        return f"SynthCity 未安装或依赖缺失: {e}\n  请: pip install -e . 或 pip install -r requirements.txt"
    except AttributeError as e:
        if "RMSNorm" in str(e) or "rms_norm" in str(e).lower():
            return (
                f"SynthCity 导入时触发了 opacus/torch 不兼容: {e}\n  {RMSNORM_HINT}"
            )
        return f"SynthCity 导入异常: {e}\n  {RMSNORM_HINT}"
    except Exception as e:
        return f"SynthCity 导入失败: {e}\n  {RMSNORM_HINT}"
    return None


def main() -> int:
    print("TabRisk 环境自检 (torch / opacus / synthcity)...")
    failed: list[str] = []

    err = check_torch()
    if err:
        print("[FAIL] torch:", err)
        failed.append("torch")
    else:
        import torch
        print("[OK] torch:", getattr(torch, "__version__", "?"))

    err = check_opacus()
    if err:
        print("[FAIL] opacus:", err)
        failed.append("opacus")
    else:
        try:
            import opacus
            print("[OK] opacus:", getattr(opacus, "__version__", "未安装（可选）"))
        except ImportError:
            print("[OK] opacus: 未单独安装（由 synthcity 拉取）")

    err = check_synthcity_import()
    if err:
        print("[FAIL] synthcity:", err)
        failed.append("synthcity")
    else:
        print("[OK] synthcity: 可正常导入")

    if failed:
        print("\n请按上述提示修复后重新运行: python scripts/check_env.py")
        return 1
    print("\n环境检查通过，可运行: python -m synthgen.train")
    return 0


if __name__ == "__main__":
    sys.exit(main())
