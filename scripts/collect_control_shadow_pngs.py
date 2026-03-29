#!/usr/bin/env python3
"""
遍历 outputs 下名称包含 'control' 的文件夹，若存在 shadow_pair_metrics 子目录，
则将其中的全部 .png 文件复制到统一目录下（按 run 分子目录，便于区分）。

用法:
  python scripts/collect_control_shadow_pngs.py [--output-dir outputs] [--out outputs/control_shadow_metrics_pngs]
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _safe_dirname(s: str) -> str:
    """目录名中不宜做子路径的字符替换为下划线。"""
    for c in "/\\:*?\"<>|":
        s = s.replace(c, "_")
    return s.strip() or "unnamed"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将带 control 的实验目录下 shadow_pair_metrics 中的 PNG 统一复制到指定目录"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="outputs 根目录（在此下查找 *control* 实验目录）",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "control_shadow_metrics_pngs",
        help="PNG 统一保存的根目录（每个 run 在此下占一个子目录）",
    )
    args = parser.parse_args()

    outputs_dir = args.output_dir.resolve()
    out_root = args.out.resolve()
    if not outputs_dir.is_dir():
        print(f"outputs 目录不存在: {outputs_dir}")
        return

    # 只遍历一层子目录
    collected = 0
    for d in sorted(outputs_dir.iterdir()):
        if not d.is_dir() or "control" not in d.name:
            continue
        metrics_dir = d / "shadow_pair_metrics"
        if not metrics_dir.is_dir():
            continue
        pngs = list(metrics_dir.rglob("*.png"))
        if not pngs:
            continue
        # 每个 run 一个子目录，避免不同 run 间重名
        subdir = out_root / _safe_dirname(d.name)
        subdir.mkdir(parents=True, exist_ok=True)
        for p in pngs:
            rel = p.relative_to(metrics_dir)
            # 子目录内的 png 用扁平化文件名，避免重名
            flat_name = _safe_dirname(rel.with_suffix("").as_posix().replace("/", "_") + ".png")
            dest = subdir / flat_name
            shutil.copy2(p, dest)
            print(f"  {p.relative_to(outputs_dir)} -> {dest.relative_to(out_root)}")
            collected += 1

    if collected == 0:
        print(f"未在 {outputs_dir} 下找到任何带 'control' 且 shadow_pair_metrics 内含 .png 的目录")
        return
    print(f"共复制 {collected} 个 PNG 到: {out_root}")


if __name__ == "__main__":
    main()
