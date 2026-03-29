#!/usr/bin/env python3
"""
将 outputs 下「同名（除时间戳外）重复」的 run 目录去重：每组只保留时间戳最新的一条，
其余移动到 outputs/replicate。

目录名约定：末尾为 _YYYYMMDD_HHMMSS，例如：
  train_adult_openml_preprocess_monotonic_train1000_synth1000_ctgan_..._control_seed42_20260220_234247

用法:
  # 仅打印将要移动的目录，不执行
  python scripts/dedupe_outputs_by_timestamp.py --outputs outputs

  # 真正执行移动
  python scripts/dedupe_outputs_by_timestamp.py --outputs outputs --execute
"""

from __future__ import annotations

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path

# 时间戳正则：末尾 _YYYYMMDD_HHMMSS
TS_PATTERN = re.compile(r"^(.+)_(\d{8}_\d{6})$")


def parse_name(name: str) -> tuple[str | None, str | None]:
    """返回 (base_name, timestamp) 或 (None, None)。"""
    m = TS_PATTERN.match(name.strip().rstrip("/"))
    if m:
        return m.group(1), m.group(2)
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="outputs 下去重：同 setting 只保留最新时间戳，旧目录移到 outputs/replicate"
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs",
        help="outputs 根目录",
    )
    parser.add_argument(
        "--replicate",
        type=Path,
        default=None,
        help="旧 run 移动目标目录，默认 outputs/replicate",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="执行移动；不传则只做 dry-run 打印",
    )
    args = parser.parse_args()

    outputs_dir = args.outputs.resolve()
    replicate_dir = (args.replicate or (outputs_dir / "replicate")).resolve()

    if not outputs_dir.is_dir():
        print(f"outputs 目录不存在: {outputs_dir}")
        return

    # 只处理 outputs 下直接子目录，且名称带时间戳；排除 replicate 自身
    groups: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    for d in outputs_dir.iterdir():
        if not d.is_dir():
            continue
        if d.resolve() == replicate_dir or d.name == "replicate":
            continue
        base, ts = parse_name(d.name)
        if base is not None and ts is not None:
            groups[base].append((d, ts))

    # 每组按时间戳排序，保留最新，其余待移动
    to_move: list[Path] = []
    for base, items in groups.items():
        if len(items) <= 1:
            continue
        items.sort(key=lambda x: x[1], reverse=True)  # 最新在前
        # 从第二项起都要移走
        for d, _ in items[1:]:
            to_move.append(d)

    if not to_move:
        print("没有需要移动的重复目录。")
        return

    print(f"以下 {len(to_move)} 个目录将移动到 {replicate_dir}（每组保留时间戳最新的一条）：")
    for p in sorted(to_move, key=lambda x: x.name):
        print(f"  {p.relative_to(outputs_dir)}")

    if not args.execute:
        print("\n此为 dry-run，未实际移动。要执行请加上 --execute")
        return

    replicate_dir.mkdir(parents=True, exist_ok=True)
    for p in to_move:
        dest = replicate_dir / p.name
        if dest.exists():
            print(f"跳过（目标已存在）: {p.name}")
            continue
        shutil.move(str(p), str(dest))
        print(f"已移动: {p.name} -> replicate/")
    print("完成。")


if __name__ == "__main__":
    main()
