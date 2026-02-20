#!/usr/bin/env python3
"""
遍历 outputs 下名称包含 'control' 的文件夹，若存在 shadow_pair_metrics 子目录，
则读取其中的 global_delta_stats.json 和 control_comparison.json，
并汇总为一张 CSV：包含 train size、iter、batchsize（及重复实验编号）、以及两个 JSON 的展平字段与原始内容。

用法:
  python scripts/collect_control_shadow_metrics.py [--output outputs/control_shadow_metrics_summary.csv]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_params_from_dirname(dirname: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    从实验目录名解析 train size、iter、batchsize。
    目录名示例: train_adult_openml_preprocess_monotonic_train2000_synth2000_ctgan_300iter_bs1024_..._control_...
    返回 (train_size, iter, batchsize)，无法解析的为 None。
    """
    train_size = None
    iters = None
    batchsize = None

    # train size: _train(\d+)_synth 或 monotonic_train(\d+)
    m = re.search(r"_train(\d+)_synth", dirname)
    if m:
        train_size = int(m.group(1))

    # iter: (\d+)iter
    m = re.search(r"(\d+)iter", dirname)
    if m:
        iters = int(m.group(1))

    # batchsize: bs(\d+)
    m = re.search(r"bs(\d+)", dirname)
    if m:
        batchsize = int(m.group(1))

    return (train_size, iters, batchsize)


def flatten_json(obj: Any, prefix: str = "") -> Dict[str, Any]:
    """将嵌套 dict 展平为一层 key，key 用 prefix 和 _ 连接。"""
    out: Dict[str, Any] = {}
    if not isinstance(obj, dict):
        out[prefix.rstrip("_")] = obj
        return out
    for k, v in obj.items():
        new_key = f"{prefix}{k}"
        if isinstance(v, dict) and v and not any(isinstance(x, (dict, list)) for x in v.values()):
            for k2, v2 in v.items():
                out[f"{new_key}_{k2}"] = v2
        else:
            out[new_key] = v if not isinstance(v, (dict, list)) else json.dumps(v, ensure_ascii=False)
    return out


def load_json_safe(path: Path) -> Optional[Dict[str, Any]]:
    """安全读取 JSON，失败返回 None。"""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="收集带 control 的实验目录下 shadow_pair_metrics 的 JSON 并汇总为 CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "control_shadow_metrics_summary.csv",
        help="输出 CSV 路径",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="outputs 根目录",
    )
    args = parser.parse_args()

    outputs_dir = args.output_dir.resolve()
    if not outputs_dir.is_dir():
        print(f"outputs 目录不存在: {outputs_dir}")
        return

    # 只遍历一层子目录（实验 run 目录）
    control_dirs: List[Path] = []
    for d in outputs_dir.iterdir():
        if not d.is_dir():
            continue
        if "control" not in d.name:
            continue
        metrics_dir = d / "shadow_pair_metrics"
        if not metrics_dir.is_dir():
            continue
        gds = metrics_dir / "global_delta_stats.json"
        cc = metrics_dir / "control_comparison.json"
        if gds.exists() and cc.exists():
            control_dirs.append(d)

    if not control_dirs:
        print(f"未在 {outputs_dir} 下找到任何带 'control' 且含有 shadow_pair_metrics/global_delta_stats.json 与 control_comparison.json 的目录")
        return

    rows: List[Dict[str, Any]] = []
    # 用于对 (train, iter, bs) 重复实验编号
    key_counts: Dict[Tuple[int, int, int], int] = {}

    for run_dir in sorted(control_dirs):
        dirname = run_dir.name
        train_size, iters, batchsize = parse_params_from_dirname(dirname)
        # 重复实验编号：同一 (train_size, iter, batchsize) 下 run_id 递增
        key = (
            train_size if train_size is not None else -1,
            iters if iters is not None else -1,
            batchsize if batchsize is not None else -1,
        )
        key_counts[key] = key_counts.get(key, 0) + 1
        run_id = key_counts[key]

        gds_path = run_dir / "shadow_pair_metrics" / "global_delta_stats.json"
        cc_path = run_dir / "shadow_pair_metrics" / "control_comparison.json"
        gds_data = load_json_safe(gds_path)
        cc_data = load_json_safe(cc_path)

        row: Dict[str, Any] = {
            "train_size": train_size,
            "iter": iters,
            "batchsize": batchsize,
            "run_id": run_id,
            "run_dir_name": dirname,
        }

        if gds_data:
            flat_gds = flatten_json(gds_data, prefix="gds_")
            for k, v in flat_gds.items():
                row[k] = v
            row["global_delta_stats_json"] = json.dumps(gds_data, ensure_ascii=False, indent=0)
        else:
            row["global_delta_stats_json"] = ""

        if cc_data:
            flat_cc = flatten_json(cc_data, prefix="cc_")
            for k, v in flat_cc.items():
                row[k] = v
            row["control_comparison_json"] = json.dumps(cc_data, ensure_ascii=False, indent=0)
        else:
            row["control_comparison_json"] = ""

        rows.append(row)

    df = pd.DataFrame(rows)

    # 列顺序：先固定列，再 gds_*、cc_*，最后两个完整 JSON 列
    fixed = ["train_size", "iter", "batchsize", "run_id", "run_dir_name"]
    gds_cols = sorted(c for c in df.columns if c.startswith("gds_") and c != "global_delta_stats_json")
    cc_cols = sorted(c for c in df.columns if c.startswith("cc_") and c != "control_comparison_json")
    rest = [c for c in df.columns if c not in fixed and c not in gds_cols and c not in cc_cols and c not in ("global_delta_stats_json", "control_comparison_json")]
    json_cols = ["global_delta_stats_json", "control_comparison_json"]
    col_order = fixed + gds_cols + cc_cols + rest + json_cols
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"已写入 {len(df)} 条记录到: {args.output}")


if __name__ == "__main__":
    main()
