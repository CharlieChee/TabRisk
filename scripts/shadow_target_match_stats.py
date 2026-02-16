#!/usr/bin/env python3
"""
对 shadow 下某个 target（如 target_0）的 in/out synthetic 分别统计：匹配该 target 的条数（fraction_match，默认 fraction=0.8, epsilon=0.01），并输出 in 平均、out 平均。

用法:
  python scripts/shadow_target_match_stats.py --run-dir outputs/xxx --target-idx 0
  python scripts/shadow_target_match_stats.py --run-dir outputs/xxx --target-idx 0 --fraction 0.8 --epsilon 0.01
"""

from __future__ import annotations

import argparse
import re
import sys
from multiprocessing import Pool
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from synthgen.data.schema import Schema
from synthgen.match import run_all_methods


def _count_worker(task):
    """单文件：读 synthetic，算与 target 的 fraction_match 条数。返回 (path_str, count)。"""
    path_str, target_row, schema, epsilon, fraction = task
    syn = pd.read_csv(path_str)
    res = run_all_methods(
        syn,
        target_row,
        schema=schema,
        epsilon=epsilon,
        fraction=fraction,
    )
    c = int(res["fraction_match"][0].sum())
    return (path_str, c)


def _get_target_row(run_dir: Path, target_idx: int, candidate_df: pd.DataFrame) -> pd.Series:
    """从 run_dir 的 target_manifest 或 candidate 得到 target_idx 对应的 target 行（Series）。"""
    manifest_path = run_dir / "shadow" / "target_manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        row = manifest[manifest["target_idx"] == target_idx]
        if len(row) == 0:
            raise ValueError(f"target_manifest 中无 target_idx={target_idx}")
        candidate_row_idx = int(row.iloc[0]["candidate_row_idx"])
    else:
        candidate_row_idx = target_idx
    if candidate_row_idx < 0 or candidate_row_idx >= len(candidate_df):
        raise ValueError(f"candidate_row_idx={candidate_row_idx} 超出 candidate 行范围")
    return candidate_df.iloc[candidate_row_idx]


def main():
    parser = argparse.ArgumentParser(
        description="对 shadow target_X 的 in/out synthetic 统计匹配条数，输出 in 平均、out 平均。"
    )
    parser.add_argument("--run-dir", type=str, required=True, help="run 目录（含 shadow/target_X/, candidate.csv, schema.json）")
    parser.add_argument("--target-idx", type=int, default=0, help="target 编号，如 0 表示 target_0")
    parser.add_argument("--fraction", type=float, default=0.8, help="fraction_match 的 fraction")
    parser.add_argument("--epsilon", type=float, default=0.01, help="fraction_match 的 epsilon")
    parser.add_argument("--n-jobs", type=int, default=64, help="并行进程数（默认 64）")
    parser.add_argument("--output", type=str, default=None, help="结果写入该 CSV（可选）")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    if not run_dir.exists():
        print("错误: run_dir 不存在: {}".format(run_dir))
        sys.exit(1)

    candidate_path = run_dir / "candidate.csv"
    schema_path = run_dir / "schema.json"
    if not candidate_path.exists():
        print("错误: candidate.csv 不存在: {}".format(candidate_path))
        sys.exit(1)
    candidate_df = pd.read_csv(candidate_path)
    schema = Schema.load(str(schema_path)) if schema_path.exists() else None

    target_row = _get_target_row(run_dir, args.target_idx, candidate_df)
    target_dir = run_dir / "shadow" / "target_{}".format(args.target_idx)
    if not target_dir.exists():
        print("错误: shadow 目录不存在: {}".format(target_dir))
        sys.exit(1)

    def _round_num(path: Path) -> int:
        m = re.match(r"synthetic_round_(\d+)_(in|out)\.csv", path.name)
        return int(m.group(1)) if m else -1

    in_files = sorted(
        [f for f in target_dir.iterdir() if f.is_file() and f.name.endswith("_in.csv")],
        key=_round_num,
    )
    out_files = sorted(
        [f for f in target_dir.iterdir() if f.is_file() and f.name.endswith("_out.csv")],
        key=_round_num,
    )

    if not in_files or not out_files:
        print("错误: target_{} 下未找到 synthetic_round_*_in.csv 或 _out.csv".format(args.target_idx))
        sys.exit(1)

    n_jobs = max(1, args.n_jobs)
    task_args = (target_row, schema, args.epsilon, args.fraction)
    in_tasks = [(str(p),) + task_args for p in in_files]
    out_tasks = [(str(p),) + task_args for p in out_files]

    with Pool(processes=n_jobs) as pool:
        in_results = pool.map(_count_worker, in_tasks)
        out_results = pool.map(_count_worker, out_tasks)
    in_count_map = {path_str: c for path_str, c in in_results}
    out_count_map = {path_str: c for path_str, c in out_results}
    in_counts = [in_count_map[str(p)] for p in in_files]
    out_counts = [out_count_map[str(p)] for p in out_files]

    mean_in = sum(in_counts) / len(in_counts) if in_counts else 0.0
    mean_out = sum(out_counts) / len(out_counts) if out_counts else 0.0

    print("=" * 60)
    print("Shadow target_{} 匹配统计（fraction={}, epsilon={}, 并行进程数 {}）".format(
        args.target_idx, args.fraction, args.epsilon, n_jobs))
    print("=" * 60)
    print("  in  模型数: {},  匹配条数均值: {:.2f}".format(len(in_counts), mean_in))
    print("  out 模型数: {},  匹配条数均值: {:.2f}".format(len(out_counts), mean_out))
    print("  diff (in - out): {:.2f}".format(mean_in - mean_out))
    print()

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([
            {"role": "in", "round": i, "match_count": in_counts[i]} for i in range(len(in_counts))
        ] + [
            {"role": "out", "round": i, "match_count": out_counts[i]} for i in range(len(out_counts))
        ]).to_csv(out_path, index=False)
        print("明细已写入: {}".format(out_path))


if __name__ == "__main__":
    main()
