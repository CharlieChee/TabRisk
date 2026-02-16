#!/usr/bin/env python3
"""
对 shadow 下某个 target（如 target_0）的 in/out synthetic 分别统计：匹配该 target 的条数（fraction_match，默认 fraction=0.8, epsilon=0.01），并输出 in 平均、out 平均。

用法:
  python scripts/shadow_target_match_stats.py --run-dir outputs/xxx --target-idx 0
  统计前 100 个 target 并汇总: python scripts/shadow_target_match_stats.py --run-dir outputs/xxx --max-targets 100 --output-dir outputs/xxx/shadow_stats
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


MEMBER_COL = "is_member"


def _get_target_row(run_dir: Path, target_idx: int, candidate_df: pd.DataFrame):
    """从 run_dir 的 target_manifest 或 candidate 得到 target_idx 对应的 target 行及 candidate 行号。返回 (target_row, candidate_row_idx)。"""
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
    return candidate_df.iloc[candidate_row_idx], candidate_row_idx


def _run_one_target(
    run_dir: Path,
    target_idx: int,
    candidate_df: pd.DataFrame,
    schema,
    args,
    n_jobs: int,
) -> dict:
    """对单个 target_idx 做完整统计，返回一行汇总（及 in/out 明细列表）。"""
    try:
        target_row, candidate_row_idx = _get_target_row(run_dir, target_idx, candidate_df)
    except Exception as e:
        return {"target_idx": target_idx, "error": str(e)}

    is_member = None
    if MEMBER_COL in candidate_df.columns:
        is_member = int(candidate_df.iloc[candidate_row_idx][MEMBER_COL])
    member_label = "member (训练集)" if is_member == 1 else "non-member (非训练集)" if is_member == 0 else "未知"

    target_dir = run_dir / "shadow" / "target_{}".format(target_idx)
    if not target_dir.exists():
        return {"target_idx": target_idx, "error": "shadow 目录不存在"}

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
        return {"target_idx": target_idx, "error": "无 in/out synthetic 文件"}

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
    mean_in = sum(in_counts) / len(in_counts)
    mean_out = sum(out_counts) / len(out_counts)

    main_synthetic_count = None
    main_synthetic_path = run_dir / "synthetic.csv"
    if main_synthetic_path.exists():
        main_syn = pd.read_csv(main_synthetic_path)
        res_main = run_all_methods(
            main_syn,
            target_row,
            schema=schema,
            epsilon=args.epsilon,
            fraction=args.fraction,
        )
        main_synthetic_count = int(res_main["fraction_match"][0].sum())

    return {
        "target_idx": target_idx,
        "candidate_row_idx": candidate_row_idx,
        "is_member": is_member,
        "member_label": member_label,
        "main_synthetic_match_count": main_synthetic_count,
        "n_in_models": len(in_counts),
        "n_out_models": len(out_counts),
        "mean_in": round(mean_in, 2),
        "mean_out": round(mean_out, 2),
        "diff": round(mean_in - mean_out, 2),
    }


def main():
    parser = argparse.ArgumentParser(
        description="对 shadow target_X 的 in/out synthetic 统计匹配条数，输出 in 平均、out 平均。"
    )
    parser.add_argument("--run-dir", type=str, required=True, help="run 目录（含 shadow/target_X/, candidate.csv, schema.json）")
    parser.add_argument("--target-idx", type=int, default=None, help="单个 target 编号（与 --max-targets 二选一）")
    parser.add_argument("--max-targets", type=int, default=None, help="统计前 N 个 target（如 100），逐个打印并最后汇总保存")
    parser.add_argument("--fraction", type=float, default=0.8, help="fraction_match 的 fraction")
    parser.add_argument("--epsilon", type=float, default=0.01, help="fraction_match 的 epsilon")
    parser.add_argument("--n-jobs", type=int, default=64, help="并行进程数（默认 64）")
    parser.add_argument("--output", type=str, default=None, help="单 target 时结果 CSV（可选）")
    parser.add_argument("--output-dir", type=str, default=None, help="--max-targets 时汇总 CSV 保存目录（必填）")
    args = parser.parse_args()

    if args.target_idx is not None and args.max_targets is not None:
        parser.error("--target-idx 与 --max-targets 二选一")
    if args.max_targets is not None and not args.output_dir:
        parser.error("使用 --max-targets 时请指定 --output-dir 以保存汇总结果")
    if args.target_idx is None and args.max_targets is None:
        args.target_idx = 0

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
    n_jobs = max(1, args.n_jobs)

    if args.max_targets is not None:
        shadow_dir = run_dir / "shadow"
        if not shadow_dir.exists():
            print("错误: shadow 目录不存在: {}".format(shadow_dir))
            sys.exit(1)
        target_indices = []
        for d in shadow_dir.iterdir():
            if not d.is_dir():
                continue
            if d.name.startswith("target_") and d.name[7:].isdigit():
                target_indices.append(int(d.name[7:]))
        target_indices = sorted(target_indices)[: args.max_targets]
        if not target_indices:
            print("错误: shadow 下未找到 target_0, target_1, ...")
            sys.exit(1)
        print("统计前 {} 个 target: {}".format(len(target_indices), target_indices[:20] if len(target_indices) > 20 else target_indices))
        print("fraction={}, epsilon={}, n_jobs={}\n".format(args.fraction, args.epsilon, n_jobs))

        rows = []
        for ti in target_indices:
            row = _run_one_target(run_dir, ti, candidate_df, schema, args, n_jobs)
            if "error" in row:
                print("target_{}: 跳过 ({})".format(ti, row["error"]))
                rows.append({"target_idx": ti, "error": row["error"]})
                continue
            print("target_{}: 身份={}, candidate_row={}, 主synthetic匹配={}, in均值={:.2f}, out均值={:.2f}, diff={:.2f}".format(
                ti,
                row["member_label"],
                row["candidate_row_idx"],
                row["main_synthetic_match_count"] if row["main_synthetic_match_count"] is not None else "N/A",
                row["mean_in"],
                row["mean_out"],
                row["diff"],
            ))
            rows.append(row)
        out_dir = Path(args.output_dir)
        if not out_dir.is_absolute():
            out_dir = PROJECT_ROOT / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / "shadow_targets_summary.csv"
        df = pd.DataFrame(rows)
        df.to_csv(summary_path, index=False)
        print("\n共 {} 个 target，汇总已保存: {}".format(len(rows), summary_path))
        return

    ti = args.target_idx
    row = _run_one_target(run_dir, ti, candidate_df, schema, args, n_jobs)
    if "error" in row:
        print("错误: target_{} -> {}".format(ti, row["error"]))
        sys.exit(1)

    print("=" * 60)
    print("Shadow target_{} 匹配统计（fraction={}, epsilon={}, 并行进程数 {}）".format(
        ti, args.fraction, args.epsilon, n_jobs))
    print("=" * 60)
    print("  target 身份: {} (candidate 行 {}，is_member={})".format(
        row["member_label"], row["candidate_row_idx"], row["is_member"] if row["is_member"] is not None else "N/A"))
    if row["main_synthetic_match_count"] is not None:
        print("  主 synthetic.csv 匹配条数: {}".format(row["main_synthetic_match_count"]))
    print("  in  模型数: {},  匹配条数均值: {:.2f}".format(row["n_in_models"], row["mean_in"]))
    print("  out 模型数: {},  匹配条数均值: {:.2f}".format(row["n_out_models"], row["mean_out"]))
    print("  diff (in - out): {:.2f}".format(row["diff"]))
    print()

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(out_path, index=False)
        print("汇总已写入: {}".format(out_path))


if __name__ == "__main__":
    main()
