#!/usr/bin/env python3
"""
对 ctgan、rounds=20、多 iter（如 200,500,1000,1500,2000）、多 seed（如 5 组）的 run 逐个跑 MIA，
再按 iter 对 AUC 求均值（跨 seed）。

用法:
  python scripts/run_mia_by_iter_seeds.py --iters 200 500 1000 1500 2000 --rounds 20 --n-seeds 5 --n-jobs 32
  python scripts/run_mia_by_iter_seeds.py --iters 200 500 1000 1500 2000 --rounds 20 --seed-list 42 43 44 45 46 --n-jobs 32 --skip-run   # 仅汇总已有 CSV
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SCRIPT_MIA = PROJECT_ROOT / "scripts" / "shadow_mia_from_raw.py"
RESULTS_BASE = PROJECT_ROOT / "scripts" / "demo_auc_results"

# 从目录名解析 iter / rounds / seed
ITER_PATTERN = re.compile(r"(\d+)iter(?:_|$|bs)")
ROUNDS_PATTERN = re.compile(r"rounds(\d+)(?:_|$)")
SEED_PATTERN = re.compile(r"(?:control_)?seed(\d+)(?:_|$)")


def find_run_dirs(
    model: str = "ctgan",
    rounds: int = 20,
    iters: list[int] | None = None,
    seed_list: list[int] | None = None,
    dataset_substr: str = "adult_openml",
) -> list[tuple[Path, int, int]]:
    """
    扫描 outputs/，返回 [(run_dir, n_iter, seed), ...]。
    要求目录名含 model、rounds{N}、{iter}iter、control、seed{M}；
    若指定 iters/seed_list 则过滤。
    """
    if not OUTPUTS_DIR.exists():
        return []
    iters_set = set(iters) if iters else None
    seeds_set = set(seed_list) if seed_list else None
    out: list[tuple[Path, int, int]] = []
    for d in OUTPUTS_DIR.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if model not in name or dataset_substr not in name or "control" not in name:
            continue
        m_iter = ITER_PATTERN.search(name)
        m_rounds = ROUNDS_PATTERN.search(name)
        m_seed = SEED_PATTERN.search(name)
        if not m_iter or not m_rounds or not m_seed:
            continue
        n_iter = int(m_iter.group(1))
        n_rounds = int(m_rounds.group(1))
        seed = int(m_seed.group(1))
        if n_rounds != rounds:
            continue
        if iters_set is not None and n_iter not in iters_set:
            continue
        if seeds_set is not None and seed not in seeds_set:
            continue
        out.append((d.resolve(), n_iter, seed))
    # 去重：同一 (n_iter, seed) 保留一个（取第一个）
    key_to_path: dict[tuple[int, int], Path] = {}
    for d, n_iter, seed in sorted(out, key=lambda x: (x[1], x[2], str(x[0]))):
        key = (n_iter, seed)
        if key not in key_to_path:
            key_to_path[key] = d
    return [(path, ni, s) for (ni, s), path in sorted(key_to_path.items(), key=lambda x: (x[0][0], x[0][1]))]


def run_mia(run_dir: Path, out_csv: Path, n_jobs: int) -> bool:
    """对单个 run 执行 shadow_mia_from_raw.py，结果写入 out_csv。"""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SCRIPT_MIA),
        "--run-dir", str(run_dir),
        "--output", str(out_csv),
        "--n-jobs", str(n_jobs),
    ]
    ret = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=7200)
    if ret.returncode != 0:
        print(f"  [FAIL] {run_dir.name[:60]}... : {ret.stderr[:300] if ret.stderr else ret.stdout[:300]}", file=sys.stderr)
        return False
    return out_csv.exists()


def load_all_csvs(results_dir: Path) -> pd.DataFrame:
    """读取 results_dir 下 iter*_seed*.csv，返回带 iter/seed 列的合并表。"""
    rows = []
    for p in sorted(results_dir.glob("iter*_seed*.csv")):
        m = re.match(r"iter(\d+)_seed(\d+)\.csv", p.name)
        if not m:
            continue
        n_iter, seed = int(m.group(1)), int(m.group(2))
        try:
            one = pd.read_csv(p)
            one = one.assign(n_iter=n_iter, seed=seed)
            rows.append(one)
        except Exception as e:
            print(f"  skip {p.name}: {e}", file=sys.stderr)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def auc_columns(df: pd.DataFrame) -> list[str]:
    """所有 auc_* 列。"""
    return [c for c in df.columns if c.startswith("auc_") and c not in ("n_iter", "seed")]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run MIA on ctgan rounds=20 runs by iter × seed, then average AUC by iter."
    )
    parser.add_argument(
        "--iters",
        type=int,
        nargs="+",
        default=[200, 500, 1000, 1500, 2000],
        help="n_iter 列表，默认 200 500 1000 1500 2000",
    )
    parser.add_argument("--rounds", type=int, default=20, help="只保留目录名含 rounds<N> 且 N=rounds")
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        help="每个 iter 需要的 seed 数量（与 --seed-list 二选一）",
    )
    parser.add_argument(
        "--seed-list",
        type=int,
        nargs="+",
        default=None,
        help="显式指定 seed 列表，如 42 43 44 45 46；不指定则用前 n-seeds 个找到的 seed",
    )
    parser.add_argument("--model", type=str, default="ctgan")
    parser.add_argument("--dataset", type=str, default="adult_openml", help="目录名需包含的 dataset 片段")
    parser.add_argument("--n-jobs", type=int, default=32, help="MIA 并行进程数")
    parser.add_argument("--skip-run", action="store_true", help="不执行 MIA，仅用已有 CSV 汇总求均值")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="结果目录，默认 scripts/demo_auc_results/mia_by_iter_ctgan_rounds20",
    )
    args = parser.parse_args()

    iters = args.iters
    rounds = args.rounds
    model = args.model
    seed_list = args.seed_list
    if seed_list is None:
        # 先扫描得到所有 (iter, seed)，再按 iter 取前 n_seeds 个 seed
        all_runs = find_run_dirs(
            model=model,
            rounds=rounds,
            iters=iters,
            seed_list=None,
            dataset_substr=args.dataset,
        )
        # 按 iter 分组，每个 iter 取前 n_seeds 个不同 seed
        from collections import defaultdict
        by_iter: dict[int, list[tuple[Path, int, int]]] = defaultdict(list)
        for run_dir, n_iter, seed in all_runs:
            by_iter[n_iter].append((run_dir, n_iter, seed))
        runs = []
        for n_iter in sorted(by_iter.keys()):
            seen_seeds: set[int] = set()
            for run_dir, ni, seed in sorted(by_iter[n_iter], key=lambda x: (x[2], str(x[0]))):
                if seed in seen_seeds:
                    continue
                seen_seeds.add(seed)
                if len(seen_seeds) > args.n_seeds:
                    break
                runs.append((run_dir, ni, seed))
        runs.sort(key=lambda x: (x[1], x[2]))
    else:
        runs = find_run_dirs(
            model=model,
            rounds=rounds,
            iters=iters,
            seed_list=seed_list,
            dataset_substr=args.dataset,
        )

    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_BASE / f"mia_by_iter_{model}_rounds{rounds}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"条件: model={model}, rounds={rounds}, iters={iters}, dataset 含 '{args.dataset}'")
    print(f"找到 {len(runs)} 个 run (iter × seed)")
    if not runs:
        print("未找到符合条件的 run 目录。", file=sys.stderr)
        return 1

    if not args.skip_run:
        for run_dir, n_iter, seed in runs:
            csv_path = out_dir / f"iter{n_iter}_seed{seed}.csv"
            if csv_path.exists():
                print(f"  跳过已存在: iter{n_iter} seed{seed}")
                continue
            print(f"  运行: iter{n_iter} seed{seed} ...")
            run_mia(run_dir, csv_path, n_jobs=args.n_jobs)

    df = load_all_csvs(out_dir)
    if df.empty:
        print("无结果 CSV，无法汇总。请先去掉 --skip-run 运行一次。", file=sys.stderr)
        return 1

    cols = auc_columns(df)
    if not cols:
        print("未找到 auc_* 列。", file=sys.stderr)
        return 1

    # 按 n_iter 求均值（跨 seed）
    grp = df.groupby("n_iter", as_index=True)[cols].agg(["mean", "std"]).reset_index()
    mean_flat = grp["n_iter"].to_frame()
    for c in cols:
        mean_flat[c] = grp[(c, "mean")]
        mean_flat[f"{c}_std"] = grp[(c, "std")]
    summary_path = out_dir / "summary_mean_by_iter.csv"
    mean_flat.to_csv(summary_path, index=False)
    print(f"已写入: {summary_path}")

    # 全量明细（每 run 一行）
    detail_path = out_dir / "all_runs.csv"
    df.to_csv(detail_path, index=False)
    print(f"已写入: {detail_path}")

    # 打印均值表（主要 AUC 列）
    print("\n--- AUC 均值 (按 iter，跨 seed) ---")
    iter_order = sorted(df["n_iter"].unique())
    for n_iter in iter_order:
        sub = df.loc[df["n_iter"] == n_iter, cols]
        means = sub.mean()
        stds = sub.std().fillna(0.0)
        print(f"  iter={n_iter}: " + " | ".join(f"{c}: {means[c]:.4f}±{stds[c]:.4f}" for c in cols[:6]))
    print("\n  (完整见 summary_mean_by_iter.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
