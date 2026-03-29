#!/usr/bin/env python3
"""
Demo: 对 outputs/ 下 指定数据集 + ctgan + 固定 config（如 train500_synth500_ctgan_1000iter_bs256）
按 rounds 变化的 run 执行 shadow_mia_from_raw.py，汇总 6 种攻击的 AUC，画图：横轴 rounds，纵轴 AUC。
用法: --dataset adult_openml --model ctgan [--base-filter 1000iter_bs256] --n-jobs 64
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SCRIPT_MIA = PROJECT_ROOT / "scripts" / "shadow_mia_from_raw.py"
RESULTS_BASE = PROJECT_ROOT / "scripts" / "demo_auc_results"

ROUNDS_PATTERN = re.compile(r"rounds(\d+)(?:_|$)")
SEED_PATTERN = re.compile(r"seed(\d+)(?:_|$)")

# 6 种攻击（与 MIA 文档一致）：每组取一个代表列
SERIES_CONFIG = [
    ("auc_naive_knn_k8", "Naive k-NN (k=8)", "C0"),
    ("auc_naive_density", "Naive Density", "C1"),
    ("auc_naive_learned_lr", "Naive Learned (LR)", "C2"),
    ("auc_delta_knn_k8", "Delta k-NN (k=8)", "C3"),
    ("auc_delta_density", "Delta Density", "C4"),
    ("auc_delta_learned_lr", "Delta Learned (LR)", "C5"),
]


def get_results_paths(dataset: str, model: str) -> tuple[Path, Path, Path]:
    """返回 (results_dir, combined_csv, fig_path)。"""
    results_dir = RESULTS_BASE / "auc_vs_rounds" / f"{dataset}_{model}"
    return results_dir, results_dir / "all_runs.csv", results_dir / "auc_vs_rounds.png"


def find_run_dirs(
    dataset: str, model: str, base_filter: list[str]
) -> list[tuple[Path, int, int]]:
    """返回 [(run_dir, rounds, seed), ...]，筛选 dataset + model + base_filter。"""
    if not OUTPUTS_DIR.exists():
        return []
    required = [dataset, model] + base_filter
    out: list[tuple[Path, int, int]] = []
    for d in OUTPUTS_DIR.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if not all(s in name for s in required):
            continue
        m_rounds = ROUNDS_PATTERN.search(name)
        m_seed = SEED_PATTERN.search(name)
        if not m_rounds or not m_seed:
            continue
        rounds = int(m_rounds.group(1))
        seed = int(m_seed.group(1))
        out.append((d.resolve(), rounds, seed))
    # 去重：同一 (rounds, seed) 保留一个（取第一个）
    key_to_path: dict[tuple[int, int], Path] = {}
    for d, rounds, seed in sorted(out, key=lambda x: (x[1], x[2], str(x[0]))):
        key = (rounds, seed)
        if key not in key_to_path:
            key_to_path[key] = d
    return [(path, r, s) for (r, s), path in sorted(key_to_path.items(), key=lambda x: (x[0][0], x[0][1]))]


def run_mia_and_save(
    run_dir: Path, rounds: int, seed: int, results_dir: Path, n_jobs: int = 4
) -> Path | None:
    """对单个 run 执行 shadow_mia_from_raw.py，结果保存到 results_dir。"""
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / f"rounds{rounds}_seed{seed}.csv"
    cmd = [
        sys.executable,
        str(SCRIPT_MIA),
        "--run-dir", str(run_dir),
        "--output", str(out_csv),
        "--n-jobs", str(n_jobs),
    ]
    ret = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=3600)
    if ret.returncode != 0:
        print(f"  [FAIL] {run_dir.name}: {ret.stderr[:200] if ret.stderr else ret.stdout[:200]}", file=sys.stderr)
        return None
    return out_csv if out_csv.exists() else None


def load_and_aggregate(results_dir: Path, combined_csv: Path) -> pd.DataFrame:
    """读取 results_dir 下所有 rounds*_seed*.csv；若存在 combined_csv 则优先读。"""
    if combined_csv.exists():
        return pd.read_csv(combined_csv)
    rows = []
    for p in sorted(results_dir.glob("rounds*_seed*.csv")):
        m = re.match(r"rounds(\d+)_seed(\d+)\.csv", p.name)
        if not m:
            continue
        rounds, seed = int(m.group(1)), int(m.group(2))
        try:
            one = pd.read_csv(p)
            one = one.assign(rounds=rounds, seed=seed)
            rows.append(one)
        except Exception as e:
            print(f"  skip {p.name}: {e}", file=sys.stderr)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def plot_auc_vs_rounds(df: pd.DataFrame, fig_path: Path, dataset: str, model: str) -> None:
    """横轴 rounds，纵轴 AUC；6 条线为 6 种攻击，带均值±std。"""
    if df.empty:
        raise SystemExit("无数据可画图")
    rounds_list = sorted(df["rounds"].unique())
    if not rounds_list:
        raise SystemExit("无 rounds 列或为空")

    available = [c for c, _, _ in SERIES_CONFIG if c in df.columns]
    if not available:
        raise SystemExit("DataFrame 中缺少所需的 AUC 列")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.array(rounds_list)
    for col, label, color in SERIES_CONFIG:
        if col not in df.columns:
            continue
        means, stds = [], []
        for r in rounds_list:
            sub = df.loc[df["rounds"] == r, col].dropna()
            if len(sub) == 0:
                means.append(np.nan)
                stds.append(np.nan)
            else:
                means.append(sub.mean())
                stds.append(sub.std() if len(sub) > 1 else 0.0)
        means = np.array(means)
        stds = np.array(stds)
        ax.plot(x, means, "o-", label=label, color=color)
        ax.fill_between(x, means - stds, means + stds, color=color, alpha=0.15)
        ax.errorbar(x, means, yerr=stds, fmt="none", color=color, capsize=2)

    ax.set_xlabel("Rounds")
    ax.set_ylabel("AUC")
    ax.set_xticks(rounds_list)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"MIA AUC vs Rounds ({dataset}, {model.upper()}, train500, 1000iter_bs256; 6 attacks; mean ± std over seeds)")
    ax.set_ylim(0.45, 1.0)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"图已保存: {fig_path}")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Run MIA on dataset + model runs varying by rounds, plot AUC vs rounds (6 attacks)."
    )
    parser.add_argument("--dataset", type=str, default="adult_openml", help="数据集名，与 outputs 目录名一致")
    parser.add_argument("--model", type=str, default="ctgan", help="合成模型，如 ctgan")
    parser.add_argument(
        "--base-filter",
        type=str,
        nargs="*",
        default=["1000iter_bs256"],
        help="目录名必须包含的片段，默认 1000iter_bs256 以只保留该 config",
    )
    parser.add_argument("--skip-run", action="store_true", help="不执行 MIA，仅用已有 CSV 画图")
    parser.add_argument("--n-jobs", type=int, default=4, help="MIA 并行进程数")
    args = parser.parse_args()

    dataset = args.dataset
    model = args.model
    base_filter = args.base_filter if args.base_filter else []
    results_dir, combined_csv, fig_path = get_results_paths(dataset, model)
    runs = find_run_dirs(dataset, model, base_filter)
    print(f"[{dataset}, {model}] base_filter={base_filter} 找到 {len(runs)} 个 run 目录 (rounds × seed)")
    if not runs:
        print(f"未找到符合条件目录 (需包含: {[dataset, model] + base_filter})", file=sys.stderr)
        return 1

    from collections import Counter
    rounds_counts = Counter(r for _, r, _ in runs)
    print("  rounds 分布:", dict(sorted(rounds_counts.items())))

    if not args.skip_run:
        results_dir.mkdir(parents=True, exist_ok=True)
        for run_dir, rounds, seed in runs:
            csv_path = results_dir / f"rounds{rounds}_seed{seed}.csv"
            if csv_path.exists():
                print(f"  跳过已存在: rounds{rounds} seed{seed}")
                continue
            print(f"  运行: rounds{rounds} seed{seed} ...")
            run_mia_and_save(run_dir, rounds, seed, results_dir, n_jobs=args.n_jobs)
        df = load_and_aggregate(results_dir, combined_csv)
        if not df.empty:
            df.to_csv(combined_csv, index=False)
            print(f"已合并写入: {combined_csv}")
    else:
        df = load_and_aggregate(results_dir, combined_csv)

    if df.empty:
        print("无结果数据，无法画图。请先去掉 --skip-run 运行一次。", file=sys.stderr)
        return 1
    plot_auc_vs_rounds(df, fig_path, dataset, model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
