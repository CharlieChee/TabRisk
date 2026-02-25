#!/usr/bin/env python3
"""
Demo: 对 outputs/ 下 adult_openml + ctgan + rounds=20 的 run（train=200,500,1000,1500,2000 × 5 seeds）
执行 shadow_mia_from_raw.py，汇总 AUC，并画图：横轴 train_size，纵轴 AUC，带方差柱、5 条线（k-NN k=1,8,32, density, learned）。
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
RESULTS_DIR = PROJECT_ROOT / "scripts" / "demo_auc_results"
COMBINED_CSV = RESULTS_DIR / "all_runs.csv"
FIG_PATH = RESULTS_DIR / "auc_vs_trainsize.png"

TRAIN_SIZES = [200, 500, 1000, 1500, 2000]
N_SEEDS = 5
REQUIRED_SUBSTRINGS = ["adult_openml", "ctgan", "rounds20"]
TRAIN_PATTERN = re.compile(r"train(\d+)(?:_|$)")
SEED_PATTERN = re.compile(r"seed(\d+)(?:_|$)")


def find_run_dirs() -> list[tuple[Path, int, int]]:
    """返回 [(run_dir, train_size, seed), ...]，仅保留 train in TRAIN_SIZES 且 seed 在 5 个以内去重。"""
    if not OUTPUTS_DIR.exists():
        return []
    out: list[tuple[Path, int, int]] = []
    for d in OUTPUTS_DIR.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if not all(s in name for s in REQUIRED_SUBSTRINGS):
            continue
        m_train = TRAIN_PATTERN.search(name)
        m_seed = SEED_PATTERN.search(name)
        if not m_train or not m_seed:
            continue
        train_size = int(m_train.group(1))
        seed = int(m_seed.group(1))
        if train_size not in TRAIN_SIZES:
            continue
        out.append((d.resolve(), train_size, seed))
    # 每个 (train_size, seed) 只保留一个目录（若有多个时间戳取第一个）
    key_to_path: dict[tuple[int, int], Path] = {}
    for d, train_size, seed in sorted(out, key=lambda x: (x[1], x[2], str(x[0]))):
        key = (train_size, seed)
        if key not in key_to_path:
            key_to_path[key] = d
    return [(path, ts, seed) for (ts, seed), path in sorted(key_to_path.items(), key=lambda x: (x[0][0], x[0][1]))]


def run_mia_and_save(run_dir: Path, train_size: int, seed: int, n_jobs: int = 4) -> Path | None:
    """对单个 run 执行 shadow_mia_from_raw.py，结果保存到 RESULTS_DIR，返回 CSV 路径。"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / f"train{train_size}_seed{seed}.csv"
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
    if out_csv.exists():
        return out_csv
    return None


def load_and_aggregate() -> pd.DataFrame:
    """读取 RESULTS_DIR 下所有 run 的 CSV，并附加 train_size, seed 列；若存在 COMBINED_CSV 则优先读。"""
    if COMBINED_CSV.exists():
        df = pd.read_csv(COMBINED_CSV)
        return df
    rows = []
    for p in sorted(RESULTS_DIR.glob("train*_seed*.csv")):
        m = re.match(r"train(\d+)_seed(\d+)\.csv", p.name)
        if not m:
            continue
        train_size, seed = int(m.group(1)), int(m.group(2))
        try:
            one = pd.read_csv(p)
            one = one.assign(train_size=train_size, seed=seed)
            rows.append(one)
        except Exception as e:
            print(f"  skip {p.name}: {e}", file=sys.stderr)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def plot_auc_vs_trainsize(df: pd.DataFrame) -> None:
    """横轴 train_size，纵轴 AUC；每条线一个方法，带均值±std 的 error bar。"""
    if df.empty:
        raise SystemExit("无数据可画图")
    train_sizes = sorted(df["train_size"].unique())
    if not train_sizes:
        raise SystemExit("无 train_size 列或为空")

    # 要画的列：Naive 与 Delta (differential) 各一套；每条 (列, 标签, 颜色, 线型)
    series_config = [
        ("auc_naive_knn_k1", "Naive k-NN (k=1)", "C0", "-"),
        ("auc_delta_knn_k1", "Delta k-NN (k=1)", "C0", "--"),
        ("auc_naive_knn_k8", "Naive k-NN (k=8)", "C1", "-"),
        ("auc_delta_knn_k8", "Delta k-NN (k=8)", "C1", "--"),
        ("auc_naive_knn_k32", "Naive k-NN (k=32)", "C2", "-"),
        ("auc_delta_knn_k32", "Delta k-NN (k=32)", "C2", "--"),
        ("auc_naive_density", "Naive Density", "C3", "-"),
        ("auc_delta_density", "Delta Density", "C3", "--"),
        ("auc_naive_learned_lr", "Naive Learned (LR)", "C4", "-"),
        ("auc_delta_learned_lr", "Delta Learned (LR)", "C4", "--"),
    ]
    available = [c for c, _, _, _ in series_config if c in df.columns]
    if not available:
        raise SystemExit("DataFrame 中缺少所需的 AUC 列")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.array(train_sizes)
    for col, label, color, ls in series_config:
        if col not in df.columns:
            continue
        means = []
        stds = []
        for ts in train_sizes:
            sub = df.loc[df["train_size"] == ts, col].dropna()
            if len(sub) == 0:
                means.append(np.nan)
                stds.append(np.nan)
            else:
                means.append(sub.mean())
                stds.append(sub.std() if len(sub) > 1 else 0.0)
        means = np.array(means)
        stds = np.array(stds)
        ax.plot(x, means, "o-", label=label, color=color, linestyle=ls)
        ax.fill_between(x, means - stds, means + stds, color=color, alpha=0.15)
        ax.errorbar(x, means, yerr=stds, fmt="none", color=color, capsize=2)

    ax.set_xlabel("Train size")
    ax.set_ylabel("AUC")
    ax.set_xticks(train_sizes)
    ax.legend(loc="best", ncol=2, fontsize=8)
    ax.set_title("MIA AUC vs train size (adult_openml, CTGAN, rounds=20; Naive & Delta; mean ± std over 5 seeds)")
    ax.set_ylim(0.45, 1.0)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)
    print(f"图已保存: {FIG_PATH}")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Run MIA on adult_openml+ctgan+rounds20 runs and plot AUC vs train_size.")
    parser.add_argument("--skip-run", action="store_true", help="不执行 MIA，仅用已有 CSV 画图")
    parser.add_argument("--n-jobs", type=int, default=4, help="MIA 并行进程数")
    args = parser.parse_args()

    runs = find_run_dirs()
    print(f"找到 {len(runs)} 个 run 目录 (train_size × seed)")
    if not runs:
        print("未找到符合条件目录 (adult_openml, ctgan, rounds20, train in [200,500,1000,1500,2000])", file=sys.stderr)
        return 1

    # 检查是否每组 5 个 seed
    from collections import Counter
    train_counts = Counter(ts for _, ts, _ in runs)
    for ts in TRAIN_SIZES:
        n = train_counts.get(ts, 0)
        if n != N_SEEDS:
            print(f"  警告: train_size={ts} 有 {n} 个 run，期望 {N_SEEDS}", file=sys.stderr)

    if not args.skip_run:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        for run_dir, train_size, seed in runs:
            csv_path = RESULTS_DIR / f"train{train_size}_seed{seed}.csv"
            if csv_path.exists():
                print(f"  跳过已存在: train{train_size} seed{seed}")
                continue
            print(f"  运行: train{train_size} seed{seed} ...")
            run_mia_and_save(run_dir, train_size, seed, n_jobs=args.n_jobs)
        # 合并保存
        df = load_and_aggregate()
        if not df.empty:
            df.to_csv(COMBINED_CSV, index=False)
            print(f"已合并写入: {COMBINED_CSV}")
    else:
        df = load_and_aggregate()

    if df.empty:
        print("无结果数据，无法画图。请先去掉 --skip-run 运行一次。", file=sys.stderr)
        return 1
    plot_auc_vs_trainsize(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
