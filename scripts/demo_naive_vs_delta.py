#!/usr/bin/env python3
"""
单 run 展示 Delta（差分）相对 Naive 的优越性：
1) AUC 对比柱状图：三组攻击（k-NN / Density / Learned）的 Naive vs Delta AUC；
2) 分数分布图：Naive 下 member/control 分数重叠 vs Delta 下 ±δ 的更好分离。
用法: --run-dir outputs/train_adult_openml_..._seed42_... [--outdir 输出目录] [--n-jobs 4]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_MIA = PROJECT_ROOT / "scripts" / "shadow_mia_from_raw.py"
DEFAULT_OUTDIR = PROJECT_ROOT / "scripts" / "demo_auc_results" / "naive_vs_delta"


def run_mia_and_export(run_dir: Path, out_auc_csv: Path, out_scores_csv: Path, n_jobs: int) -> bool:
    """对 run_dir 跑 MIA 并导出 AUC + 每对分数。"""
    cmd = [
        sys.executable,
        str(SCRIPT_MIA),
        "--run-dir", str(run_dir),
        "--output", str(out_auc_csv),
        "--export-scores", str(out_scores_csv),
        "--n-jobs", str(n_jobs),
    ]
    ret = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=3600)
    if ret.returncode != 0:
        print(ret.stderr or ret.stdout, file=sys.stderr)
        return False
    return out_auc_csv.exists() and out_scores_csv.exists()


def plot_auc_comparison(auc_row: pd.Series, fig_path: Path, run_name: str) -> None:
    """Bar chart: Naive vs Delta AUC for k-NN, Density, Learned."""
    groups = [
        ("k-NN (k=8)", "auc_naive_knn_k8", "auc_delta_knn_k8"),
        ("Density", "auc_naive_density", "auc_delta_density"),
        ("Learned (LR)", "auc_naive_learned_lr", "auc_delta_learned_lr"),
    ]
    labels = [g[0] for g in groups]
    naive_vals = [auc_row.get(g[1], np.nan) for g in groups]
    delta_vals = [auc_row.get(g[2], np.nan) for g in groups]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar(x - w / 2, naive_vals, w, label="Naive", color="steelblue", edgecolor="black", linewidth=0.8)
    bars2 = ax.bar(x + w / 2, delta_vals, w, label="Differential", color="coral", edgecolor="black", linewidth=0.8)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.8, label="Random (0.5)")
    ax.set_ylabel("AUC", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_ylim(0.45, 1.02)
    ax.set_title(f"Naive vs Differential: AUC comparison\n({run_name})", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    for b in bars1:
        h = b.get_height()
        if np.isfinite(h):
            ax.annotate(f"{h:.3f}", xy=(b.get_x() + b.get_width() / 2, h), ha="center", va="bottom", fontsize=9, rotation=0)
    for b in bars2:
        h = b.get_height()
        if np.isfinite(h):
            ax.annotate(f"{h:.3f}", xy=(b.get_x() + b.get_width() / 2, h), ha="center", va="bottom", fontsize=9, rotation=0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")


def _plot_one_score_distribution(
    naive_m: pd.Series, naive_c: pd.Series, diff_m: pd.Series, diff_c: pd.Series,
    title_suffix: str, fig_path: Path, run_name: str,
) -> None:
    """One 1x2 figure: Naive (member vs control) | Differential (member vs control). All labels in English."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    nb = lambda n: min(50, max(20, n // 5))

    ax1.hist(naive_m, bins=nb(len(naive_m)), alpha=0.6, density=True, label="Member", color="steelblue", edgecolor="white")
    ax1.hist(naive_c, bins=nb(len(naive_c)), alpha=0.6, density=True, label="Control", color="coral", edgecolor="white")
    ax1.set_xlabel(f"Naive {title_suffix} score")
    ax1.set_ylabel("Density")
    ax1.set_title("Naive: Member vs Control\n(more overlap)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.hist(diff_m, bins=nb(len(diff_m)), alpha=0.6, density=True, label="Member", color="steelblue", edgecolor="white")
    ax2.hist(diff_c, bins=nb(len(diff_c)), alpha=0.6, density=True, label="Control", color="coral", edgecolor="white")
    if (diff_m.values < 0).any() or (diff_c.values > 0).any():
        ax2.axvline(0, color="gray", linestyle="--", alpha=0.8)
    ax2.set_xlabel(f"Differential {title_suffix} score")
    ax2.set_ylabel("Density")
    ax2.set_title("Differential: Member vs Control\n(better separation)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Score distribution — {title_suffix}", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")


def plot_score_distributions_knn_or_density(
    df: pd.DataFrame, fig_path: Path, run_name: str, k: int | None = None, density: bool = False
) -> None:
    """Score distribution for one attack: k-NN (k=1,8,32) or Density. k=None and density=True for Density."""
    if density:
        m_col, c_col, diff_col = "naive_density_member", "naive_density_control", "delta_density"
        title_suffix = "Density"
    else:
        if k is None:
            k = 8
        m_col = f"naive_knn_k{k}_member"
        c_col = f"naive_knn_k{k}_control"
        diff_col = f"delta_knn_k{k}"
        title_suffix = f"k-NN (k={k})"
    if m_col not in df.columns or diff_col not in df.columns:
        return
    naive_m = df[m_col].dropna()
    naive_c = df[c_col].dropna()
    diff_m = df[diff_col].dropna()
    diff_c = (-df[diff_col]).dropna()
    _plot_one_score_distribution(naive_m, naive_c, diff_m, diff_c, title_suffix, fig_path, run_name)


def plot_learned_score_distribution(df: pd.DataFrame, fig_path: Path, run_name: str) -> None:
    """Learned (LR) score distribution: Naive vs Differential. All English."""
    m_col = "naive_learned_member"
    c_col = "naive_learned_control"
    dm_col = "delta_learned_member"
    dc_col = "delta_learned_control"
    if m_col not in df.columns or dm_col not in df.columns:
        return
    naive_m = df[m_col].dropna()
    naive_c = df[c_col].dropna()
    diff_m = df[dm_col].dropna()
    diff_c = df[dc_col].dropna()
    _plot_one_score_distribution(naive_m, naive_c, diff_m, diff_c, "Learned (LR)", fig_path, run_name)


def main() -> int:
    parser = argparse.ArgumentParser(description="单 run 展示 Naive vs Delta 的 AUC 与分数分布。")
    parser.add_argument("--run-dir", type=str, required=True, help="run 目录路径（相对或绝对）")
    parser.add_argument("--outdir", type=str, default=None, help="图与中间 CSV 输出目录，默认 scripts/demo_auc_results/naive_vs_delta")
    parser.add_argument("--mia-csv", type=str, default=None, help="已有 MIA AUC CSV 则直接读，不重跑 MIA")
    parser.add_argument("--scores-csv", type=str, default=None, help="已有 per-pair 分数 CSV 则直接读")
    parser.add_argument("--n-jobs", type=int, default=4, help="MIA 并行进程数")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    if not run_dir.exists():
        print(f"错误: run_dir 不存在: {run_dir}", file=sys.stderr)
        return 1

    outdir = Path(args.outdir) if args.outdir else DEFAULT_OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)
    run_name = run_dir.name[:60] + ("..." if len(run_dir.name) > 60 else "")

    auc_csv = outdir / "mia_auc.csv"
    scores_csv = outdir / "per_pair_scores.csv"

    if args.mia_csv and Path(args.mia_csv).exists():
        auc_df = pd.read_csv(args.mia_csv)
        if auc_df.empty:
            print("错误: 提供的 MIA CSV 为空", file=sys.stderr)
            return 1
        auc_row = auc_df.iloc[0]
        if not args.scores_csv or not Path(args.scores_csv).exists():
            print("未提供 --scores-csv，将重跑 MIA 以导出 per-pair 分数（仅用于分布图）")
            if not run_mia_and_export(run_dir, auc_csv, scores_csv, args.n_jobs):
                return 1
            scores_csv = outdir / "per_pair_scores.csv"
        else:
            scores_csv = Path(args.scores_csv)
    else:
        if not run_mia_and_export(run_dir, auc_csv, scores_csv, args.n_jobs):
            return 1
        auc_df = pd.read_csv(auc_csv)
        auc_row = auc_df.iloc[0]

    if not scores_csv.exists():
        print("Warning: no per-pair scores file, only plotting AUC comparison.")
    else:
        scores_df = pd.read_csv(scores_csv)
        for k in [1, 8, 32]:
            plot_score_distributions_knn_or_density(
                scores_df, outdir / f"score_distributions_knn_k{k}.png", run_name, k=k
            )
        plot_score_distributions_knn_or_density(
            scores_df, outdir / "score_distributions_density.png", run_name, density=True
        )
        plot_learned_score_distribution(scores_df, outdir / "score_distributions_learned.png", run_name)

    plot_auc_comparison(auc_row, outdir / "auc_naive_vs_delta.png", run_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
