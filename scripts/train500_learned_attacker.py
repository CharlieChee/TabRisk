#!/usr/bin/env python3
"""
Step 4: Learned Attacker — 用 pair_metrics + pair_metrics_control 组合特征训练 LR，5-fold CV 得 AUC，
在 train=500, iter=1000 下画 learned AUC vs rounds。

用法:
  python scripts/train500_learned_attacker.py --summary outputs/adult_train500_full_summary.csv --out outputs/figures/learned_attacker_auc_vs_rounds.png
  或指定 outputs 直接扫描（无 summary 时）:
  python scripts/train500_learned_attacker.py --outputs outputs --out outputs/figures/learned_attacker_auc_vs_rounds.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 可选特征列（pair_metrics 中）：MMD、delta、列统计等
FEATURE_CANDIDATES = [
    "mmd_numeric", "mmd_mixed", "min_dist_in", "min_dist_out", "delta_min_dist",
    "num_mean_diff_mean", "num_mean_diff_max", "num_std_diff_mean", "num_std_diff_max",
    "cat_tv_mean", "cat_tv_max",
]


def _parse_dirname(dirname: str) -> Dict[str, Any]:
    out = {"train_rows": None, "n_iter": None, "rounds": None, "batch_size": None, "seed": None}
    m = re.search(r"_train(\d+)_synth", dirname)
    if m:
        out["train_rows"] = int(m.group(1))
    m = re.search(r"_(\d+)iter", dirname)
    if m:
        out["n_iter"] = int(m.group(1))
    m = re.search(r"_rounds(\d+)_", f"_{dirname}_")
    if m:
        out["rounds"] = int(m.group(1))
    m = re.search(r"_bs(\d+)(?:_|$)", dirname)
    if m:
        out["batch_size"] = int(m.group(1))
    m = re.search(r"_seed(\d+)(?:_|$)", dirname)
    if m:
        out["seed"] = int(m.group(1))
    return out


def _get_rounds_from_config(run_dir: Path) -> Optional[int]:
    cfg_path = run_dir / "train_config.yaml"
    if not cfg_path.is_file():
        return None
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(cfg_path)
        v = OmegaConf.select(cfg, "shadow.num_shadow_rounds", default=None)
        return int(v) if v is not None else None
    except Exception:
        return None


def compute_learned_auc_one_run(run_dir: Path, cv: int = 5) -> Tuple[Optional[float], Optional[str]]:
    """
    对单个 run：读 pair_metrics（label=1）与 pair_metrics_control（label=0），
    取共有数值列作为特征，LogisticRegression 5-fold CV 得到 learned AUC。
    返回 (auc, error_msg)。
    """
    m = run_dir / "shadow_pair_metrics"
    pair_path = m / "pair_metrics.csv"
    pair_control_path = m / "pair_metrics_control.csv"
    if not pair_path.exists() or not pair_control_path.exists():
        return None, "missing pair CSV"
    df_m = pd.read_csv(pair_path)
    df_c = pd.read_csv(pair_control_path)
    if df_m.empty or df_c.empty:
        return None, "empty CSV"
    # 共有数值列（排除 target_idx, round 等 ID）
    numeric_m = set(df_m.select_dtypes(include=[np.number]).columns)
    numeric_c = set(df_c.select_dtypes(include=[np.number]).columns)
    common = numeric_m & numeric_c
    skip = {"target_idx", "round"}
    feature_cols = [c for c in FEATURE_CANDIDATES if c in common] or [c for c in sorted(common) if c not in skip]
    if not feature_cols:
        return None, "no common numeric features"
    X_m = df_m[feature_cols].replace([np.inf, -np.inf], np.nan).dropna(how="all")
    X_c = df_c[feature_cols].replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if X_m.empty or X_c.empty:
        return None, "no valid rows after dropna"
    # 对齐列（可能某列全 nan 被 dropna 掉）
    for c in feature_cols:
        if c not in X_m.columns:
            X_m[c] = np.nan
        if c not in X_c.columns:
            X_c[c] = np.nan
    X_m = X_m[feature_cols].fillna(0)
    X_c = X_c[feature_cols].fillna(0)
    X = pd.concat([X_m, X_c], axis=0, ignore_index=True)
    y = np.concatenate([np.ones(len(X_m)), np.zeros(len(X_c))])
    if np.unique(y).size < 2 or len(y) < cv:
        return None, "single class or too few samples"
    clf = LogisticRegression(max_iter=1000, random_state=42)
    res = cross_validate(clf, X, y, cv=cv, scoring="roc_auc", return_train_score=False)
    aucs = res["test_score"]
    return float(np.mean(aucs)), None


def collect_learned_auc_train500_iter1000(
    summary_path: Optional[Path],
    outputs_dir: Optional[Path],
) -> pd.DataFrame:
    """
    收集 train_rows=500, n_iter=1000 的每个 run 的 learned AUC。
    若提供 summary_path 则从表里取 run_dir_path；否则扫描 outputs_dir 下 train500_synth500_ctgan+control。
    """
    rows: List[Dict[str, Any]] = []
    if summary_path and summary_path.exists():
        df = pd.read_csv(summary_path)
        df = df[(df["train_rows"] == 500) & (df["n_iter"] == 1000)]
        for _, r in df.iterrows():
            run_dir = Path(r["run_dir_path"])
            if not run_dir.is_dir():
                continue
            auc_val, err = compute_learned_auc_one_run(run_dir)
            if err:
                continue
            rounds = r.get("rounds")
            if pd.isna(rounds):
                rounds = _get_rounds_from_config(run_dir)
            rows.append({"rounds": rounds, "seed": r.get("seed"), "learned_auc": auc_val, "run_dir_name": r.get("run_dir_name", "")})
    elif outputs_dir and outputs_dir.is_dir():
        for d in outputs_dir.iterdir():
            if not d.is_dir():
                continue
            name = d.name
            if "train500_synth500_ctgan" not in name or "control" not in name:
                continue
            parsed = _parse_dirname(name)
            if parsed.get("n_iter") != 1000:
                continue
            if not (d / "shadow_pair_metrics" / "pair_metrics.csv").exists() or not (d / "shadow_pair_metrics" / "pair_metrics_control.csv").exists():
                continue
            rounds = parsed.get("rounds") or _get_rounds_from_config(d)
            auc_val, err = compute_learned_auc_one_run(d)
            if err:
                continue
            rows.append({"rounds": rounds, "seed": parsed.get("seed"), "learned_auc": auc_val, "run_dir_name": name})
    return pd.DataFrame(rows)


def plot_learned_auc_vs_rounds(df: pd.DataFrame, out_path: Path) -> None:
    """画 learned AUC vs rounds：train=500, iter=1000；每个 rounds 散点 + mean ± std。"""
    if df.empty or "rounds" not in df.columns or "learned_auc" not in df.columns:
        print("无 learned AUC 数据，跳过画图")
        return
    df = df.dropna(subset=["rounds", "learned_auc"])
    df["rounds"] = df["rounds"].astype(int)
    rounds_vals = sorted(df["rounds"].unique())
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, r in enumerate(rounds_vals):
        sub = df[df["rounds"] == r]
        x = np.full(len(sub), r) + np.random.uniform(-0.15, 0.15, len(sub))
        ax.scatter(x, sub["learned_auc"], alpha=0.6, s=40, color="C0", edgecolors="k", linewidths=0.5)
        mean_y = sub["learned_auc"].mean()
        std_y = sub["learned_auc"].std()
        std_y = std_y if pd.notna(std_y) and std_y > 0 else 0
        ax.errorbar(r, mean_y, yerr=std_y, fmt="o", color="C1", markersize=10, capsize=5, capthick=2, label=None if i else "mean ± std")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("rounds_per_target")
    ax.set_ylabel("Learned Attacker AUC (5-fold CV)")
    ax.set_title("Train=500, iter=1000: Learned Attacker AUC vs rounds")
    ax.set_xticks(rounds_vals)
    if rounds_vals:
        ax.legend(loc="best")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Learned Attacker AUC vs rounds (train500, iter=1000)")
    parser.add_argument("--summary", type=Path, default=None, help="adult_train500_full_summary.csv（可选）")
    parser.add_argument("--outputs", type=Path, default=PROJECT_ROOT / "outputs", help="无 summary 时扫描的 outputs 目录")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "figures" / "learned_attacker_auc_vs_rounds.png")
    args = parser.parse_args()
    df = collect_learned_auc_train500_iter1000(args.summary, args.outputs)
    if df.empty:
        print("未得到任何 learned AUC 数据（请先跑 train500_ablations.py build-summary 或确保 outputs 下有 train500_synth500_ctgan+control 且 n_iter=1000 的 run）")
        return 1
    print(f"共 {len(df)} 个 run 的 learned AUC")
    plot_learned_auc_vs_rounds(df, args.out)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
