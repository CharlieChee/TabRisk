#!/usr/bin/env python3
"""
桥接表与图：攻击优势 adv_abs 与稳定性代理 s_run 的正相关验证。

主指标：adv_abs = |AUC − 0.5|（等价 AUC_opt = max(AUC, 1-AUC)，adv_abs = AUC_opt − 0.5）。
所有相关性、回归、图均以 adv_abs 为 y 轴。

核心逻辑：相邻数据集（只差 1 条 target）训练出的生成分布差异越大（越不稳定）
→ LOO-MIA 攻击优势越大。即 adv_abs 与 in/out 分布差异正相关。

稳定性定义：replace-one stability（in/out 输出分布接近）。s_run 越大 = 差异越大 = 越不稳定。

交付物：
- run_level_bridge_table.csv：每 run 一行（含 adv_auc, adv_abs, s_run, r_run, paired_d 等）
- Fig1' adv_abs vs s_run；Fig4' adv_abs vs sqrt(s_run)
- Fig1-ctgan-only / Fig1-ddpm-only（去混杂）
- ddpm_within_N_spearman.csv：DDPM 各 N 的 Within-N Spearman ρ
- DDPM N=1000 sanity：pair-level MMD 分布图 + top-10 outlier CSV

过滤规则：
- 仅 control_branch=true 的 run（目录名含 control）
- DDPM: train_size in [200,500,1000,1500]，每 seed 1 条，共 20 条（4×5）
- CTGAN: train_size in [200,500,1000,1500,2000]，rounds=20，5 seeds；同配置同 seed 取时间戳较新的一条，共 25 条

用法:
  python scripts/build_run_level_bridge.py --outputs outputs --outdir outputs/figures --table outputs/run_level_bridge_table.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 数据集过滤（与用户约定一致）
DDPM_TRAIN_SIZES = [200, 500, 1000, 1500]
CTGAN_TRAIN_SIZES = [200, 500, 1000, 1500, 2000]
CTGAN_ROUNDS = 20
MAINLINE_SEEDS = [42, 43, 44, 45, 46]


def _is_ddpm_run(dirname: str) -> bool:
    n = dirname.lower()
    return "ddpm" in n or "tabddpm" in n


def _is_ctgan_run(dirname: str) -> bool:
    return "ctgan" in dirname.lower()


def parse_run_dir(dirname: str) -> Dict[str, Any]:
    """从 run 目录名解析：train_size, n_iter, batch_size, rounds, candidate, seed, timestamp, model."""
    out = {
        "train_size": None,
        "n_iter": None,
        "batch_size": None,
        "rounds": None,
        "candidate": None,
        "seed": None,
        "timestamp": None,
        "model": None,
    }
    m = re.search(r"_train(\d+)_synth", dirname)
    if m:
        out["train_size"] = int(m.group(1))
    m = re.search(r"_(\d+)iter", dirname)
    if m:
        out["n_iter"] = int(m.group(1))
    m = re.search(r"_bs(\d+)(?:_|$)", dirname)
    if m:
        out["batch_size"] = int(m.group(1))
    m = re.search(r"_rounds(\d+)_", f"_{dirname}_")
    if m:
        out["rounds"] = int(m.group(1))
    m = re.search(r"_candidate(\d+)(?:_|$)", dirname)
    if m:
        out["candidate"] = int(m.group(1))
    m = re.search(r"_seed(\d+)(?:_|$)", dirname)
    if m:
        out["seed"] = int(m.group(1))
    m = re.search(r"_(\d{8}_\d{6})$", dirname)
    if m:
        out["timestamp"] = m.group(1)
    if _is_ddpm_run(dirname):
        out["model"] = "ddpm"
    elif _is_ctgan_run(dirname):
        out["model"] = "ctgan"
    return out


def load_json_safe(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _choose_delta_column(df: pd.DataFrame) -> Optional[str]:
    for c in ("delta_min_dist", "delta_min_dist_mean", "delta_min_dist_median"):
        if c in df.columns:
            return c
    delta_cols = [c for c in df.columns if "delta" in c.lower()]
    return delta_cols[0] if delta_cols else None


def _auc_from_scores_labels(scores: np.ndarray, labels: np.ndarray) -> float:
    if scores.size == 0 or labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return float("nan")


def compute_adv_auc(
    pair_path: Path,
    pair_control_path: Path,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[str]]:
    """member=pair_metrics, non-member=pair_metrics_control；返回 (adv_auc, ci_low, ci_high, error_msg)。"""
    if not pair_path.exists():
        return None, None, None, "missing pair_metrics.csv"
    if not pair_control_path.exists():
        return None, None, None, "missing pair_metrics_control.csv"
    df_m = pd.read_csv(pair_path)
    df_c = pd.read_csv(pair_control_path)
    if df_m.empty or df_c.empty:
        return None, None, None, "empty pair or control CSV"
    delta_col = _choose_delta_column(df_m) or _choose_delta_column(df_c)
    if not delta_col or delta_col not in df_m.columns or delta_col not in df_c.columns:
        return None, None, None, "missing delta column"
    scores_m = df_m[delta_col].astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    scores_c = df_c[delta_col].astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if scores_m.size == 0 or scores_c.size == 0:
        return None, None, None, "no valid delta scores"
    scores = np.concatenate([scores_m, scores_c])
    labels = np.concatenate([np.ones(len(scores_m)), np.zeros(len(scores_c))])
    n = len(labels)
    auc = _auc_from_scores_labels(scores, labels)
    if np.isnan(auc):
        return None, None, None, "AUC computation failed"
    adv = auc - 0.5
    rng = np.random.default_rng(seed)
    boot_adv = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        a = _auc_from_scores_labels(scores[idx], labels[idx])
        if not np.isnan(a):
            boot_adv.append(a - 0.5)
    boot_arr = np.array(boot_adv)
    if boot_arr.size == 0:
        ci_low, ci_high = adv, adv
    else:
        ci_low = float(np.percentile(boot_arr, 2.5))
        ci_high = float(np.percentile(boot_arr, 97.5))
    return adv, ci_low, ci_high, None


def stability_from_pair_metrics(pair_path: Path) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    从 pair_metrics.csv 计算 run 级稳定性代理。
    返回 (s_run, s_run_std, r_run, r_run_std)。
    s_run = mean(mmd_mixed) 分布级；r_run = mean(delta_min_dist) record-level。
    """
    if not pair_path.exists():
        return None, None, None, None
    df = pd.read_csv(pair_path)
    if df.empty:
        return None, None, None, None
    mmd_col = "mmd_mixed" if "mmd_mixed" in df.columns else "mmd_numeric"
    delta_col = _choose_delta_column(df)
    s_run = float(df[mmd_col].mean()) if mmd_col in df.columns else None
    s_run_std = float(df[mmd_col].std()) if mmd_col in df.columns and len(df) > 1 else None
    r_run = float(df[delta_col].mean()) if delta_col and delta_col in df.columns else None
    r_run_std = float(df[delta_col].std()) if delta_col and delta_col in df.columns and len(df) > 1 else None
    return s_run, s_run_std, r_run, r_run_std


def paired_d_from_control_comparison(cc_path: Path) -> Optional[float]:
    """从 control_comparison.json 取 cohens_d_paired（展平后可能为 cc_cohens_d_paired 或嵌套）。"""
    if not cc_path.exists():
        return None
    data = load_json_safe(cc_path)
    if not data:
        return None
    if "cohens_d_paired" in data:
        return float(data["cohens_d_paired"])
    if "t_test_paired" in data and isinstance(data["t_test_paired"], dict):
        # 无 cohens_d 时返回 None，主表仍可填 paired_d 为 nan
        return None
    for k, v in data.items():
        if "cohens" in k.lower() and "paired" in k.lower() and isinstance(v, (int, float)):
            return float(v)
    return None


def _flatten_cc(cc: Dict[str, Any], prefix: str = "cc_") -> Dict[str, Any]:
    out = {}
    for k, v in cc.items():
        if isinstance(v, dict) and v and not any(isinstance(x, (dict, list)) for x in v.values()):
            for k2, v2 in v.items():
                out[f"{prefix}{k}_{k2}"] = v2
        elif not isinstance(v, (dict, list)):
            out[f"{prefix}{k}"] = v
    return out


def collect_run_row(run_dir: Path, metrics_dir: Path) -> Optional[Dict[str, Any]]:
    """单 run 收集一行：解析目录名 + pair_metrics 稳定性 + AUC + control_comparison。"""
    parsed = parse_run_dir(run_dir.name)
    if not parsed.get("model"):
        return None
    pair_path = metrics_dir / "pair_metrics.csv"
    pair_control_path = metrics_dir / "pair_metrics_control.csv"
    cc_path = metrics_dir / "control_comparison.json"
    gds_path = metrics_dir / "global_delta_stats.json"

    s_run, s_run_std, r_run, r_run_std = stability_from_pair_metrics(pair_path)
    adv_auc, adv_ci_low, adv_ci_high, auc_err = compute_adv_auc(pair_path, pair_control_path)
    # adv_abs = |AUC - 0.5| = AUC_opt - 0.5, 主指标
    adv_abs = abs(adv_auc) if adv_auc is not None else np.nan
    paired_d = paired_d_from_control_comparison(cc_path)
    if paired_d is None and cc_path.exists():
        cc = load_json_safe(cc_path)
        if cc:
            flat = _flatten_cc(cc)
            paired_d = flat.get("cc_cohens_d_paired")
            if paired_d is not None and isinstance(paired_d, (int, float)) and not (isinstance(paired_d, float) and np.isnan(paired_d)):
                paired_d = float(paired_d)
            else:
                paired_d = None

    row = {
        "model": parsed["model"],
        "train_size": parsed["train_size"],
        "n_iter": parsed["n_iter"],
        "rounds": parsed["rounds"],
        "batch_size": parsed["batch_size"],
        "seed": parsed["seed"],
        "timestamp": parsed["timestamp"],
        "run_dir_name": run_dir.name,
        "run_dir_path": str(run_dir.resolve()),
        "adv_auc": adv_auc if adv_auc is not None else np.nan,
        "adv_abs": adv_abs,
        "adv_auc_ci95_low": adv_ci_low if adv_ci_low is not None else np.nan,
        "adv_auc_ci95_high": adv_ci_high if adv_ci_high is not None else np.nan,
        "s_run": s_run if s_run is not None else np.nan,
        "s_run_std": s_run_std if s_run_std is not None else np.nan,
        "r_run": r_run if r_run is not None else np.nan,
        "r_run_std": r_run_std if r_run_std is not None else np.nan,
        "paired_d": paired_d if paired_d is not None else np.nan,
    }
    return row


def scan_control_runs(outputs_dir: Path) -> List[Path]:
    """扫描 outputs 下名称含 control 且含 shadow_pair_metrics 的 run 目录（仅一层子目录）。"""
    outputs_dir = outputs_dir.resolve()
    if not outputs_dir.is_dir():
        return []
    run_dirs = []
    for d in outputs_dir.iterdir():
        if not d.is_dir():
            continue
        if "control" not in d.name:
            continue
        metrics_dir = d / "shadow_pair_metrics"
        if not metrics_dir.is_dir() or not (metrics_dir / "pair_metrics.csv").exists():
            continue
        run_dirs.append(d)
    # 也扫描一层子目录（如 outputs/postprocess_adult_ctgan/ 下的 run 名）
    for sub in outputs_dir.iterdir():
        if not sub.is_dir() or sub.name.startswith("."):
            continue
        for d in sub.iterdir():
            if not d.is_dir():
                continue
            if "control" not in d.name:
                continue
            metrics_dir = d / "shadow_pair_metrics"
            if not metrics_dir.is_dir() or not (metrics_dir / "pair_metrics.csv").exists():
                continue
            run_dirs.append(d)
    return sorted(set(run_dirs), key=lambda p: (p.name, str(p)))


def filter_and_dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """
    DDPM: train_size in [200,500,1000,1500], seed in MAINLINE_SEEDS → 20 条。
    CTGAN: train_size in [200,500,1000,1500,2000], rounds=20, seed in MAINLINE_SEEDS；
          同 (model, train_size, rounds, n_iter, batch_size, seed) 取 timestamp 最大的一条 → 25 条。
    """
    rows = []
    # DDPM
    ddpm = df[df["model"] == "ddpm"].copy()
    ddpm = ddpm[ddpm["train_size"].isin(DDPM_TRAIN_SIZES) & ddpm["seed"].isin(MAINLINE_SEEDS)]
    if not ddpm.empty:
        ddpm = ddpm.sort_values("timestamp", ascending=True, na_position="last")
        ddpm = ddpm.drop_duplicates(subset=["train_size", "seed"], keep="last")
        rows.append(ddpm)
    # CTGAN
    ctgan = df[df["model"] == "ctgan"].copy()
    ctgan = ctgan[
        ctgan["train_size"].isin(CTGAN_TRAIN_SIZES)
        & (ctgan["rounds"] == CTGAN_ROUNDS)
        & ctgan["seed"].isin(MAINLINE_SEEDS)
    ]
    if not ctgan.empty:
        ctgan = ctgan.sort_values("timestamp", ascending=True, na_position="last")
        ctgan = ctgan.drop_duplicates(
            subset=["train_size", "seed"],
            keep="last",
        )
        rows.append(ctgan)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["model", "train_size", "seed"]).reset_index(drop=True)


def _scatter_regression(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    ax: plt.Axes,
    color_by_model: bool = True,
    show_fit: bool = True,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """散点 + 按 model 标色；可选线性拟合。返回 (r_pearson, p_pearson, rho_spearman, p_spearman)。"""
    df = df.dropna(subset=[x_col, y_col])
    if df.empty or len(df) < 2:
        return None, None, None, None
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    if color_by_model and "model" in df.columns:
        models = sorted(df["model"].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(models), 1)))
        for i, mod in enumerate(models):
            sub = df[df["model"] == mod]
            ax.scatter(sub[x_col], sub[y_col], c=[colors[i]], s=50, alpha=0.8, label=mod, edgecolors="k", linewidths=0.5)
    else:
        ax.scatter(df[x_col], df[y_col], s=50, alpha=0.8, edgecolors="k", linewidths=0.5)
    r_pearson, p_pearson, rho, p_spearman = None, None, None, None
    if x.size > 2 and np.isfinite(x).all() and np.isfinite(y).all() and show_fit:
        slope, intercept, r_pearson, p_pearson, _ = stats.linregress(x, y)
        rho, p_spearman = stats.spearmanr(x, y, nan_policy="omit")
        if not np.isfinite(rho):
            rho, p_spearman = np.nan, np.nan
        xx = np.linspace(x.min(), x.max(), 100)
        ax.plot(xx, slope * xx + intercept, "k-", linewidth=1.5, label="linear fit")
    if color_by_model and "model" in df.columns:
        ax.legend()
    return r_pearson, p_pearson, rho, p_spearman


def fig1_prime_adv_abs_vs_s_run(df: pd.DataFrame, out_path: Path) -> None:
    """Fig1'：adv_abs vs s_run，散点 + 回归 + Pearson/Spearman，按 ctgan/ddpm 标色。"""
    df = df.dropna(subset=["s_run", "adv_abs"]).copy()
    if df.empty or len(df) < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    r_pearson, p_pearson, rho, p_spearman = _scatter_regression(df, "s_run", "adv_abs", ax, color_by_model=True, show_fit=True)
    title_r = f"Pearson r = {r_pearson:.3f} (p = {p_pearson:.4f})" if r_pearson is not None else ""
    title_s = f"Spearman ρ = {rho:.3f} (p = {p_spearman:.4f})" if rho is not None and np.isfinite(rho) else "Spearman ρ = N/A"
    ax.set_title(f"adv_abs vs s_run (MMD)\n{title_r}, {title_s}")
    ax.set_xlabel("s_run (mean MMD in/out)")
    ax.set_ylabel("adv_abs = |AUC − 0.5|")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig4_prime_adv_abs_vs_sqrt_s_run(df: pd.DataFrame, out_path: Path) -> None:
    """Fig4'：adv_abs vs sqrt(s_run)，散点 + 回归 + Pearson/Spearman，按 model 标色。"""
    df = df.dropna(subset=["s_run", "adv_abs"]).copy()
    if df.empty or (df["s_run"] <= 0).all():
        return
    df = df[df["s_run"] > 0].copy()
    if df.empty:
        return
    df = df.assign(sqrt_s=np.sqrt(df["s_run"]))
    fig, ax = plt.subplots(figsize=(7, 5))
    r_pearson, p_pearson, rho, p_spearman = _scatter_regression(df, "sqrt_s", "adv_abs", ax, color_by_model=True, show_fit=True)
    title_r = f"Pearson r = {r_pearson:.3f} (p = {p_pearson:.4f})" if r_pearson is not None else ""
    title_s = f"Spearman ρ = {rho:.3f} (p = {p_spearman:.4f})" if rho is not None and np.isfinite(rho) else "Spearman ρ = N/A"
    ax.set_title(f"adv_abs vs sqrt(s_run)\n{title_r}, {title_s}")
    ax.set_xlabel("sqrt(s_run)")
    ax.set_ylabel("adv_abs = |AUC − 0.5|")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig1_ctgan_only(df: pd.DataFrame, out_path: Path) -> None:
    """Fig1-ctgan-only：仅 CTGAN，adv_abs vs s_run。"""
    df = df[(df["model"] == "ctgan")].dropna(subset=["s_run", "adv_abs"]).copy()
    if df.empty or len(df) < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    r_pearson, p_pearson, rho, p_spearman = _scatter_regression(df, "s_run", "adv_abs", ax, color_by_model=False, show_fit=True)
    title_r = f"Pearson r = {r_pearson:.3f} (p = {p_pearson:.4f})" if r_pearson is not None else ""
    title_s = f"Spearman ρ = {rho:.3f} (p = {p_spearman:.4f})" if rho is not None and np.isfinite(rho) else "Spearman ρ = N/A"
    ax.set_title(f"adv_abs vs s_run (CTGAN only)\n{title_r}, {title_s}")
    ax.set_xlabel("s_run")
    ax.set_ylabel("adv_abs")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig1_ddpm_only(df: pd.DataFrame, out_path: Path) -> None:
    """Fig1-ddpm-only：仅 DDPM，adv_abs vs s_run。"""
    df = df[(df["model"] == "ddpm")].dropna(subset=["s_run", "adv_abs"]).copy()
    if df.empty or len(df) < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    r_pearson, p_pearson, rho, p_spearman = _scatter_regression(df, "s_run", "adv_abs", ax, color_by_model=False, show_fit=True)
    title_r = f"Pearson r = {r_pearson:.3f} (p = {p_pearson:.4f})" if r_pearson is not None else ""
    title_s = f"Spearman ρ = {rho:.3f} (p = {p_spearman:.4f})" if rho is not None and np.isfinite(rho) else "Spearman ρ = N/A"
    ax.set_title(f"adv_abs vs s_run (DDPM only)\n{title_r}, {title_s}")
    ax.set_xlabel("s_run")
    ax.set_ylabel("adv_abs")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig2_stability_vs_train_size(df: pd.DataFrame, out_path: Path) -> None:
    """图2：train_size N vs s_run，每 N 多 seed 散点 + 均值/误差条。"""
    df = df.dropna(subset=["train_size", "s_run"]).copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model]
        agg = sub.groupby("train_size", as_index=False).agg(
            mean_s=( "s_run", "mean"),
            std_s=("s_run", "std"),
            count=("s_run", "count"),
        )
        x = agg["train_size"].values
        y = agg["mean_s"].values
        err = agg["std_s"].fillna(0).values
        has_err = agg["count"].values > 1
        ax.errorbar(x, y, yerr=np.where(has_err, err, 0), marker="o", capsize=4, label=model)
        ax.scatter(sub["train_size"], sub["s_run"], alpha=0.5, s=30)
    ax.set_xlabel("Train size N")
    ax.set_ylabel("s_run (stability proxy)")
    ax.set_title("Stability scaling: N vs s_run (mean ± std over seeds)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig3_adv_abs_vs_train_size(df: pd.DataFrame, out_path: Path) -> None:
    """图3：train_size N vs adv_abs，与图2 同一套 run。"""
    df = df.dropna(subset=["train_size", "adv_abs"]).copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model]
        agg = sub.groupby("train_size", as_index=False).agg(
            mean_adv=("adv_abs", "mean"),
            std_adv=("adv_abs", "std"),
            count=("adv_abs", "count"),
        )
        x = agg["train_size"].values
        y = agg["mean_adv"].values
        err = agg["std_adv"].fillna(0).values
        has_err = agg["count"].values > 1
        ax.errorbar(x, y, yerr=np.where(has_err, err, 0), marker="o", capsize=4, label=model)
        ax.scatter(sub["train_size"], sub["adv_abs"], alpha=0.5, s=30)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Train size N")
    ax.set_ylabel("adv_abs = |AUC − 0.5|")
    ax.set_title("adv_abs vs Train size N (same runs as Fig 2)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig4_stability_vs_iter(df: pd.DataFrame, out_path: Path) -> None:
    """图4（可选）：n_iter 或 rounds vs s_run，按 N 分线。"""
    df = df.dropna(subset=["s_run"]).copy()
    if df.empty:
        return
    # 用 rounds 若存在且多值，否则 n_iter
    if "rounds" in df.columns and df["rounds"].nunique() > 1:
        x_col = "rounds"
    else:
        x_col = "n_iter"
    if df[x_col].isna().all():
        return
    df = df.dropna(subset=[x_col, "s_run"])
    if df.empty or df[x_col].nunique() < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for ts in sorted(df["train_size"].dropna().unique()):
        sub = df[df["train_size"] == ts]
        agg = sub.groupby(x_col, as_index=False).agg(mean_s=("s_run", "mean"), std_s=("s_run", "std"))
        ax.errorbar(agg[x_col], agg["mean_s"], yerr=agg["std_s"].fillna(0), marker="o", capsize=4, label=f"N={int(ts)}")
    ax.set_xlabel(x_col)
    ax.set_ylabel("s_run")
    ax.set_title("Stability vs training intensity (iter/rounds) by N")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig5_adv_abs_vs_sqrt_stability(df: pd.DataFrame, out_path: Path) -> None:
    """图5（可选）：adv_abs vs sqrt(s_run)，与 Fig4' 一致，保留兼容。"""
    fig4_prime_adv_abs_vs_sqrt_s_run(df, out_path)


def write_ddpm_within_N_spearman_table(df: pd.DataFrame, out_path: Path) -> None:
    """DDPM 每个 N 单独算 adv_abs vs s_run 的 Spearman ρ，输出表。"""
    ddpm = df[(df["model"] == "ddpm")].dropna(subset=["train_size", "adv_abs", "s_run"])
    if ddpm.empty:
        return
    rows = []
    for n in sorted(ddpm["train_size"].unique()):
        sub = ddpm[ddpm["train_size"] == n]
        if len(sub) < 2:
            rows.append({"train_size": int(n), "n_runs": len(sub), "spearman_rho": np.nan, "spearman_p": np.nan})
            continue
        x = sub["s_run"].to_numpy(dtype=float)
        y = sub["adv_abs"].to_numpy(dtype=float)
        rho, p = stats.spearmanr(x, y, nan_policy="omit")
        if not np.isfinite(rho):
            rho, p = np.nan, np.nan
        rows.append({"train_size": int(n), "n_runs": len(sub), "spearman_rho": float(rho), "spearman_p": float(p)})
    out_df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"Within-N Spearman 表: {out_path}")


def run_ddpm_N1000_sanity(bridge_df: pd.DataFrame, outdir: Path) -> None:
    """DDPM N=1000：pair-level MMD 分布（箱线+直方）+ top-10 outlier pairs (run, target_idx, round_k, mmd)。"""
    ddpm_1000 = bridge_df[(bridge_df["model"] == "ddpm") & (bridge_df["train_size"] == 1000)]
    if ddpm_1000.empty:
        print("无 DDPM N=1000 run，跳过 sanity")
        return
    pair_rows: List[Dict[str, Any]] = []
    for _, row in ddpm_1000.iterrows():
        run_path = Path(row["run_dir_path"])
        pair_path = run_path / "shadow_pair_metrics" / "pair_metrics.csv"
        if not pair_path.exists():
            continue
        df = pd.read_csv(pair_path)
        mmd_col_use = "mmd_mixed" if "mmd_mixed" in df.columns else ("mmd_numeric" if "mmd_numeric" in df.columns else None)
        if mmd_col_use is None:
            continue
        for _, r in df.iterrows():
            pair_rows.append({
                "run": row["run_dir_name"],
                "target_idx": r.get("target_idx", np.nan),
                "round_k": r.get("round", np.nan),
                "mmd": float(r[mmd_col_use]),
            })
    if not pair_rows:
        print("DDPM N=1000 无 pair-level 数据，跳过 sanity")
        return
    pair_df = pd.DataFrame(pair_rows)
    pair_df = pair_df.dropna(subset=["mmd"])
    if pair_df.empty:
        return
    # 直方 + 箱线
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(pair_df["mmd"], bins=40, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("MMD (pair-level)")
    axes[0].set_ylabel("count")
    axes[0].set_title("DDPM N=1000 pair-level MMD distribution (hist)")
    axes[1].boxplot(pair_df["mmd"], vert=True)
    axes[1].set_ylabel("MMD")
    axes[1].set_title("DDPM N=1000 pair-level MMD (boxplot)")
    plt.tight_layout()
    fig_path = outdir / "ddpm_N1000_pair_mmd_distribution.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"DDPM N=1000 MMD 分布图: {fig_path}")
    # top-10 outliers by MMD (largest)
    top10 = pair_df.nlargest(10, "mmd")[["run", "target_idx", "round_k", "mmd"]].reset_index(drop=True)
    top10_path = outdir / "ddpm_N1000_top10_mmd_outliers.csv"
    top10.to_csv(top10_path, index=False, float_format="%.6f")
    print(f"DDPM N=1000 top-10 MMD outliers: {top10_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="构建 run_level_bridge_table.csv 并画 Adv vs 稳定性 5 张图",
    )
    parser.add_argument("--outputs", type=Path, default=PROJECT_ROOT / "outputs", help="outputs 根目录")
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "outputs" / "figures", help="图片输出目录")
    parser.add_argument("--table", type=Path, default=PROJECT_ROOT / "outputs" / "run_level_bridge_table.csv", help="桥接表 CSV 路径")
    parser.add_argument("--no-fig4", action="store_true", help="不画图4（iter/rounds vs s）")
    parser.add_argument("--no-fig5", action="store_true", help="不画图5（Adv vs sqrt(s)）")
    args = parser.parse_args()

    outputs_dir = args.outputs.resolve()
    run_dirs = scan_control_runs(outputs_dir)
    if not run_dirs:
        print(f"未在 {outputs_dir} 下找到任何带 control 且含 shadow_pair_metrics/pair_metrics.csv 的 run")
        return

    rows = []
    for run_dir in run_dirs:
        metrics_dir = run_dir / "shadow_pair_metrics"
        row = collect_run_row(run_dir, metrics_dir)
        if row:
            rows.append(row)
    raw_df = pd.DataFrame(rows)
    if raw_df.empty:
        print("没有成功收集到任何 run 行")
        return

    bridge_df = filter_and_dedupe(raw_df)
    if bridge_df.empty:
        print("过滤/去重后无 run（请检查 DDPM train_size in [200,500,1000,1500]、CTGAN rounds=20 等）")
        bridge_df = raw_df

    args.table.parent.mkdir(parents=True, exist_ok=True)
    bridge_df.to_csv(args.table, index=False, encoding="utf-8-sig", float_format="%.6f")
    print(f"桥接表已写入: {args.table} (rows={len(bridge_df)})")

    args.outdir.mkdir(parents=True, exist_ok=True)
    # 核心图：adv_abs 为主指标
    fig1_prime_path = args.outdir / "fig1_prime_adv_abs_vs_s_run.png"
    fig4_prime_path = args.outdir / "fig4_prime_adv_abs_vs_sqrt_s_run.png"
    fig1_ctgan_path = args.outdir / "fig1_ctgan_only_adv_abs_vs_s_run.png"
    fig1_ddpm_path = args.outdir / "fig1_ddpm_only_adv_abs_vs_s_run.png"
    fig2_path = args.outdir / "fig_stability_vs_train_size.png"
    fig3_path = args.outdir / "fig_adv_abs_vs_train_size.png"
    fig4_iter_path = args.outdir / "fig_stability_vs_iter.png"
    fig5_path = args.outdir / "fig_adv_abs_vs_sqrt_stability.png"

    fig1_prime_adv_abs_vs_s_run(bridge_df, fig1_prime_path)
    fig4_prime_adv_abs_vs_sqrt_s_run(bridge_df, fig4_prime_path)
    fig1_ctgan_only(bridge_df, fig1_ctgan_path)
    fig1_ddpm_only(bridge_df, fig1_ddpm_path)
    fig2_stability_vs_train_size(bridge_df, fig2_path)
    fig3_adv_abs_vs_train_size(bridge_df, fig3_path)
    print(f"Fig1': {fig1_prime_path}")
    print(f"Fig4': {fig4_prime_path}")
    print(f"Fig1-ctgan-only: {fig1_ctgan_path}")
    print(f"Fig1-ddpm-only: {fig1_ddpm_path}")
    print(f"图2: {fig2_path}")
    print(f"图3: {fig3_path}")

    # Within-N Spearman 表（DDPM）
    within_n_path = args.outdir / "ddpm_within_N_spearman.csv"
    write_ddpm_within_N_spearman_table(bridge_df, within_n_path)

    # DDPM N=1000 sanity：pair-level MMD 分布 + top-10 outliers
    run_ddpm_N1000_sanity(bridge_df, args.outdir)

    if not args.no_fig4:
        fig4_stability_vs_iter(bridge_df, fig4_iter_path)
        print(f"图4(iter): {fig4_iter_path}")
    if not args.no_fig5:
        fig5_adv_abs_vs_sqrt_stability(bridge_df, fig5_path)
        print(f"图5: {fig5_path}")


if __name__ == "__main__":
    main()
