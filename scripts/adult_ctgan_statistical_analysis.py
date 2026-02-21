#!/usr/bin/env python3
"""
Adult+CTGAN LOO-MIA 统计分析：锁定数据集、显著性检验、效应量、多重校正、偏相关与稳健回归、learned attacker 检验。

指令 0：读 adult_ctgan_run_summary.csv，去重 + 平衡，输出 scaling_balanced / rounds_balanced / iter_balanced + dropped_runs.csv
指令 1：Scaling / Saturation / Iter 的 Welch + MWU + Hedges' g + Holm；补充图（CI + 显著性）
指令 2：Partial correlation（控制 train_size）+ 稳健回归；residual 图
指令 3：Learned attacker vs 0.5 + 组间检验
指令 4/5：产出审稿人友好的表与图，记录 dropped_runs

用法:
  python scripts/adult_ctgan_statistical_analysis.py --summary outputs/postprocess_adult_ctgan/adult_ctgan_run_summary.csv --outdir outputs/postprocess_adult_ctgan/stats
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
from scipy import stats
from sklearn.linear_model import HuberRegressor, LinearRegression, LogisticRegression, TheilSenRegressor
from sklearn.model_selection import cross_validate

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Learned attacker 特征列（与 postprocess 一致）
FEATURE_CANDIDATES = [
    "mmd_numeric", "mmd_mixed", "min_dist_in", "min_dist_out", "delta_min_dist",
    "num_mean_diff_mean", "num_mean_diff_max", "num_std_diff_mean", "num_std_diff_max",
    "cat_tv_mean", "cat_tv_max",
]

# 与 postprocess 一致的组定义
MAINLINE_SEEDS = [42, 43, 44, 45, 46]
SCALING_N_ITER = 1000
SCALING_ROUNDS = 20
SCALING_TRAIN_SIZES = [200, 500, 1000, 1500, 2000]
SATURATION_TRAIN = 500
SATURATION_N_ITER = 1000
SATURATION_ROUNDS = [5, 10, 20, 40, 80, 160]
ITER_TRAIN = 500
ITER_ROUNDS = 20
ITER_N_ITERS = [200, 300, 600, 1000, 2000, 4000]

# Reference levels for tests
REF_TRAIN_SCALING = 200   # strongest leakage
REF_ROUNDS = 20
REF_N_ITER = 1000

N_BOOT = 2000
RNG = np.random.default_rng(42)


def _parse_timestamp_from_run_dir(run_dir: str) -> str:
    """从 run_dir 名末尾解析时间戳 _YYYYMMDD_HHMMSS，无则返回空串。"""
    m = re.search(r"_(\d{8}_\d{6})$", run_dir)
    return m.group(1) if m else ""


def _standardize_summary(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名：train_rows -> train_size, rounds -> rounds_per_target；保证 int 类型。"""
    out = df.copy()
    if "train_rows" in out.columns and "train_size" not in out.columns:
        out["train_size"] = out["train_rows"].astype("Int64")
    if "rounds" in out.columns and "rounds_per_target" not in out.columns:
        out["rounds_per_target"] = out["rounds"].astype("Int64")
    for c in ["train_size", "rounds_per_target", "n_iter", "batch_size", "seed"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _check_completeness(run_dir_path: str, candidate: Optional[int], rounds: Optional[int]) -> bool:
    """可选：检查 run 是否完整（pair_metrics 存在且行数合理）。"""
    p = Path(run_dir_path)
    pm = p / "shadow_pair_metrics" / "pair_metrics.csv"
    if not pm.exists():
        return False
    try:
        n_rows = sum(1 for _ in open(pm, "rb")) - 1  # header
    except Exception:
        return False
    if candidate is not None and rounds is not None and rounds > 0:
        expected = candidate * rounds
        return n_rows >= expected * 0.95  # 允许少量缺失
    return n_rows > 0


def lock_dataset(
    summary_path: Path,
    verify_completeness: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    指令 0：读入总表，标准化字段，去重（同 key 保留时间戳最新），记录被丢弃的 run。
    返回 (deduped_df, dropped_df)。deduped_df 含 train_size, rounds_per_target, n_iter, batch_size, seed, run_dir, run_dir_path, adv_auc, ...
    """
    df = pd.read_csv(summary_path)
    df = _standardize_summary(df)
    if "run_dir" not in df.columns:
        df["run_dir"] = df.get("run_dir_path", "").apply(lambda x: Path(x).name if x else "")
    df["_timestamp"] = df["run_dir"].astype(str).apply(_parse_timestamp_from_run_dir)

    key_cols = ["train_size", "rounds_per_target", "n_iter", "batch_size", "seed"]
    for c in key_cols:
        if c not in df.columns:
            df[c] = np.nan

    # 可选：过滤不完整
    if verify_completeness and "run_dir_path" in df.columns:
        complete = []
        for _, row in df.iterrows():
            ok = _check_completeness(
                str(row.get("run_dir_path", "")),
                row.get("candidate"),
                row.get("rounds_per_target") or row.get("rounds"),
            )
            complete.append(ok)
        df["_complete"] = complete
        dropped_incomplete = df[~df["_complete"]].copy()
        dropped_incomplete["drop_reason"] = "incomplete"
        df = df[df["_complete"]].drop(columns=["_complete"])
    else:
        dropped_incomplete = pd.DataFrame()

    # 去重：同 key 保留时间戳最新
    df["_sort"] = df["_timestamp"].fillna("") + df["run_dir"].astype(str)
    kept = df.sort_values("_sort", ascending=False).groupby(key_cols, dropna=False).first().reset_index()
    merged = df.merge(
        kept[key_cols + ["run_dir"]],
        on=key_cols + ["run_dir"],
        how="left",
        indicator="_keep",
    )
    dropped_dup = df.loc[merged["_keep"] == "left_only"].copy()
    if not dropped_dup.empty:
        dropped_dup["drop_reason"] = "duplicate-older"

    deduped = kept.drop(columns=["_sort", "_timestamp"], errors="ignore")
    dropped = pd.concat([dropped_incomplete, dropped_dup], ignore_index=True)
    if not dropped.empty and "drop_reason" not in dropped.columns:
        dropped["drop_reason"] = "duplicate-older"
    return deduped, dropped


def balance_subsets(deduped: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[int], int, int]:
    """
    平衡：Scaling / Saturation / Iter 三组，每组使用相同 seed 集合（交集）。
    返回 (scaling_balanced, rounds_balanced, iter_balanced, seeds_a, smin_b, imin_c)。
    """
    # Scaling: n_iter=1000, rounds=20, train_size in [200,500,1000,1500,2000]
    a = deduped[
        (deduped["n_iter"] == SCALING_N_ITER)
        & (deduped["rounds_per_target"] == SCALING_ROUNDS)
        & (deduped["train_size"].isin(SCALING_TRAIN_SIZES))
    ].copy()
    a = a[a["seed"].isin(MAINLINE_SEEDS)]
    seed_counts_a = a.groupby("train_size")["seed"].nunique()
    seeds_a = MAINLINE_SEEDS[: int(seed_counts_a.min())] if len(seed_counts_a) else []
    a = a[a["seed"].isin(seeds_a)].drop_duplicates(subset=["train_size", "seed"], keep="first")
    scaling_balanced = a

    # Saturation: train=500, n_iter=1000, rounds in [5,10,20,40,80,160]
    b = deduped[
        (deduped["train_size"] == SATURATION_TRAIN)
        & (deduped["n_iter"] == SATURATION_N_ITER)
        & (deduped["rounds_per_target"].isin(SATURATION_ROUNDS))
    ].copy()
    smin_b = int(b.groupby("rounds_per_target")["seed"].nunique().min()) if not b.empty else 0
    keep_seeds_b = b.groupby("rounds_per_target")["seed"].apply(lambda x: sorted(x.unique())[:smin_b]).to_dict()
    rows_b = []
    for r, seeds in keep_seeds_b.items():
        sub = b[(b["rounds_per_target"] == r) & (b["seed"].isin(seeds))].drop_duplicates(subset=["seed"], keep="first")
        rows_b.append(sub)
    rounds_balanced = pd.concat(rows_b, ignore_index=True) if rows_b else pd.DataFrame()

    # Iter: train=500, rounds=20, n_iter in [200,...,4000]
    c = deduped[
        (deduped["train_size"] == ITER_TRAIN)
        & (deduped["rounds_per_target"] == ITER_ROUNDS)
        & (deduped["n_iter"].isin(ITER_N_ITERS))
    ].copy()
    imin_c = int(c.groupby("n_iter")["seed"].nunique().min()) if not c.empty else 0
    keep_seeds_c = c.groupby("n_iter")["seed"].apply(lambda x: sorted(x.unique())[:imin_c]).to_dict()
    rows_c = []
    for it, seeds in keep_seeds_c.items():
        sub = c[(c["n_iter"] == it) & (c["seed"].isin(seeds))].drop_duplicates(subset=["seed"], keep="first")
        rows_c.append(sub)
    iter_balanced = pd.concat(rows_c, ignore_index=True) if rows_c else pd.DataFrame()

    return scaling_balanced, rounds_balanced, iter_balanced, seeds_a, smin_b, imin_c


def _hedges_g_raw(x: np.ndarray, y: np.ndarray) -> float:
    """Hedges' g 单次计算（小样本修正），不递归。"""
    nx, ny = len(x), len(y)
    mx, my = x.mean(), y.mean()
    sx = x.std(ddof=1) if nx > 1 else 0.0
    sy = y.std(ddof=1) if ny > 1 else 0.0
    pooled = np.sqrt(((nx - 1) * sx * sx + (ny - 1) * sy * sy) / (nx + ny - 2)) if (nx + ny) > 2 else (sx + sy) / 2 or 1e-10
    g = (mx - my) / pooled if pooled > 0 else 0.0
    corr = (nx + ny - 2) / (nx + ny - 2.5) if (nx + ny) > 2.5 else 1.0
    return g * corr


def hedges_g(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Hedges' g（小样本修正）及 95% bootstrap CI。"""
    g = _hedges_g_raw(x, y)
    nx, ny = len(x), len(y)
    gs = []
    for _ in range(N_BOOT):
        bx = RNG.choice(x, size=nx, replace=True)
        by = RNG.choice(y, size=ny, replace=True)
        gs.append(_hedges_g_raw(bx, by))
    gs = np.array(gs)
    return g, float(np.percentile(gs, 2.5)), float(np.percentile(gs, 97.5))


def holm_correction(p_values: List[float]) -> List[float]:
    """Holm–Bonferroni 校正。"""
    p = np.array(p_values)
    order = np.argsort(p)
    n = len(p)
    p_holm = np.zeros(n)
    for i, idx in enumerate(order):
        p_holm[idx] = min(1.0, p[idx] * (n - i))
    return p_holm.tolist()


# ---------- 1A Scaling tests ----------
def run_scaling_tests(scaling_balanced: pd.DataFrame) -> pd.DataFrame:
    """Reference train_size=200；对比 500/1000/1500/2000；Welch + MWU + Hedges' g + Holm。"""
    df = scaling_balanced.dropna(subset=["train_size", "adv_auc"]).copy()
    if df.empty:
        return pd.DataFrame()
    ref = df[df["train_size"] == REF_TRAIN_SCALING]["adv_auc"].to_numpy()
    if ref.size == 0:
        return pd.DataFrame()
    others = [500, 1000, 1500, 2000]
    rows = []
    p_welch_list, p_mwu_list = [], []
    for tr in others:
        other = df[df["train_size"] == tr]["adv_auc"].to_numpy()
        if other.size == 0:
            continue
        t_stat, p_welch = stats.ttest_ind(ref, other, equal_var=False)
        u_stat, p_mwu = stats.mannwhitneyu(ref, other, alternative="two-sided")
        g, g_lo, g_hi = hedges_g(ref, other)
        rows.append({
            "compare": f"{REF_TRAIN_SCALING} vs {tr}",
            "mean_ref": float(ref.mean()),
            "mean_other": float(other.mean()),
            "mean_diff": float(other.mean() - ref.mean()),
            "t_stat": float(t_stat),
            "p_welch": float(p_welch),
            "p_mwu": float(p_mwu),
            "hedges_g": g,
            "hedges_g_ci95_low": g_lo,
            "hedges_g_ci95_high": g_hi,
            "n_ref": len(ref),
            "n_other": len(other),
        })
        p_welch_list.append(float(p_welch))
        p_mwu_list.append(float(p_mwu))
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm_welch"] = holm_correction(p_welch_list)
        out["p_holm_mwu"] = holm_correction(p_mwu_list)
    return out


# ---------- 1B Rounds tests ----------
def run_rounds_tests(rounds_balanced: pd.DataFrame) -> pd.DataFrame:
    """Reference rounds=20；对比 5/10/40/80/160。"""
    df = rounds_balanced.dropna(subset=["rounds_per_target", "adv_auc"]).copy()
    if "rounds" in df.columns and "rounds_per_target" not in df.columns:
        df["rounds_per_target"] = df["rounds"]
    if df.empty:
        return pd.DataFrame()
    ref = df[df["rounds_per_target"] == REF_ROUNDS]["adv_auc"].to_numpy()
    if ref.size == 0:
        return pd.DataFrame()
    others = [r for r in [5, 10, 40, 80, 160] if r in df["rounds_per_target"].values]
    rows = []
    p_welch_list, p_mwu_list = [], []
    for r in others:
        other = df[df["rounds_per_target"] == r]["adv_auc"].to_numpy()
        if other.size == 0:
            continue
        t_stat, p_welch = stats.ttest_ind(ref, other, equal_var=False)
        u_stat, p_mwu = stats.mannwhitneyu(ref, other, alternative="two-sided")
        g, g_lo, g_hi = hedges_g(ref, other)
        rows.append({
            "compare": f"{REF_ROUNDS} vs {r}",
            "mean_ref": float(ref.mean()),
            "mean_other": float(other.mean()),
            "mean_diff": float(other.mean() - ref.mean()),
            "t_stat": float(t_stat),
            "p_welch": float(p_welch),
            "p_mwu": float(p_mwu),
            "hedges_g": g,
            "hedges_g_ci95_low": g_lo,
            "hedges_g_ci95_high": g_hi,
            "n_ref": len(ref),
            "n_other": len(other),
        })
        p_welch_list.append(float(p_welch))
        p_mwu_list.append(float(p_mwu))
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm_welch"] = holm_correction(p_welch_list)
        out["p_holm_mwu"] = holm_correction(p_mwu_list)
    return out


# ---------- 1C Iter tests ----------
def run_iter_tests(iter_balanced: pd.DataFrame) -> pd.DataFrame:
    """Reference n_iter=1000；对比 200/300/600/2000/4000。"""
    df = iter_balanced.dropna(subset=["n_iter", "adv_auc"]).copy()
    if df.empty:
        return pd.DataFrame()
    ref = df[df["n_iter"] == REF_N_ITER]["adv_auc"].to_numpy()
    if ref.size == 0:
        return pd.DataFrame()
    others = [it for it in [200, 300, 600, 2000, 4000] if it in df["n_iter"].values]
    rows = []
    p_welch_list, p_mwu_list = [], []
    for it in others:
        other = df[df["n_iter"] == it]["adv_auc"].to_numpy()
        if other.size == 0:
            continue
        t_stat, p_welch = stats.ttest_ind(ref, other, equal_var=False)
        u_stat, p_mwu = stats.mannwhitneyu(ref, other, alternative="two-sided")
        g, g_lo, g_hi = hedges_g(ref, other)
        rows.append({
            "compare": f"{REF_N_ITER} vs {it}",
            "mean_ref": float(ref.mean()),
            "mean_other": float(other.mean()),
            "mean_diff": float(other.mean() - ref.mean()),
            "t_stat": float(t_stat),
            "p_welch": float(p_welch),
            "p_mwu": float(p_mwu),
            "hedges_g": g,
            "hedges_g_ci95_low": g_lo,
            "hedges_g_ci95_high": g_hi,
            "n_ref": len(ref),
            "n_other": len(other),
        })
        p_welch_list.append(float(p_welch))
        p_mwu_list.append(float(p_mwu))
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm_welch"] = holm_correction(p_welch_list)
        out["p_holm_mwu"] = holm_correction(p_mwu_list)
    return out


def _bootstrap_mean_ci(x: np.ndarray, n_boot: int = 2000) -> Tuple[float, float, float]:
    m = x.mean()
    boot = [RNG.choice(x, size=len(x), replace=True).mean() for _ in range(n_boot)]
    return m, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ---------- Figures ----------
def fig7_adv_scaling_ci_sig(scaling_balanced: pd.DataFrame, tests_df: pd.DataFrame, out_path: Path) -> None:
    """train_size vs adv_auc：均值 + 95% bootstrap CI + 显著性标注（vs 200）。"""
    df = scaling_balanced.dropna(subset=["train_size", "adv_auc"]).copy()
    if df.empty:
        return
    tr_vals = sorted(df["train_size"].unique())
    means, lo, hi, sigs = [], [], [], []
    ref_tr = REF_TRAIN_SCALING
    for tr in tr_vals:
        block = df[df["train_size"] == tr]["adv_auc"].to_numpy()
        m, l, h = _bootstrap_mean_ci(block)
        means.append(m)
        lo.append(l)
        hi.append(h)
        if tr == ref_tr:
            sigs.append("")
        else:
            row = tests_df[tests_df["compare"] == f"{ref_tr} vs {tr}"]
            p = row["p_holm_welch"].iloc[0] if not row.empty else 1.0
            sigs.append(_sig_stars(p))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(tr_vals, means, yerr=[np.array(means) - np.array(lo), np.array(hi) - np.array(means)], fmt="o-", capsize=5, capthick=2)
    for i, (t, s) in enumerate(zip(tr_vals, sigs)):
        if s:
            ax.annotate(s, (t, hi[i] + 0.01), ha="center", fontsize=12)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Train size")
    ax.set_ylabel("Adv(AUC)")
    ax.set_title("Adv(AUC) vs Train Size (95% bootstrap CI; * p<0.05, ** p<0.01, *** p<0.001 vs 200)")
    ax.set_xticks(tr_vals)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig8_adv_rounds_delta_from_20(rounds_balanced: pd.DataFrame, out_path: Path) -> None:
    """rounds vs adv_auc 均值 + 95% CI；以及相对增益 adv_auc(r) - adv_auc(20)。"""
    df = rounds_balanced.dropna(subset=["rounds_per_target", "adv_auc"]).copy()
    if "rounds" in df.columns and "rounds_per_target" not in df.columns:
        df["rounds_per_target"] = df["rounds"]
    if df.empty:
        return
    r_vals = sorted(df["rounds_per_target"].unique())
    means, lo, hi = [], [], []
    for r in r_vals:
        block = df[df["rounds_per_target"] == r]["adv_auc"].to_numpy()
        m, l, h = _bootstrap_mean_ci(block)
        means.append(m)
        lo.append(l)
        hi.append(h)
    ref_idx = r_vals.index(REF_ROUNDS) if REF_ROUNDS in r_vals else 0
    ref_mean = means[ref_idx]
    delta = [m - ref_mean for m in means]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    ax1.errorbar(r_vals, means, yerr=[np.array(means) - np.array(lo), np.array(hi) - np.array(means)], fmt="o-", capsize=5)
    ax1.axhline(0, color="gray", linestyle="--")
    ax1.set_xlabel("Shadow rounds per target")
    ax1.set_ylabel("Adv(AUC)")
    ax1.set_title("Adv(AUC) vs Rounds (95% bootstrap CI)")
    ax1.set_xticks(r_vals)
    ax1.grid(True, alpha=0.3)

    ax2.bar(r_vals, delta, color="steelblue", edgecolor="k")
    ax2.axhline(0, color="gray", linestyle="--")
    ax2.set_xlabel("Shadow rounds per target")
    ax2.set_ylabel("Adv(AUC) − Adv(AUC @ rounds=20)")
    ax2.set_title("Relative gain vs reference (rounds=20)")
    ax2.set_xticks(r_vals)
    ax2.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig9_partial_residuals(deduped: pd.DataFrame, out_path: Path) -> None:
    """对 adv_auc、stability 分别回归掉 train_size 后的残差散点 + 拟合线。"""
    df = deduped.dropna(subset=["train_size", "adv_auc", "stability_metric"]).copy()
    if len(df) < 5:
        return
    X_ts = df[["train_size"]].to_numpy()
    y_adv = df["adv_auc"].to_numpy()
    y_stab = df["stability_metric"].to_numpy()
    # Residualize
    lr_adv = LinearRegression().fit(X_ts, y_adv)
    lr_stab = LinearRegression().fit(X_ts, y_stab)
    res_adv = y_adv - lr_adv.predict(X_ts)
    res_stab = y_stab - lr_stab.predict(X_ts)
    # Partial correlation (Spearman on residuals)
    rho, p = stats.spearmanr(res_adv, res_stab)
    # OLS on residuals: res_adv ~ res_stab
    lr_partial = LinearRegression().fit(res_stab.reshape(-1, 1), res_adv)
    x_line = np.linspace(res_stab.min(), res_stab.max(), 100)
    y_line = lr_partial.predict(x_line.reshape(-1, 1))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(res_stab, res_adv, alpha=0.7, s=50)
    ax.plot(x_line, y_line, "k-", lw=2, label=f"fit (partial); Spearman ρ={rho:.3f}, p={p:.4f}")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Stability residual (after regressing out train_size)")
    ax.set_ylabel("Adv(AUC) residual (after regressing out train_size)")
    ax.set_title("Partial correlation: Adv(AUC) vs Stability | train_size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig10_iter_ci_sig(iter_balanced: pd.DataFrame, iter_tests: pd.DataFrame, out_path: Path) -> None:
    """n_iter vs adv_auc：均值 + 95% CI + 显著性（vs 1000）。"""
    df = iter_balanced.dropna(subset=["n_iter", "adv_auc"]).copy()
    if df.empty:
        return
    it_vals = sorted(df["n_iter"].unique())
    means, lo, hi, sigs = [], [], [], []
    for it in it_vals:
        block = df[df["n_iter"] == it]["adv_auc"].to_numpy()
        m, l, h = _bootstrap_mean_ci(block)
        means.append(m)
        lo.append(l)
        hi.append(h)
        if it == REF_N_ITER:
            sigs.append("")
        else:
            row = iter_tests[iter_tests["compare"] == f"{REF_N_ITER} vs {it}"]
            p = row["p_holm_welch"].iloc[0] if not row.empty else 1.0
            sigs.append(_sig_stars(p))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(it_vals, means, yerr=[np.array(means) - np.array(lo), np.array(hi) - np.array(means)], fmt="o-", capsize=5, capthick=2)
    for i, (it, s) in enumerate(zip(it_vals, sigs)):
        if s:
            ax.annotate(s, (it, hi[i] + 0.01), ha="center", fontsize=12)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Training iterations (n_iter)")
    ax.set_ylabel("Adv(AUC)")
    ax.set_title("Adv(AUC) vs n_iter (95% CI; * p<0.05, ** p<0.01, *** p<0.001 vs 1000); non-monotonic")
    ax.set_xticks(it_vals)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------- 2 Partial correlation & robust regression ----------
def run_partial_corr_and_robust(deduped: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Partial correlation（控制 train_size）+ OLS 与 Huber/Theil-Sen 回归。"""
    df = deduped.dropna(subset=["train_size", "adv_auc", "stability_metric"]).copy()
    if len(df) < 5:
        return pd.DataFrame(), pd.DataFrame()

    # Partial: residualize then Spearman
    X_ts = df[["train_size"]].to_numpy()
    y_adv = df["adv_auc"].to_numpy()
    y_stab = df["stability_metric"].to_numpy()
    res_adv = y_adv - LinearRegression().fit(X_ts, y_adv).predict(X_ts)
    res_stab = y_stab - LinearRegression().fit(X_ts, y_stab).predict(X_ts)
    rho_partial, p_partial = stats.spearmanr(res_adv, res_stab)
    # OLS: adv_auc ~ stability + train_size
    X = df[["stability_metric", "train_size"]].to_numpy()
    lr = LinearRegression().fit(X, y_adv)
    # t/p for stability coefficient (simplified: use scipy one-way or just report coef)
    pred = lr.predict(X)
    resid = y_adv - pred
    n_obs, k_reg = len(X), X.shape[1]  # k_reg = 2 (stability, train_size)
    var_resid = np.var(resid, ddof=k_reg)
    X_with_const = np.column_stack([np.ones(len(X)), X])
    try:
        cov = var_resid * np.linalg.inv(X_with_const.T @ X_with_const)
        se_stab = np.sqrt(cov[1, 1])
        t_stab = lr.coef_[0] / se_stab if se_stab > 0 else 0
        p_stab = 2 * (1 - stats.t.cdf(abs(t_stab), n_obs - k_reg - 1))
    except Exception:
        t_stab, p_stab = np.nan, np.nan

    table_partial = pd.DataFrame([{
        "partial_spearman_rho": rho_partial,
        "partial_p": p_partial,
        "ols_stability_coef": lr.coef_[0],
        "ols_stability_t": t_stab,
        "ols_stability_p": p_stab,
        "n": len(df),
    }])

    # Robust: adv_auc ~ stability (no train_size for direct comparison with OLS slope)
    x_s = df["stability_metric"].to_numpy().reshape(-1, 1)
    y_a = df["adv_auc"].to_numpy()
    ols = LinearRegression().fit(x_s, y_a)
    huber = HuberRegressor().fit(x_s, y_a)
    theil = TheilSenRegressor().fit(x_s, y_a)
    # Bootstrap CI for slopes
    slopes_huber = []
    slopes_theil = []
    for _ in range(N_BOOT):
        idx = RNG.integers(0, len(y_a), size=len(y_a))
        slopes_huber.append(HuberRegressor().fit(x_s[idx], y_a[idx]).coef_[0])
        slopes_theil.append(TheilSenRegressor().fit(x_s[idx], y_a[idx]).coef_[0])
    table_robust = pd.DataFrame([{
        "ols_slope": ols.coef_[0],
        "huber_slope": huber.coef_[0],
        "huber_slope_ci95_low": float(np.percentile(slopes_huber, 2.5)),
        "huber_slope_ci95_high": float(np.percentile(slopes_huber, 97.5)),
        "theilsen_slope": theil.coef_[0],
        "theilsen_slope_ci95_low": float(np.percentile(slopes_theil, 2.5)),
        "theilsen_slope_ci95_high": float(np.percentile(slopes_theil, 97.5)),
        "n": len(df),
    }])
    return table_partial, table_robust


def _compute_learned_auc_one_run(run_dir_path: str, cv: int = 5) -> Optional[float]:
    """单 run 的 learned AUC（LR 5-fold CV），失败返回 None。"""
    run_dir = Path(run_dir_path)
    m = run_dir / "shadow_pair_metrics"
    pair_path = m / "pair_metrics.csv"
    pair_control_path = m / "pair_metrics_control.csv"
    if not pair_path.exists() or not pair_control_path.exists():
        return None
    df_m = pd.read_csv(pair_path)
    df_c = pd.read_csv(pair_control_path)
    if df_m.empty or df_c.empty:
        return None
    numeric_m = set(df_m.select_dtypes(include=[np.number]).columns)
    numeric_c = set(df_c.select_dtypes(include=[np.number]).columns)
    common = numeric_m & numeric_c
    skip = {"target_idx", "round"}
    feature_cols = [c for c in FEATURE_CANDIDATES if c in common] or [c for c in sorted(common) if c not in skip]
    if not feature_cols:
        return None
    X_m = df_m[feature_cols].replace([np.inf, -np.inf], np.nan).dropna(how="all")
    X_c = df_c[feature_cols].replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if X_m.empty or X_c.empty:
        return None
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
        return None
    clf = LogisticRegression(max_iter=1000, random_state=42)
    res = cross_validate(clf, X, y, cv=cv, scoring="roc_auc", return_train_score=False)
    return float(np.mean(res["test_score"]))


# ---------- 3 Learned attacker tests ----------
def run_learned_vs_random(scaling_with_learned: pd.DataFrame) -> pd.DataFrame:
    """每个 train_size：单样本 t (H0: mean=0.5)、Wilcoxon；Holm 校正。"""
    df = scaling_with_learned.dropna(subset=["train_size", "learned_adv"]).copy()
    if df.empty or "learned_adv" not in df.columns:
        return pd.DataFrame()
    rows = []
    p_list = []
    for tr in sorted(df["train_size"].unique()):
        block = (df[df["train_size"] == tr]["learned_adv"] - 0.5).to_numpy()
        if block.size < 2:
            continue
        t_stat, p_ttest = stats.ttest_1samp(block + 0.5, 0.5)
        p_wilcoxon = stats.wilcoxon(block, alternative="two-sided").pvalue
        m = (block + 0.5).mean()
        se = block.std(ddof=1) / np.sqrt(len(block)) if len(block) > 1 else 0
        ci_lo = m - 1.96 * se
        ci_hi = m + 1.96 * se
        rows.append({
            "train_size": int(tr),
            "mean_learned_auc": m,
            "mean_minus_0.5": m - 0.5,
            "ci95_low": ci_lo,
            "ci95_high": ci_hi,
            "p_ttest": p_ttest,
            "p_wilcoxon": p_wilcoxon,
            "n": len(block),
        })
        p_list.append(p_ttest)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm_ttest"] = holm_correction(p_list)
    return out


def run_learned_group_test(scaling_with_learned: pd.DataFrame) -> pd.DataFrame:
    """One-way ANOVA / Kruskal-Wallis：train_size 是否影响 learned_auc。"""
    df = scaling_with_learned.dropna(subset=["train_size", "learned_adv"]).copy()
    if df.empty or "learned_adv" not in df.columns:
        return pd.DataFrame()
    groups = [df[df["train_size"] == tr]["learned_auc"].to_numpy() for tr in sorted(df["train_size"].unique())]
    groups = [g for g in groups if len(g) >= 1]
    if len(groups) < 2:
        return pd.DataFrame()
    f_stat, p_anova = stats.f_oneway(*groups)
    try:
        h_stat, p_kw = stats.kruskal(*groups)
    except Exception:
        h_stat, p_kw = np.nan, np.nan
    return pd.DataFrame([{
        "test": "ANOVA",
        "statistic": f_stat,
        "p_value": p_anova,
        "n_groups": len(groups),
    }, {
        "test": "Kruskal-Wallis",
        "statistic": h_stat,
        "p_value": p_kw,
        "n_groups": len(groups),
    }])


def main() -> int:
    parser = argparse.ArgumentParser(description="Adult+CTGAN 统计分析：锁定数据、显著性检验、偏相关、稳健回归、learned 检验")
    parser.add_argument("--summary", type=Path, default=PROJECT_ROOT / "outputs" / "postprocess_adult_ctgan" / "adult_ctgan_run_summary.csv", help="adult_ctgan_run_summary.csv 路径")
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "outputs" / "postprocess_adult_ctgan" / "stats", help="输出目录")
    parser.add_argument("--verify-completeness", action="store_true", help="是否对每个 run 再检查完整性（pair_metrics 行数）")
    parser.add_argument("--scaling-with-learned", type=Path, default=None, help="若已有 scaling_balanced + learned_adv 的 CSV，可传入以做 learned 检验")
    parser.add_argument("--compute-learned", action="store_true", help="若 scaling_balanced 无 learned_adv 列，则根据 run_dir_path 现场计算（较慢）")
    args = parser.parse_args()

    summary_path = args.summary.resolve()
    if not summary_path.exists():
        print(f"总表不存在: {summary_path}")
        return 1

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    fig_dir = outdir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 指令 0：锁定数据集 ----------
    deduped, dropped = lock_dataset(summary_path, verify_completeness=args.verify_completeness)
    if dropped is not None and not dropped.empty:
        dropped_out = dropped[["run_dir", "run_dir_path", "drop_reason"]].drop_duplicates() if "run_dir_path" in dropped.columns else dropped[["run_dir", "drop_reason"]].drop_duplicates()
        dropped_out.to_csv(outdir / "dropped_runs.csv", index=False, encoding="utf-8-sig")
        print(f"dropped_runs.csv: {len(dropped_out)} 条")

    scaling_balanced, rounds_balanced, iter_balanced, seeds_a, smin_b, imin_c = balance_subsets(deduped)

    # 输出标准化列名（含 train_size, rounds_per_target）
    def _ensure_cols(d: pd.DataFrame) -> pd.DataFrame:
        if d.empty:
            return d
        if "train_rows" in d.columns and "train_size" not in d.columns:
            d = d.rename(columns={"train_rows": "train_size"})
        if "rounds" in d.columns and "rounds_per_target" not in d.columns:
            d = d.rename(columns={"rounds": "rounds_per_target"})
        return d

    scaling_balanced = _ensure_cols(scaling_balanced)
    rounds_balanced = _ensure_cols(rounds_balanced)
    iter_balanced = _ensure_cols(iter_balanced)

    scaling_balanced.to_csv(outdir / "scaling_balanced.csv", index=False, encoding="utf-8-sig")
    rounds_balanced.to_csv(outdir / "rounds_balanced.csv", index=False, encoding="utf-8-sig")
    iter_balanced.to_csv(outdir / "iter_balanced.csv", index=False, encoding="utf-8-sig")
    print(f"scaling_balanced: {outdir / 'scaling_balanced.csv'} (rows={len(scaling_balanced)}, seeds={seeds_a})")
    print(f"rounds_balanced: {outdir / 'rounds_balanced.csv'} (rows={len(rounds_balanced)}, smin={smin_b})")
    print(f"iter_balanced: {outdir / 'iter_balanced.csv'} (rows={len(iter_balanced)}, imin={imin_c})")

    if deduped.empty:
        print("去重后无数据，退出")
        return 0

    # ---------- 指令 1：Scaling / Rounds / Iter 检验 + 图 ----------
    table1 = run_scaling_tests(scaling_balanced)
    table1.to_csv(outdir / "table1_scaling_tests.csv", index=False, encoding="utf-8-sig")
    if not table1.empty:
        fig7_adv_scaling_ci_sig(scaling_balanced, table1, fig_dir / "fig7_adv_scaling_ci_sig.png")

    table2 = run_rounds_tests(rounds_balanced)
    table2.to_csv(outdir / "table2_rounds_tests.csv", index=False, encoding="utf-8-sig")
    if not rounds_balanced.empty:
        fig8_adv_rounds_delta_from_20(rounds_balanced, fig_dir / "fig8_adv_rounds_delta_from_20.png")

    table3 = run_iter_tests(iter_balanced)
    table3.to_csv(outdir / "table3_iter_tests.csv", index=False, encoding="utf-8-sig")
    if not iter_balanced.empty and not table3.empty:
        fig10_iter_ci_sig(iter_balanced, table3, fig_dir / "fig10_iter_ci_sig.png")

    # ---------- 指令 2：Partial correlation + robust regression ----------
    table_partial, table_robust = run_partial_corr_and_robust(deduped)
    if not table_partial.empty:
        table_partial.to_csv(outdir / "table4_partial_corr.csv", index=False, encoding="utf-8-sig")
    if not table_robust.empty:
        table_robust.to_csv(outdir / "table5_robust_regression.csv", index=False, encoding="utf-8-sig")
    if not deduped.empty and "stability_metric" in deduped.columns:
        fig9_partial_residuals(deduped, fig_dir / "fig9_partial_residuals_stability.png")

    # ---------- 指令 3：Learned attacker ----------
    if args.scaling_with_learned and args.scaling_with_learned.exists():
        df_learned = pd.read_csv(args.scaling_with_learned)
        df_learned = _ensure_cols(df_learned)
    else:
        df_learned = scaling_balanced.copy()
        if "learned_adv" not in df_learned.columns and args.compute_learned and "run_dir_path" in df_learned.columns:
            learned_vals = [_compute_learned_auc_one_run(p) for p in df_learned["run_dir_path"].astype(str)]
            df_learned["learned_adv"] = [np.nan if v is None else v for v in learned_vals]

    if not df_learned.empty and "learned_adv" in df_learned.columns:
        table_learned = run_learned_vs_random(df_learned)
        table_learned.to_csv(outdir / "table6_learned_vs_random.csv", index=False, encoding="utf-8-sig")
        table_learned_group = run_learned_group_test(df_learned)
        table_learned_group.to_csv(outdir / "table7_learned_group_test.csv", index=False, encoding="utf-8-sig")

    print("Done. Tables and figures in", outdir)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
