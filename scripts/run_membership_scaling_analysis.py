#!/usr/bin/env python3
"""
Membership signal scaling 分析：验证 adv_abs 随 N 衰减，并比较多种局部信号与 tail risk。

统一输入：每个 run 需有 shadow_pair_metrics/pair_metrics.csv 与 pair_metrics_control.csv（用于 AUC）。
输出：run_level_adv_abs.csv, run_level_adv_abs_density.csv, run_level_adv_abs_clf.csv,
      run_level_tail_metrics.csv, scaling_model_summary.csv, shift_partial_summary.csv，
      以及 fig_main_adv_abs_vs_N.png, fig_main_adv_abs_vs_N_pooled.png, fig_density_*, fig_clf_*, fig_tail_*, fig_fraction_*.

用法:
  python scripts/run_membership_scaling_analysis.py --outputs outputs --outdir outputs/figures
  --table 可指定已有 run 表；--n_jobs N 为并行进程数（默认 CPU 数-1，<=1 单进程）。
"""

from __future__ import annotations

import argparse
import multiprocessing
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 与 build_run_level_bridge 一致
DDPM_TRAIN_SIZES = [200, 500, 1000, 1500]
CTGAN_TRAIN_SIZES = [200, 500, 1000, 1500, 2000]
CTGAN_ROUNDS = 20
MAINLINE_SEEDS = [42, 43, 44, 45, 46]
DENSITY_EPS = 1e-10


def _is_ddpm_run(dirname: str) -> bool:
    n = dirname.lower()
    return "ddpm" in n or "tabddpm" in n


def _is_ctgan_run(dirname: str) -> bool:
    return "ctgan" in dirname.lower()


def parse_run_dir(dirname: str) -> Dict[str, Any]:
    out = {
        "train_size": None, "n_iter": None, "batch_size": None, "rounds": None,
        "candidate": None, "seed": None, "timestamp": None, "model": None,
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


def scan_control_runs(outputs_dir: Path) -> List[Path]:
    outputs_dir = outputs_dir.resolve()
    if not outputs_dir.is_dir():
        return []
    run_dirs = []
    for d in outputs_dir.iterdir():
        if not d.is_dir() or "control" not in d.name:
            continue
        metrics_dir = d / "shadow_pair_metrics"
        if metrics_dir.is_dir() and (metrics_dir / "pair_metrics.csv").exists():
            run_dirs.append(d)
    for sub in outputs_dir.iterdir():
        if not sub.is_dir() or sub.name.startswith("."):
            continue
        for d in sub.iterdir():
            if not d.is_dir() or "control" not in d.name:
                continue
            metrics_dir = d / "shadow_pair_metrics"
            if metrics_dir.is_dir() and (metrics_dir / "pair_metrics.csv").exists():
                run_dirs.append(d)
    return sorted(set(run_dirs), key=lambda p: (p.name, str(p)))


def filter_and_dedupe_runs(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    ddpm = df[df["model"] == "ddpm"].copy()
    ddpm = ddpm[ddpm["train_size"].isin(DDPM_TRAIN_SIZES) & ddpm["seed"].isin(MAINLINE_SEEDS)]
    if not ddpm.empty:
        ddpm = ddpm.sort_values("timestamp", ascending=True, na_position="last")
        ddpm = ddpm.drop_duplicates(subset=["train_size", "seed"], keep="last")
    ctgan = df[df["model"] == "ctgan"].copy()
    ctgan = ctgan[
        ctgan["train_size"].isin(CTGAN_TRAIN_SIZES)
        & (ctgan["rounds"] == CTGAN_ROUNDS)
        & ctgan["seed"].isin(MAINLINE_SEEDS)
    ]
    if not ctgan.empty:
        ctgan = ctgan.sort_values("timestamp", ascending=True, na_position="last")
        ctgan = ctgan.drop_duplicates(subset=["train_size", "seed"], keep="last")
    out = pd.concat([ddpm, ctgan], ignore_index=True) if not ddpm.empty or not ctgan.empty else pd.DataFrame()
    return out.sort_values(["model", "train_size", "seed"]).reset_index(drop=True) if not out.empty else out


def s_run_trim95_from_pair_metrics(pair_path: Path) -> Optional[float]:
    """从 pair_metrics 的 mmd_mixed 列计算 trim95 均值。"""
    if not pair_path.exists():
        return None
    df = pd.read_csv(pair_path)
    col = "mmd_mixed" if "mmd_mixed" in df.columns else "mmd_numeric"
    if col not in df.columns:
        return None
    M = df[col].astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(M) == 0:
        return None
    n = len(M)
    lo = max(0, int(np.ceil(0.025 * n)))
    hi = max(lo, min(n, int(np.floor(0.975 * n))))
    if hi <= lo:
        return float(np.median(M))
    return float(np.mean(np.sort(M)[lo:hi]))


# ---------- PART 1: adv_abs ----------
def compute_run_adv_abs(pair_path: Path, pair_control_path: Path) -> Tuple[Optional[float], int]:
    """member=pair_metrics, non-member=control；返回 (adv_abs, n_targets)。n_targets = member 行数。"""
    if not pair_path.exists() or not pair_control_path.exists():
        return None, 0
    df_m = pd.read_csv(pair_path)
    df_c = pd.read_csv(pair_control_path)
    if df_m.empty or df_c.empty:
        return None, 0
    delta_col = _choose_delta_column(df_m) or _choose_delta_column(df_c)
    if not delta_col or delta_col not in df_m.columns or delta_col not in df_c.columns:
        return None, len(df_m)
    scores_m = df_m[delta_col].astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    scores_c = df_c[delta_col].astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if scores_m.size == 0 or scores_c.size == 0:
        return None, len(df_m)
    scores = np.concatenate([scores_m, scores_c])
    labels = np.concatenate([np.ones(len(scores_m)), np.zeros(len(scores_c))])
    auc = _auc_from_scores_labels(scores, labels)
    if np.isnan(auc):
        return None, len(df_m)
    adv_abs = float(abs(auc - 0.5))
    return adv_abs, len(df_m)


# ---------- PART 2: density (k=1: log(r_in)-log(r_out)) ----------
def compute_density_score_column(df: pd.DataFrame) -> Optional[np.ndarray]:
    """要求 df 有 min_dist_in, min_dist_out。返回 density_score 数组。"""
    if "min_dist_in" not in df.columns or "min_dist_out" not in df.columns:
        return None
    din = df["min_dist_in"].astype(float).replace([np.inf, -np.inf], np.nan)
    dout = df["min_dist_out"].astype(float).replace([np.inf, -np.inf], np.nan)
    din = np.maximum(din.to_numpy(), DENSITY_EPS)
    dout = np.maximum(dout.to_numpy(), DENSITY_EPS)
    return np.log(din) - np.log(dout)


def compute_run_adv_abs_density(pair_path: Path, pair_control_path: Path) -> Tuple[Optional[float], int]:
    """用 density_score 做 AUC，返回 (adv_abs_density, n_targets)。"""
    if not pair_path.exists() or not pair_control_path.exists():
        return None, 0
    df_m = pd.read_csv(pair_path)
    df_c = pd.read_csv(pair_control_path)
    if df_m.empty or df_c.empty:
        return None, 0
    dm = compute_density_score_column(df_m)
    dc = compute_density_score_column(df_c)
    if dm is None or dc is None:
        return None, len(df_m)
    dm = dm[np.isfinite(dm)]
    dc = dc[np.isfinite(dc)]
    if dm.size == 0 or dc.size == 0:
        return None, len(df_m)
    scores = np.concatenate([dm, dc])
    labels = np.concatenate([np.ones(len(dm)), np.zeros(len(dc))])
    auc = _auc_from_scores_labels(scores, labels)
    if np.isnan(auc):
        return None, len(df_m)
    adv_abs_density = float(abs(auc - 0.5))
    return adv_abs_density, len(df_m)


# ---------- PART 3: classifier ----------
def build_feature_matrix(
    df: pd.DataFrame,
    delta_col: str,
    density_arr: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """特征：delta, density, min_dist_in, min_dist_out。若缺列则用 0 或跳过。"""
    need = [delta_col, "min_dist_in", "min_dist_out"]
    if not all(c in df.columns for c in need):
        return None
    X = df[need].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()
    if density_arr is not None and len(density_arr) == len(X):
        X = np.column_stack([X, density_arr])
    else:
        if "min_dist_in" in df.columns and "min_dist_out" in df.columns:
            din = np.maximum(df["min_dist_in"].astype(float).fillna(0).to_numpy(), DENSITY_EPS)
            dout = np.maximum(df["min_dist_out"].astype(float).fillna(0).to_numpy(), DENSITY_EPS)
            density = np.log(din) - np.log(dout)
            X = np.column_stack([X, density])
    return X


def compute_run_adv_abs_clf(
    pair_path: Path,
    pair_control_path: Path,
    cv: int = 5,
    random_state: int = 42,
) -> Tuple[Optional[float], int]:
    """对 run 内 member + control 的 feature 训练 LR，CV 得 AUC_clf，返回 (adv_abs_clf, n_targets)。"""
    if not pair_path.exists() or not pair_control_path.exists():
        return None, 0
    df_m = pd.read_csv(pair_path)
    df_c = pd.read_csv(pair_control_path)
    if df_m.empty or df_c.empty:
        return None, 0
    delta_col = _choose_delta_column(df_m) or _choose_delta_column(df_c)
    if not delta_col:
        return None, len(df_m)
    dm = compute_density_score_column(df_m)
    dc = compute_density_score_column(df_c)
    Xm = build_feature_matrix(df_m, delta_col, dm)
    Xc = build_feature_matrix(df_c, delta_col, dc)
    if Xm is None or Xc is None or Xm.shape[1] != Xc.shape[1]:
        return None, len(df_m)
    X = np.vstack([Xm, Xc])
    y = np.concatenate([np.ones(len(Xm)), np.zeros(len(Xc))])
    if np.unique(y).size < 2 or X.shape[0] < 10:
        return None, len(df_m)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, random_state=random_state)
    try:
        preds = cross_val_predict(clf, Xs, y, cv=min(cv, len(y) // 2), method="predict_proba")
        proba = preds[:, 1]
        auc_clf = _auc_from_scores_labels(proba, y)
    except Exception:
        return None, len(df_m)
    if np.isnan(auc_clf):
        return None, len(df_m)
    adv_abs_clf = float(abs(auc_clf - 0.5))
    return adv_abs_clf, len(df_m)


# ---------- PART 4: tail ----------
def compute_tail_metrics(pair_path: Path) -> Tuple[Optional[float], Optional[float], Optional[float], int]:
    """signal_strength = |delta_min_dist|；返回 (top1_mean, top5_mean, fraction_high_risk, n)。"""
    if not pair_path.exists():
        return None, None, None, 0
    df = pd.read_csv(pair_path)
    delta_col = _choose_delta_column(df)
    if not delta_col or delta_col not in df.columns:
        return None, None, None, len(df)
    s = np.abs(df[delta_col].astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy())
    if len(s) == 0:
        return None, None, None, len(df)
    n = len(s)
    med = np.median(s)
    frac_high = float(np.mean(s > med))
    top1 = float(np.mean(s[s >= np.percentile(s, 99)])) if n >= 100 else float(np.mean(s))
    top5 = float(np.mean(s[s >= np.percentile(s, 95)])) if n >= 20 else float(np.mean(s))
    return top1, top5, frac_high, n


# ---------- 多进程 worker（模块级，便于 pickle） ----------
def _worker_one_run(run_dir_str: str) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
    """单 run 计算 adv / density / clf / tail 四行。入参为 run_dir 的路径字符串。"""
    run_dir = Path(run_dir_str)
    parsed = parse_run_dir(run_dir.name)
    if not parsed.get("model"):
        return None
    metrics_dir = run_dir / "shadow_pair_metrics"
    pair_path = metrics_dir / "pair_metrics.csv"
    pair_control_path = metrics_dir / "pair_metrics_control.csv"
    run_id = run_dir.name

    adv_abs, n_targets = compute_run_adv_abs(pair_path, pair_control_path)
    adv_row = {
        "run_id": run_id,
        "run_dir_path": str(run_dir.resolve()),
        "model": parsed["model"],
        "train_size": parsed["train_size"],
        "seed": parsed["seed"],
        "adv_abs": adv_abs if adv_abs is not None else np.nan,
        "n_targets": n_targets,
    }

    adv_d, _ = compute_run_adv_abs_density(pair_path, pair_control_path)
    density_row = {
        "run_id": run_id,
        "model": parsed["model"],
        "train_size": parsed["train_size"],
        "adv_abs_density": adv_d if adv_d is not None else np.nan,
    }

    adv_c, _ = compute_run_adv_abs_clf(pair_path, pair_control_path)
    clf_row = {
        "run_id": run_id,
        "model": parsed["model"],
        "train_size": parsed["train_size"],
        "adv_abs_clf": adv_c if adv_c is not None else np.nan,
    }

    top1, top5, frac, _ = compute_tail_metrics(pair_path)
    tail_row = {
        "run_id": run_id,
        "model": parsed["model"],
        "train_size": parsed["train_size"],
        "tail_top1_mean": top1 if top1 is not None else np.nan,
        "tail_top5_mean": top5 if top5 is not None else np.nan,
        "fraction_high_risk": frac if frac is not None else np.nan,
    }
    return (adv_row, density_row, clf_row, tail_row)


# ---------- 主流程：收集 run 行（多进程） ----------
def collect_all_run_rows(
    run_dirs: List[Path],
    n_jobs: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """返回 (adv_abs_df, density_df, clf_df, tail_df)。n_jobs<=1 时单进程，>1 时多进程。"""
    run_dir_strs = [str(p.resolve()) for p in run_dirs]
    if n_jobs is None or n_jobs <= 1:
        results = []
        for s in run_dir_strs:
            try:
                r = _worker_one_run(s)
                if r is not None:
                    results.append(r)
            except Exception:
                pass
    else:
        with multiprocessing.Pool(processes=n_jobs) as pool:
            results = [r for r in pool.imap(_worker_one_run, run_dir_strs, chunksize=1) if r is not None]

    adv_rows = [r[0] for r in results]
    density_rows = [r[1] for r in results]
    clf_rows = [r[2] for r in results]
    tail_rows = [r[3] for r in results]
    return (
        pd.DataFrame(adv_rows),
        pd.DataFrame(density_rows),
        pd.DataFrame(clf_rows),
        pd.DataFrame(tail_rows),
    )


def fit_scaling_models(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled 数据拟合 A: adv_abs ~ 1/sqrt(N), B: adv_abs ~ log(N), C: adv_abs ~ N^{-alpha}。"""
    df = df.dropna(subset=["adv_abs", "train_size"]).copy()
    df = df[df["train_size"] > 0]
    if len(df) < 3:
        return pd.DataFrame()
    y = df["adv_abs"].to_numpy(dtype=float)
    N = df["train_size"].to_numpy(dtype=float)
    n = len(y)
    rows = []

    # A: adv_abs = beta * (1/sqrt(N))
    x_a = 1.0 / np.sqrt(N)
    if np.isfinite(x_a).all():
        X_a = x_a.reshape(-1, 1)
        from numpy.linalg import lstsq
        try:
            beta_a, _, _, _ = lstsq(X_a, y, rcond=None)
            beta_a = beta_a[0]
            yhat_a = beta_a * x_a
            rss_a = np.sum((y - yhat_a) ** 2)
            r2_a = 1 - rss_a / np.sum((y - np.mean(y)) ** 2) if np.var(y) > 0 else 0
            k_a = 1
            aic_a = n * np.log(rss_a / n + 1e-12) + 2 * k_a
            bic_a = n * np.log(rss_a / n + 1e-12) + k_a * np.log(n)
            _, _, r_pearson, p_pearson, _ = stats.linregress(x_a, y)
        except Exception:
            beta_a, r2_a, aic_a, bic_a, p_pearson = np.nan, np.nan, np.nan, np.nan, np.nan
        rows.append({
            "model_name": "A_1_over_sqrt_N",
            "beta": beta_a,
            "alpha": np.nan,
            "p_value": p_pearson,
            "R2": r2_a,
            "AIC": aic_a,
            "BIC": bic_a,
            "n": n,
        })

    # B: adv_abs ~ a + b*log(N)
    logN = np.log(N)
    if np.isfinite(logN).all():
        try:
            slope, intercept, r, p_b, _ = stats.linregress(logN, y)
            yhat_b = intercept + slope * logN
            rss_b = np.sum((y - yhat_b) ** 2)
            r2_b = 1 - rss_b / np.sum((y - np.mean(y)) ** 2) if np.var(y) > 0 else 0
            k_b = 2
            aic_b = n * np.log(rss_b / n + 1e-12) + 2 * k_b
            bic_b = n * np.log(rss_b / n + 1e-12) + k_b * np.log(n)
        except Exception:
            slope, r2_b, aic_b, bic_b, p_b = np.nan, np.nan, np.nan, np.nan, np.nan
        rows.append({
            "model_name": "B_log_N",
            "beta": slope,
            "alpha": np.nan,
            "p_value": p_b,
            "R2": r2_b,
            "AIC": aic_b,
            "BIC": bic_b,
            "n": n,
        })

    # C: adv_abs = c * N^{-alpha}
    def power_law(N_arr: np.ndarray, c: float, alpha: float) -> np.ndarray:
        return c * (N_arr.astype(float) ** (-alpha))

    try:
        popt, _ = curve_fit(
            lambda N_arr, c, alpha: power_law(N_arr, c, alpha),
            N, y,
            p0=[0.5, 0.5],
            bounds=([1e-6, 0.01], [10, 3]),
            maxfev=5000,
        )
        c_c, alpha_c = popt[0], popt[1]
        yhat_c = power_law(N, c_c, alpha_c)
        rss_c = np.sum((y - yhat_c) ** 2)
        r2_c = 1 - rss_c / np.sum((y - np.mean(y)) ** 2) if np.var(y) > 0 else 0
        k_c = 2
        aic_c = n * np.log(rss_c / n + 1e-12) + 2 * k_c
        bic_c = n * np.log(rss_c / n + 1e-12) + k_c * np.log(n)
        p_c = np.nan  # no simple p for nonlinear
    except Exception:
        c_c, alpha_c, r2_c, aic_c, bic_c, p_c = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    rows.append({
        "model_name": "C_N_power_alpha",
        "beta": c_c,
        "alpha": alpha_c,
        "p_value": p_c,
        "R2": r2_c,
        "AIC": aic_c,
        "BIC": bic_c,
        "n": n,
    })

    summary = pd.DataFrame(rows)
    if "AIC" in summary.columns and summary["AIC"].notna().any():
        best = summary.loc[summary["AIC"].idxmin(), "model_name"]
        print(f"[Scaling] 最小 AIC 模型: {best}")
    return summary


def run_shift_partial(adv_df: pd.DataFrame, run_dirs: List[Path]) -> pd.DataFrame:
    """adv_abs ~ s_run_trim95 + train_size + model 与 adv_abs ~ train_size + model，比较 R2 增量。"""
    adv_df = adv_df.dropna(subset=["adv_abs", "train_size"]).copy()
    if adv_df.empty:
        return pd.DataFrame()
    run_by_path = {str(p.resolve()): p for p in run_dirs}
    path_col = "run_dir_path" if "run_dir_path" in adv_df.columns else "run_id"
    if path_col not in adv_df.columns:
        return pd.DataFrame()
    s_run_list = []
    for _, row in adv_df.iterrows():
        path = row.get("run_dir_path") or row.get("run_id")
        if isinstance(path, str) and path in run_by_path:
            run_dir = run_by_path[path]
            s95 = s_run_trim95_from_pair_metrics(run_dir / "shadow_pair_metrics" / "pair_metrics.csv")
            s_run_list.append(s95)
        else:
            s_run_list.append(np.nan)
    adv_df = adv_df.copy()
    adv_df["s_run_trim95"] = s_run_list
    adv_df = adv_df.dropna(subset=["s_run_trim95"])
    if len(adv_df) < 4:
        return pd.DataFrame()
    y = adv_df["adv_abs"].to_numpy(dtype=float)
    N_z = (adv_df["train_size"].to_numpy(dtype=float) - adv_df["train_size"].mean()) / (adv_df["train_size"].std() + 1e-12)
    model_ddpm = (adv_df["model"] == "ddpm").astype(int).to_numpy()
    X_small = np.column_stack([np.ones(len(adv_df)), N_z, model_ddpm])
    X_full = np.column_stack([np.ones(len(adv_df)), adv_df["s_run_trim95"].to_numpy(dtype=float), N_z, model_ddpm])
    from numpy.linalg import lstsq
    try:
        b_small, _, _, _ = lstsq(X_small, y, rcond=None)
        b_full, _, _, _ = lstsq(X_full, y, rcond=None)
        rss_small = np.sum((y - X_small @ b_small) ** 2)
        rss_full = np.sum((y - X_full @ b_full) ** 2)
        tss = np.sum((y - np.mean(y)) ** 2)
        r2_small = 1 - rss_small / tss if tss > 0 else 0
        r2_full = 1 - rss_full / tss if tss > 0 else 0
        delta_r2 = r2_full - r2_small
    except Exception:
        r2_small = r2_full = delta_r2 = np.nan
    return pd.DataFrame([{
        "model_without_shift": "adv_abs ~ train_size + model",
        "model_with_shift": "adv_abs ~ s_run_trim95 + train_size + model",
        "R2_without_shift": r2_small,
        "R2_with_shift": r2_full,
        "R2_increment": delta_r2,
    }])


def plot_adv_abs_vs_N(adv_df: pd.DataFrame, outdir: Path) -> None:
    """fig_main_adv_abs_vs_N.png：按 model 上色，每个 N 画 mean±95% CI。"""
    adv_df = adv_df.dropna(subset=["train_size", "adv_abs"]).copy()
    if adv_df.empty or adv_df["train_size"].nunique() < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for model in sorted(adv_df["model"].unique()):
        sub = adv_df[adv_df["model"] == model]
        ax.scatter(sub["train_size"], sub["adv_abs"], label=model, alpha=0.7, s=40)
    agg = adv_df.groupby(["model", "train_size"], as_index=False).agg(
        mean_adv=("adv_abs", "mean"),
        sem=("adv_abs", lambda x: stats.sem(x, nan_policy="omit")),
        count=("adv_abs", "count"),
    )
    for model in agg["model"].unique():
        a = agg[agg["model"] == model]
        x = a["train_size"].values
        y = a["mean_adv"].values
        sem = a["sem"].fillna(0).values
        ci = 1.96 * sem
        ax.errorbar(x, y, yerr=ci, fmt="none", capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel("train_size N (log scale)")
    ax.set_ylabel("adv_abs")
    ax.set_title("adv_abs vs N (by model)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(outdir / "fig_main_adv_abs_vs_N.png", dpi=200)
    plt.close(fig)


def plot_adv_abs_vs_N_pooled(adv_df: pd.DataFrame, outdir: Path) -> None:
    """fig_main_adv_abs_vs_N_pooled.png：所有模型一起，每个 N mean±95% CI。"""
    adv_df = adv_df.dropna(subset=["train_size", "adv_abs"]).copy()
    if adv_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(adv_df["train_size"], adv_df["adv_abs"], alpha=0.6, s=40, c="gray", label="runs")
    agg = adv_df.groupby("train_size", as_index=False).agg(
        mean_adv=("adv_abs", "mean"),
        sem=("adv_abs", lambda x: stats.sem(x, nan_policy="omit")),
    )
    x = agg["train_size"].values
    y = agg["mean_adv"].values
    sem = agg["sem"].fillna(0).values
    ci = 1.96 * sem
    ax.errorbar(x, y, yerr=ci, fmt="o-", capsize=5, label="mean ± 95% CI")
    ax.set_xscale("log")
    ax.set_xlabel("train_size N (log scale)")
    ax.set_ylabel("adv_abs")
    ax.set_title("adv_abs vs N (pooled)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(outdir / "fig_main_adv_abs_vs_N_pooled.png", dpi=200)
    plt.close(fig)


def plot_density_adv_abs_vs_N(density_df: pd.DataFrame, outdir: Path) -> None:
    col = "adv_abs_density"
    if col not in density_df.columns:
        return
    density_df = density_df.dropna(subset=["train_size", col]).copy()
    if density_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for model in sorted(density_df["model"].unique()):
        sub = density_df[density_df["model"] == model]
        ax.scatter(sub["train_size"], sub[col], label=model, alpha=0.7, s=40)
    agg = density_df.groupby(["model", "train_size"], as_index=False).agg(
        mean_adv=(col, "mean"),
        sem=(col, lambda x: stats.sem(x, nan_policy="omit")),
    )
    for model in agg["model"].unique():
        a = agg[agg["model"] == model]
        sem = a["sem"].fillna(0).values
        ax.errorbar(a["train_size"], a["mean_adv"], yerr=1.96 * sem, fmt="none", capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel("train_size N (log scale)")
    ax.set_ylabel("adv_abs_density")
    ax.set_title("adv_abs_density vs N")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(outdir / "fig_density_adv_abs_vs_N.png", dpi=200)
    plt.close(fig)


def plot_clf_adv_abs_vs_N(clf_df: pd.DataFrame, outdir: Path) -> None:
    col = "adv_abs_clf"
    if col not in clf_df.columns:
        return
    clf_df = clf_df.dropna(subset=["train_size", col]).copy()
    if clf_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for model in sorted(clf_df["model"].unique()):
        sub = clf_df[clf_df["model"] == model]
        ax.scatter(sub["train_size"], sub[col], label=model, alpha=0.7, s=40)
    agg = clf_df.groupby(["model", "train_size"], as_index=False).agg(
        mean_adv=(col, "mean"),
        sem=(col, lambda x: stats.sem(x, nan_policy="omit")),
    )
    for model in agg["model"].unique():
        a = agg[agg["model"] == model]
        ax.errorbar(a["train_size"], a["mean_adv"], yerr=1.96 * a["sem"].fillna(0), fmt="none", capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel("train_size N (log scale)")
    ax.set_ylabel("adv_abs_clf")
    ax.set_title("adv_abs_clf vs N")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(outdir / "fig_clf_adv_abs_vs_N.png", dpi=200)
    plt.close(fig)


def plot_tail_top5_vs_N(tail_df: pd.DataFrame, outdir: Path) -> None:
    col = "tail_top5_mean"
    if col not in tail_df.columns:
        return
    tail_df = tail_df.dropna(subset=["train_size", col]).copy()
    if tail_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for model in sorted(tail_df["model"].unique()):
        sub = tail_df[tail_df["model"] == model]
        ax.scatter(sub["train_size"], sub[col], label=model, alpha=0.7, s=40)
    agg = tail_df.groupby(["model", "train_size"], as_index=False).agg(
        mean_tail=(col, "mean"),
        sem=(col, lambda x: stats.sem(x, nan_policy="omit")),
    )
    for model in agg["model"].unique():
        a = agg[agg["model"] == model]
        ax.errorbar(a["train_size"], a["mean_tail"], yerr=1.96 * a["sem"].fillna(0), fmt="none", capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel("train_size N (log scale)")
    ax.set_ylabel("tail top 5% mean(|ΔNN|)")
    ax.set_title("Tail top5 mean vs N")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(outdir / "fig_tail_top5_vs_N.png", dpi=200)
    plt.close(fig)


def plot_fraction_highrisk_vs_N(tail_df: pd.DataFrame, outdir: Path) -> None:
    col = "fraction_high_risk"
    if col not in tail_df.columns:
        return
    tail_df = tail_df.dropna(subset=["train_size", col]).copy()
    if tail_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for model in sorted(tail_df["model"].unique()):
        sub = tail_df[tail_df["model"] == model]
        ax.scatter(sub["train_size"], sub[col], label=model, alpha=0.7, s=40)
    agg = tail_df.groupby(["model", "train_size"], as_index=False).agg(
        mean_frac=(col, "mean"),
        sem=(col, lambda x: stats.sem(x, nan_policy="omit")),
    )
    for model in agg["model"].unique():
        a = agg[agg["model"] == model]
        ax.errorbar(a["train_size"], a["mean_frac"], yerr=1.96 * a["sem"].fillna(0), fmt="none", capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel("train_size N (log scale)")
    ax.set_ylabel("fraction_high_risk")
    ax.set_title("Fraction high risk vs N")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(outdir / "fig_fraction_highrisk_vs_N.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Membership scaling: adv_abs vs N, density, clf, tail, shift partial")
    parser.add_argument("--outputs", type=Path, default=PROJECT_ROOT / "outputs", help="outputs 根目录")
    parser.add_argument("--table", type=Path, default=None, help="可选：已有 run 表 CSV（含 run_dir_path, model, train_size）")
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "outputs" / "figures")
    parser.add_argument("--n_jobs", type=int, default=None, help="并行进程数，默认 CPU 数-1；<=1 为单进程")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    run_dirs: List[Path] = []
    if args.table and args.table.exists():
        bridge = pd.read_csv(args.table)
        run_dir_path_col = "run_dir_path" if "run_dir_path" in bridge.columns else None
        if run_dir_path_col is None:
            run_dir_path_col = next((c for c in bridge.columns if "path" in c.lower() and "run" in c.lower()), None)
        if run_dir_path_col:
            for p in bridge[run_dir_path_col].dropna().unique():
                path = Path(str(p).strip())
                if path.exists():
                    run_dirs.append(path)
                elif not path.is_absolute() and (args.outputs / path).exists():
                    run_dirs.append(args.outputs / path)
        if not run_dirs and "run_dir_name" in bridge.columns:
            out_root = args.outputs.resolve()
            for name in bridge["run_dir_name"].dropna().unique():
                p = out_root / name
                if not p.exists():
                    for sub in out_root.iterdir():
                        if sub.is_dir() and (sub / name).exists():
                            p = sub / name
                            break
                if p.exists():
                    run_dirs.append(p)
            run_dirs = sorted(set(run_dirs), key=lambda x: (x.name, str(x)))
    if not run_dirs:
        run_dirs = scan_control_runs(args.outputs)
        run_rows = []
        for run_dir in run_dirs:
            parsed = parse_run_dir(run_dir.name)
            run_rows.append({
                "run_dir_path": str(run_dir.resolve()),
                "run_dir_name": run_dir.name,
                **parsed,
            })
        run_df = filter_and_dedupe_runs(run_rows)
        if not run_df.empty:
            run_dirs = [Path(p) for p in run_df["run_dir_path"].unique() if Path(p).exists()]
        else:
            run_dirs = run_dirs

    if not run_dirs:
        print("未找到任何 control run，请指定 --outputs 或 --table")
        return

    n_jobs = args.n_jobs
    if n_jobs is None:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)
    print(f"收集 run 数据（n_jobs={n_jobs}, runs={len(run_dirs)}）…")
    adv_df, density_df, clf_df, tail_df = collect_all_run_rows(run_dirs, n_jobs=n_jobs)

    if adv_df.empty:
        print("无有效 run 数据，退出")
        return

    # 保存表
    adv_df.to_csv(args.outdir / "run_level_adv_abs.csv", index=False)
    if not density_df.empty:
        density_df.to_csv(args.outdir / "run_level_adv_abs_density.csv", index=False)
    if not clf_df.empty:
        clf_df.to_csv(args.outdir / "run_level_adv_abs_clf.csv", index=False)
    if not tail_df.empty:
        tail_df.to_csv(args.outdir / "run_level_tail_metrics.csv", index=False)

    # Scaling 模型
    scaling_df = fit_scaling_models(adv_df)
    if not scaling_df.empty:
        scaling_df.to_csv(args.outdir / "scaling_model_summary.csv", index=False)

    # Shift partial
    shift_df = run_shift_partial(adv_df, run_dirs)
    if not shift_df.empty:
        shift_df.to_csv(args.outdir / "shift_partial_summary.csv", index=False)

    # 图
    plot_adv_abs_vs_N(adv_df, args.outdir)
    plot_adv_abs_vs_N_pooled(adv_df, args.outdir)
    if not density_df.empty:
        plot_density_adv_abs_vs_N(density_df, args.outdir)
    if not clf_df.empty:
        plot_clf_adv_abs_vs_N(clf_df, args.outdir)
    if not tail_df.empty:
        plot_tail_top5_vs_N(tail_df, args.outdir)
        plot_fraction_highrisk_vs_N(tail_df, args.outdir)

    print("已写出: run_level_adv_abs.csv, run_level_adv_abs_density.csv, run_level_adv_abs_clf.csv, run_level_tail_metrics.csv,")
    print("        scaling_model_summary.csv, shift_partial_summary.csv,")
    print("        fig_main_adv_abs_vs_N.png, fig_main_adv_abs_vs_N_pooled.png, fig_density_adv_abs_vs_N.png, fig_clf_adv_abs_vs_N.png,")
    print("        fig_tail_top5_vs_N.png, fig_fraction_highrisk_vs_N.png")


if __name__ == "__main__":
    main()
