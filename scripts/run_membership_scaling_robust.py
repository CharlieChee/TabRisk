#!/usr/bin/env python3
"""
局部攻击族稳健性与一致性测试：Density 多 k、Classifier 多模型，及核心判据汇总。

PART A: k ∈ {1,3,5,10} 的 density_score_k，run_level_density_multi_k.csv，fig_density_multi_k_scaling.png
PART B: LR / LinearSVM / XGBoost(RF fallback)，特征 ΔNN, Δlog_r1, Δlog_r3, Δlog_r5, min_dist_in/out
PART C: 汇总判据打印

依赖：shadow_pair_metrics 的 _get_round_files, _get_round_files_control, _load_target_row,
      _split_numeric_categorical, _mixed_type_distances_all；run 目录含 shadow/target_*/synthetic_round_*_in/out.csv
      及 pair_metrics.csv, pair_metrics_control.csv, candidate.csv。
"""

from __future__ import annotations

import argparse
import multiprocessing
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from shadow_pair_metrics import (
    _get_round_files,
    _get_round_files_control,
    _load_target_row,
    _split_numeric_categorical,
    _mixed_type_distances_all,
)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from sklearn.ensemble import RandomForestClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DENSITY_K_LIST = [1, 3, 5, 10]
DENSITY_EPS = 1e-10
CLF_CV = 5
RANDOM_STATE = 42


# ---------- run 发现（与 run_membership_scaling_analysis 一致） ----------
def _is_ddpm_run(dirname: str) -> bool:
    return "ddpm" in dirname.lower() or "tabddpm" in dirname.lower()


def _is_ctgan_run(dirname: str) -> bool:
    return "ctgan" in dirname.lower()


def parse_run_dir(dirname: str) -> Dict[str, Any]:
    out = {"train_size": None, "model": None, "seed": None, "rounds": None}
    m = re.search(r"_train(\d+)_synth", dirname)
    if m:
        out["train_size"] = int(m.group(1))
    m = re.search(r"_seed(\d+)(?:_|$)", dirname)
    if m:
        out["seed"] = int(m.group(1))
    m = re.search(r"_rounds(\d+)_", f"_{dirname}_")
    if m:
        out["rounds"] = int(m.group(1))
    if _is_ddpm_run(dirname):
        out["model"] = "ddpm"
    elif _is_ctgan_run(dirname):
        out["model"] = "ctgan"
    return out


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
        if (d / "shadow_pair_metrics" / "pair_metrics.csv").exists():
            run_dirs.append(d)
    for sub in outputs_dir.iterdir():
        if not sub.is_dir() or sub.name.startswith("."):
            continue
        for d in sub.iterdir():
            if not d.is_dir() or "control" not in d.name:
                continue
            if (d / "shadow_pair_metrics" / "pair_metrics.csv").exists():
                run_dirs.append(d)
    return sorted(set(run_dirs), key=lambda p: (p.name, str(p)))


# ---------- PART A: k-NN density ----------
def _r_k_sorted(distances: np.ndarray, k: int) -> float:
    """第 k 近邻距离（1-based）。不足 k 个则返回 nan。"""
    d = np.asarray(distances, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < k:
        return float("nan")
    return float(np.sort(d)[k - 1])


def _compute_density_scores_one_run(
    run_dir: Path,
    candidate_df: pd.DataFrame,
    pair_path: Path,
    pair_control_path: Path,
    k_list: List[int],
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Optional[Dict[int, np.ndarray]], Optional[Dict[int, np.ndarray]]]:
    """
    对 run 计算 member 与 control 的 density_score_k 序列（按 (target_idx, round) 顺序）。
    返回 (member_scores_by_k, control_scores_by_k, member_features, control_features)。
    member_scores_by_k[k] = 1d array of density_score_k for each member row.
    member_features = dict with keys 'delta_NN','delta_log_r1','delta_log_r3','delta_log_r5','min_dist_in','min_dist_out' -> arrays.
    """
    member_scores_by_k: Dict[int, List[float]] = {k: [] for k in k_list}
    control_scores_by_k: Dict[int, List[float]] = {k: [] for k in k_list}
    member_feat: Dict[str, List[float]] = {
        "delta_NN": [], "delta_log_r1": [], "delta_log_r3": [], "delta_log_r5": [],
        "min_dist_in": [], "min_dist_out": [],
    }
    control_feat: Dict[str, List[float]] = {k: [] for k in member_feat}

    if not pair_path.exists():
        return (
            {k: np.array(member_scores_by_k[k]) for k in k_list},
            {k: np.array(control_scores_by_k[k]) for k in k_list},
            None,
            None,
        )
    df_m = pd.read_csv(pair_path)
    if df_m.empty:
        return (
            {k: np.array(member_scores_by_k[k]) for k in k_list},
            {k: np.array(control_scores_by_k[k]) for k in k_list},
            None,
            None,
        )

    # 用 pair_metrics 的行顺序：(target_idx, round)
    for _, row in df_m.iterrows():
        target_idx = int(row["target_idx"])
        round_k = int(row["round"])
        target_dir = run_dir / "shadow" / f"target_{target_idx}"
        in_path = target_dir / f"synthetic_round_{round_k}_in.csv"
        out_path = target_dir / f"synthetic_round_{round_k}_out.csv"
        if not in_path.exists() or not out_path.exists():
            continue
        try:
            in_df = pd.read_csv(in_path)
            out_df = pd.read_csv(out_path)
            out_df = out_df[in_df.columns]
        except Exception:
            continue
        num_cols, cat_cols = _split_numeric_categorical(in_df)
        if num_cols:
            all_num = pd.concat([in_df[num_cols], out_df[num_cols]], axis=0, ignore_index=True).astype(float)
            mu_joint = all_num.mean()
            sigma_joint = all_num.std(ddof=0).replace(0.0, 1.0)
        else:
            mu_joint = sigma_joint = None
        try:
            target_row = _load_target_row(run_dir, target_idx, candidate_df)
        except Exception:
            continue
        dist_in = _mixed_type_distances_all(target_row, in_df, num_cols, cat_cols, mu_joint, sigma_joint)
        dist_out = _mixed_type_distances_all(target_row, out_df, num_cols, cat_cols, mu_joint, sigma_joint)
        n_in, n_out = len(dist_in), len(dist_out)
        r_1_in = _r_k_sorted(dist_in, 1)
        r_1_out = _r_k_sorted(dist_out, 1)
        r_3_in = _r_k_sorted(dist_in, 3)
        r_3_out = _r_k_sorted(dist_out, 3)
        r_5_in = _r_k_sorted(dist_in, 5)
        r_5_out = _r_k_sorted(dist_out, 5)
        for k in k_list:
            rk_in = _r_k_sorted(dist_in, k) if n_in >= k else float("nan")
            rk_out = _r_k_sorted(dist_out, k) if n_out >= k else float("nan")
            if np.isfinite(rk_in) and np.isfinite(rk_out) and rk_in >= DENSITY_EPS and rk_out >= DENSITY_EPS:
                sc = np.log(rk_in + DENSITY_EPS) - np.log(rk_out + DENSITY_EPS)
                member_scores_by_k[k].append(sc)
            else:
                member_scores_by_k[k].append(float("nan"))
        member_feat["min_dist_in"].append(r_1_in if np.isfinite(r_1_in) else np.nan)
        member_feat["min_dist_out"].append(r_1_out if np.isfinite(r_1_out) else np.nan)
        member_feat["delta_NN"].append((r_1_out - r_1_in) if np.isfinite(r_1_in) and np.isfinite(r_1_out) else np.nan)
        for name, rin, rout in [
            ("delta_log_r1", r_1_in, r_1_out),
            ("delta_log_r3", r_3_in, r_3_out),
            ("delta_log_r5", r_5_in, r_5_out),
        ]:
            if np.isfinite(rin) and np.isfinite(rout) and rin >= DENSITY_EPS and rout >= DENSITY_EPS:
                member_feat[name].append(np.log(rin + DENSITY_EPS) - np.log(rout + DENSITY_EPS))
            else:
                member_feat[name].append(np.nan)

    # Control
    if not pair_control_path.exists():
        return (
            {k: np.array(member_scores_by_k[k]) for k in k_list},
            {k: np.array(control_scores_by_k[k]) for k in k_list},
            {k: np.array(v) for k, v in member_feat.items()},
            None,
        )
    df_c = pd.read_csv(pair_control_path)
    if df_c.empty:
        return (
            {k: np.array(member_scores_by_k[k]) for k in k_list},
            {k: np.array(control_scores_by_k[k]) for k in k_list},
            {k: np.array(v) for k, v in member_feat.items()},
            None,
        )
    for _, row in df_c.iterrows():
        target_idx = int(row["target_idx"])
        round_k = int(row["round"])
        target_dir = run_dir / "shadow" / f"target_{target_idx}"
        in_path = target_dir / f"synthetic_round_{round_k}_control_in.csv"
        out_path = target_dir / f"synthetic_round_{round_k}_control_out.csv"
        if not in_path.exists() or not out_path.exists():
            continue
        try:
            in_df = pd.read_csv(in_path)
            out_df = pd.read_csv(out_path)
            out_df = out_df[in_df.columns]
        except Exception:
            continue
        num_cols, cat_cols = _split_numeric_categorical(in_df)
        if num_cols:
            all_num = pd.concat([in_df[num_cols], out_df[num_cols]], axis=0, ignore_index=True).astype(float)
            mu_joint = all_num.mean()
            sigma_joint = all_num.std(ddof=0).replace(0.0, 1.0)
        else:
            mu_joint = sigma_joint = None
        try:
            target_row = _load_target_row(run_dir, target_idx, candidate_df)
        except Exception:
            continue
        dist_in = _mixed_type_distances_all(target_row, in_df, num_cols, cat_cols, mu_joint, sigma_joint)
        dist_out = _mixed_type_distances_all(target_row, out_df, num_cols, cat_cols, mu_joint, sigma_joint)
        n_in, n_out = len(dist_in), len(dist_out)
        r_1_in = _r_k_sorted(dist_in, 1)
        r_1_out = _r_k_sorted(dist_out, 1)
        r_3_in = _r_k_sorted(dist_in, 3)
        r_3_out = _r_k_sorted(dist_out, 3)
        r_5_in = _r_k_sorted(dist_in, 5)
        r_5_out = _r_k_sorted(dist_out, 5)
        for k in k_list:
            rk_in = _r_k_sorted(dist_in, k) if n_in >= k else float("nan")
            rk_out = _r_k_sorted(dist_out, k) if n_out >= k else float("nan")
            if np.isfinite(rk_in) and np.isfinite(rk_out) and rk_in >= DENSITY_EPS and rk_out >= DENSITY_EPS:
                sc = np.log(rk_in + DENSITY_EPS) - np.log(rk_out + DENSITY_EPS)
                control_scores_by_k[k].append(sc)
            else:
                control_scores_by_k[k].append(float("nan"))
        control_feat["min_dist_in"].append(r_1_in if np.isfinite(r_1_in) else np.nan)
        control_feat["min_dist_out"].append(r_1_out if np.isfinite(r_1_out) else np.nan)
        control_feat["delta_NN"].append((r_1_out - r_1_in) if np.isfinite(r_1_in) and np.isfinite(r_1_out) else np.nan)
        for name, rin, rout in [
            ("delta_log_r1", r_1_in, r_1_out),
            ("delta_log_r3", r_3_in, r_3_out),
            ("delta_log_r5", r_5_in, r_5_out),
        ]:
            if np.isfinite(rin) and np.isfinite(rout) and rin >= DENSITY_EPS and rout >= DENSITY_EPS:
                control_feat[name].append(np.log(rin + DENSITY_EPS) - np.log(rout + DENSITY_EPS))
            else:
                control_feat[name].append(np.nan)

    member_arrays = {k: np.array(member_scores_by_k[k], dtype=float) for k in k_list}
    control_arrays = {k: np.array(control_scores_by_k[k], dtype=float) for k in k_list}
    return (
        member_arrays,
        control_arrays,
        {k: np.array(v, dtype=float) for k, v in member_feat.items()},
        {k: np.array(v, dtype=float) for k, v in control_feat.items()},
    )


def _adv_abs_density_for_k(member_scores: np.ndarray, control_scores: np.ndarray) -> float:
    m = member_scores[np.isfinite(member_scores)]
    c = control_scores[np.isfinite(control_scores)]
    if m.size == 0 or c.size == 0:
        return float("nan")
    scores = np.concatenate([m, c])
    labels = np.concatenate([np.ones(len(m)), np.zeros(len(c))])
    auc = _auc_from_scores_labels(scores, labels)
    if np.isnan(auc):
        return float("nan")
    return float(abs(auc - 0.5))


# ---------- PART B: Classifiers ----------
def _build_X_y(member_feat: Dict[str, np.ndarray], control_feat: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """特征顺序: delta_NN, delta_log_r1, delta_log_r3, delta_log_r5, min_dist_in, min_dist_out。"""
    keys = ["delta_NN", "delta_log_r1", "delta_log_r3", "delta_log_r5", "min_dist_in", "min_dist_out"]
    Xm = np.column_stack([np.nan_to_num(member_feat[k], nan=0.0) for k in keys])
    Xc = np.column_stack([np.nan_to_num(control_feat[k], nan=0.0) for k in keys])
    X = np.vstack([Xm, Xc])
    y = np.concatenate([np.ones(len(Xm)), np.zeros(len(Xc))])
    return X, y


def _adv_abs_clf_one(X: np.ndarray, y: np.ndarray, clf_type: str) -> float:
    if X.shape[0] < 10 or np.unique(y).size < 2:
        return float("nan")
    Xs = StandardScaler().fit_transform(X)
    try:
        if clf_type == "LR":
            clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
            proba = cross_val_predict(clf, Xs, y, cv=min(CLF_CV, len(y) // 2), method="predict_proba")[:, 1]
        elif clf_type == "LinearSVM":
            base = LinearSVC(max_iter=5000, random_state=RANDOM_STATE)
            clf = CalibratedClassifierCV(base, cv=min(3, len(y) // 3))
            proba = cross_val_predict(clf, Xs, y, cv=min(CLF_CV, len(y) // 2), method="predict_proba")[:, 1]
        elif clf_type == "XGBoost" and HAS_XGB:
            clf = xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=RANDOM_STATE, use_label_encoder=False, eval_metric="logloss")
            proba = cross_val_predict(clf, Xs, y, cv=min(CLF_CV, len(y) // 2), method="predict_proba")[:, 1]
        elif clf_type == "XGBoost" or clf_type == "RandomForest":
            clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=RANDOM_STATE)
            proba = cross_val_predict(clf, Xs, y, cv=min(CLF_CV, len(y) // 2), method="predict_proba")[:, 1]
        else:
            return float("nan")
        auc = _auc_from_scores_labels(proba, y)
        return float(abs(auc - 0.5)) if np.isfinite(auc) else float("nan")
    except Exception:
        return float("nan")


# ---------- Worker：单 run 计算 PART A + B ----------
def _worker_one_run_robust(run_dir_str: str) -> Optional[Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]]:
    """返回 (parsed, density_rows, classifier_rows) 或 None。"""
    run_dir = Path(run_dir_str)
    parsed = parse_run_dir(run_dir.name)
    if not parsed.get("model"):
        return None
    metrics_dir = run_dir / "shadow_pair_metrics"
    pair_path = metrics_dir / "pair_metrics.csv"
    pair_control_path = metrics_dir / "pair_metrics_control.csv"
    candidate_path = run_dir / "candidate.csv"
    if not candidate_path.exists():
        return None
    candidate_df = pd.read_csv(candidate_path)
    run_id = run_dir.name

    k_list = [k for k in DENSITY_K_LIST if k <= 10]
    member_by_k, control_by_k, member_feat, control_feat = _compute_density_scores_one_run(
        run_dir, candidate_df, pair_path, pair_control_path, k_list
    )
    density_rows = []
    for k in k_list:
        m = member_by_k[k]
        c = control_by_k[k]
        adv = _adv_abs_density_for_k(m, c)
        density_rows.append({
            "run_id": run_id,
            "model": parsed["model"],
            "train_size": parsed["train_size"],
            "k": k,
            "adv_abs_density": adv if np.isfinite(adv) else np.nan,
        })

    classifier_rows = []
    if member_feat and control_feat:
        X, y = _build_X_y(member_feat, control_feat)
        for clf_type in ["LR", "LinearSVM", "XGBoost"]:
            adv_c = _adv_abs_clf_one(X, y, clf_type)
            classifier_rows.append({
                "run_id": run_id,
                "model": parsed["model"],
                "train_size": parsed["train_size"],
                "clf_type": clf_type if (clf_type != "XGBoost" or HAS_XGB) else "RandomForest",
                "adv_abs_clf": adv_c if np.isfinite(adv_c) else np.nan,
            })
    return (parsed, density_rows, classifier_rows)


# ---------- 主流程：收集 + 表 + 图 + 打印 ----------
def collect_density_and_classifier_multi(
    run_dirs: List[Path],
    n_jobs: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    run_dir_strs = [str(p.resolve()) for p in run_dirs]
    if n_jobs is None or n_jobs <= 1:
        results = []
        for s in run_dir_strs:
            try:
                r = _worker_one_run_robust(s)
                if r is not None:
                    results.append(r)
            except Exception:
                pass
    else:
        with multiprocessing.Pool(processes=n_jobs) as pool:
            results = [r for r in pool.imap(_worker_one_run_robust, run_dir_strs, chunksize=1) if r is not None]
    density_rows = []
    classifier_rows = []
    for _parsed, dr, cr in results:
        density_rows.extend(dr)
        classifier_rows.extend(cr)
    return pd.DataFrame(density_rows), pd.DataFrame(classifier_rows)


def _slope_r2_logN(df: pd.DataFrame, value_col: str, group_col: str) -> Tuple[Optional[float], Optional[float]]:
    """按 group_col 分组，对 log(train_size) vs value_col 线性回归，返回 (slope, r2) 的均值或单一值。"""
    df = df.dropna(subset=["train_size", value_col])
    df = df[df["train_size"] > 0]
    if df.empty or df["train_size"].nunique() < 2:
        return None, None
    df = df.copy()
    df["logN"] = np.log(df["train_size"].astype(float))
    slopes, r2s = [], []
    for _g, g in df.groupby(group_col):
        if len(g) < 3:
            continue
        x = g["logN"].to_numpy(dtype=float)
        y = g[value_col].to_numpy(dtype=float)
        if np.isfinite(x).all() and np.isfinite(y).all():
            try:
                slope, intercept, r, p, se = stats.linregress(x, y)
                slopes.append(slope)
                r2s.append(r ** 2)
            except Exception:
                pass
    if not slopes:
        return None, None
    return float(np.mean(slopes)), float(np.mean(r2s))


def run_part_a_print(density_df: pd.DataFrame) -> None:
    """A6: 不同 k 的 scaling slope 同号、R² 一致、是否都随 N 下降。"""
    if density_df.empty:
        return
    print("\n========== PART A 判断 ==========")
    k_vals = sorted(density_df["k"].unique())
    slopes = []
    r2s = []
    for k in k_vals:
        sub = density_df[density_df["k"] == k].dropna(subset=["train_size", "adv_abs_density"])
        if sub.empty or sub["train_size"].nunique() < 2:
            slopes.append(np.nan)
            r2s.append(np.nan)
            continue
        sub = sub.copy()
        sub["logN"] = np.log(sub["train_size"].astype(float))
        try:
            slope, _, r, _, _ = stats.linregress(sub["logN"], sub["adv_abs_density"])
            slopes.append(slope)
            r2s.append(r ** 2)
        except Exception:
            slopes.append(np.nan)
            r2s.append(np.nan)
    slopes = np.array(slopes)
    r2s = np.array(r2s)
    valid_s = slopes[np.isfinite(slopes)]
    same_sign = np.all(valid_s <= 0) or np.all(valid_s >= 0) if len(valid_s) > 1 else True
    print(f"  不同 k 下 scaling slope 同号: {same_sign} (slopes by k: {dict(zip(k_vals, slopes))})")
    print(f"  不同 k 下 R²: {dict(zip(k_vals, r2s))}")
    all_decrease = np.all(valid_s <= 0) if len(valid_s) > 0 else None
    print(f"  是否所有 k 都随 N 下降 (slope<=0): {all_decrease}")


def run_part_b_print(clf_df: pd.DataFrame) -> None:
    """B5: scaling slope、是否一致随 N 下降、最强、差异显著性。"""
    if clf_df.empty:
        return
    print("\n========== PART B 判断 ==========")
    clf_types = sorted(clf_df["clf_type"].unique())
    slopes = {}
    for ct in clf_types:
        sub = clf_df[clf_df["clf_type"] == ct].dropna(subset=["train_size", "adv_abs_clf"])
        if sub.empty or sub["train_size"].nunique() < 2:
            slopes[ct] = np.nan
            continue
        sub = sub.copy()
        sub["logN"] = np.log(sub["train_size"].astype(float))
        try:
            slope, _, _, _, _ = stats.linregress(sub["logN"], sub["adv_abs_clf"])
            slopes[ct] = slope
        except Exception:
            slopes[ct] = np.nan
    print(f"  各 classifier scaling slope: {slopes}")
    valid_s = np.array([s for s in slopes.values() if np.isfinite(s)])
    all_decrease = np.all(valid_s <= 0) if len(valid_s) > 0 else None
    print(f"  是否一致随 N 下降: {all_decrease}")
    mean_adv = clf_df.groupby("clf_type")["adv_abs_clf"].mean()
    strongest = mean_adv.idxmax() if not mean_adv.empty else None
    print(f"  平均 adv_abs_clf 最强: {strongest} (means: {mean_adv.to_dict()})")
    if len(clf_types) >= 2:
        try:
            sub = clf_df.dropna(subset=["adv_abs_clf"])
            if len(sub) >= 6:
                from scipy.stats import f_oneway
                groups = [sub[sub["clf_type"] == ct]["adv_abs_clf"].values for ct in clf_types]
                groups = [g[np.isfinite(g)] for g in groups if len(g) > 0]
                if len(groups) >= 2:
                    f_stat, p_anova = f_oneway(*groups)
                    print(f"  ANOVA 不同 classifier 差异 p_value: {p_anova:.4f}")
        except Exception as e:
            print(f"  ANOVA 未计算: {e}")


def run_part_c_print(density_df: pd.DataFrame, clf_df: pd.DataFrame) -> None:
    """PART C 核心判据汇总。"""
    print("\n========== PART C 核心判据 ==========")
    # 1) 是否所有 density k 的 adv_abs 随 N 下降
    all_density_decrease = None
    if not density_df.empty:
        k_vals = sorted(density_df["k"].unique())
        slopes = []
        for k in k_vals:
            sub = density_df[density_df["k"] == k].dropna(subset=["train_size", "adv_abs_density"])
            if sub.empty or sub["train_size"].nunique() < 2:
                continue
            sub = sub.copy()
            sub["logN"] = np.log(sub["train_size"].astype(float))
            try:
                s, _, _, _, _ = stats.linregress(sub["logN"], sub["adv_abs_density"])
                slopes.append(s)
            except Exception:
                pass
        all_density_decrease = np.all(np.array(slopes) <= 0) if slopes else None
    print(f"  1) 是否所有 density k 的 adv_abs 随 N 下降: {all_density_decrease}")

    # 2) 是否所有 classifier 类型的 adv_abs 随 N 下降
    all_clf_decrease = None
    if not clf_df.empty:
        slopes = []
        for ct in clf_df["clf_type"].unique():
            sub = clf_df[clf_df["clf_type"] == ct].dropna(subset=["train_size", "adv_abs_clf"])
            if sub.empty or sub["train_size"].nunique() < 2:
                continue
            sub = sub.copy()
            sub["logN"] = np.log(sub["train_size"].astype(float))
            try:
                s, _, _, _, _ = stats.linregress(sub["logN"], sub["adv_abs_clf"])
                slopes.append(s)
            except Exception:
                pass
        all_clf_decrease = np.all(np.array(slopes) <= 0) if slopes else None
    print(f"  2) 是否所有 classifier 类型的 adv_abs 随 N 下降: {all_clf_decrease}")

    # 3) 不同攻击族 slope 方向一致
    density_slope = None
    if not density_df.empty:
        sub = density_df.dropna(subset=["train_size", "adv_abs_density"])
        if sub["train_size"].nunique() >= 2:
            sub = sub.copy()
            sub["logN"] = np.log(sub["train_size"].astype(float))
            try:
                density_slope, _, _, _, _ = stats.linregress(sub["logN"], sub["adv_abs_density"])
            except Exception:
                pass
    clf_slope = None
    if not clf_df.empty:
        sub = clf_df.dropna(subset=["train_size", "adv_abs_clf"])
        if sub["train_size"].nunique() >= 2:
            sub = sub.copy()
            sub["logN"] = np.log(sub["train_size"].astype(float))
            try:
                clf_slope, _, _, _, _ = stats.linregress(sub["logN"], sub["adv_abs_clf"])
            except Exception:
                pass
    same_direction = None
    if density_slope is not None and clf_slope is not None:
        same_direction = (density_slope <= 0 and clf_slope <= 0) or (density_slope >= 0 and clf_slope >= 0)
    print(f"  3) 不同攻击族 slope 方向一致 (density slope={density_slope}, clf slope={clf_slope}): {same_direction}")

    # 4) 大 N 时三类攻击都趋近 0
    mean_d, mean_c, large_n = np.nan, np.nan, None
    if not density_df.empty and not clf_df.empty:
        all_n = np.unique(np.concatenate([density_df["train_size"].values, clf_df["train_size"].values]))
        threshold_n = np.percentile(all_n.astype(float), 75) if len(all_n) > 0 else np.nan
        if np.isfinite(threshold_n):
            sub_d = density_df[density_df["train_size"] >= threshold_n]["adv_abs_density"].dropna()
            sub_c = clf_df[clf_df["train_size"] >= threshold_n]["adv_abs_clf"].dropna()
            mean_d = float(sub_d.mean()) if len(sub_d) > 0 else np.nan
            mean_c = float(sub_c.mean()) if len(sub_c) > 0 else np.nan
            large_n = np.isfinite(mean_d) and np.isfinite(mean_c) and mean_d < 0.1 and mean_c < 0.1
    print(f"  4) 大 N 时 adv_abs 趋近 0 (大 N 段均值): density ~ {mean_d}, clf ~ {mean_c}; 均<0.1: {large_n}")

    if all_density_decrease and all_clf_decrease:
        print("\n  → 结论: 支持「membership signal 随 N 衰减」。")
    else:
        print("\n  → 若有攻击不随 N 衰减，需单独报告并对比。")


def plot_density_multi_k(density_df: pd.DataFrame, outdir: Path) -> None:
    """fig_density_multi_k_scaling.png: 横轴 log(N), 纵轴 adv_abs_density, 每个 k 一条线，按 model 分面。"""
    density_df = density_df.dropna(subset=["train_size", "adv_abs_density"]).copy()
    if density_df.empty:
        return
    models = sorted(density_df["model"].unique())
    k_vals = sorted(density_df["k"].unique())
    nmodels = len(models)
    fig, axes = plt.subplots(1, nmodels, figsize=(5 * nmodels, 5))
    if nmodels == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        sub = density_df[density_df["model"] == model]
        for k in k_vals:
            sk = sub[sub["k"] == k]
            if sk.empty:
                continue
            agg = sk.groupby("train_size", as_index=False)["adv_abs_density"].mean()
            ax.plot(agg["train_size"], agg["adv_abs_density"], "o-", label=f"k={k}")
        ax.set_xscale("log")
        ax.set_xlabel("train_size N (log)")
        ax.set_ylabel("adv_abs_density")
        ax.set_title(f"model={model}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(outdir / "fig_density_multi_k_scaling.png", dpi=200)
    plt.close(fig)


def plot_classifier_multi_model(clf_df: pd.DataFrame, outdir: Path) -> None:
    """fig_classifier_multi_model_scaling.png: 横轴 log(N), 纵轴 adv_abs_clf, 每种 classifier 一条线。"""
    clf_df = clf_df.dropna(subset=["train_size", "adv_abs_clf"]).copy()
    if clf_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for clf_type in sorted(clf_df["clf_type"].unique()):
        sub = clf_df[clf_df["clf_type"] == clf_type]
        agg = sub.groupby(["model", "train_size"], as_index=False)["adv_abs_clf"].mean()
        for model in agg["model"].unique():
            sm = agg[agg["model"] == model]
            ax.plot(sm["train_size"], sm["adv_abs_clf"], "o-", label=f"{clf_type} ({model})")
    ax.set_xscale("log")
    ax.set_xlabel("train_size N (log)")
    ax.set_ylabel("adv_abs_clf")
    ax.set_title("Classifier multi-model scaling")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(outdir / "fig_classifier_multi_model_scaling.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="局部攻击族稳健性: density 多 k, classifier 多模型")
    parser.add_argument("--outputs", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "outputs" / "figures")
    parser.add_argument("--n_jobs", type=int, default=None)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    n_jobs = args.n_jobs if args.n_jobs is not None else max(1, multiprocessing.cpu_count() - 1)

    run_dirs = scan_control_runs(args.outputs)
    if not run_dirs:
        print("未找到 control run，请指定 --outputs")
        return
    print(f"找到 {len(run_dirs)} runs, n_jobs={n_jobs}")
    density_df, clf_df = collect_density_and_classifier_multi(run_dirs, n_jobs=n_jobs)

    if not density_df.empty:
        density_df.to_csv(args.outdir / "run_level_density_multi_k.csv", index=False)
        run_part_a_print(density_df)
        plot_density_multi_k(density_df, args.outdir)
    if not clf_df.empty:
        clf_df.to_csv(args.outdir / "run_level_classifier_multi_model.csv", index=False)
        run_part_b_print(clf_df)
        plot_classifier_multi_model(clf_df, args.outdir)
    run_part_c_print(density_df, clf_df)
    print("\n已写出: run_level_density_multi_k.csv, run_level_classifier_multi_model.csv,")
    print("        fig_density_multi_k_scaling.png, fig_classifier_multi_model_scaling.png")


if __name__ == "__main__":
    main()
