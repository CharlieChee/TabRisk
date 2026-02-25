#!/usr/bin/env python3
"""
6 组 MIA：2(Naive/Delta) × 3(k-NN/Density/Learned)。k-NN 取 k=1,8,32 为一组。

- k-NN、Density：从原始 synthetic CSV 算（或从已有 pair_metrics 读入时 learned 才可用）。
- Learned：用 pair_metrics 的 FEATURE_CANDIDATES 训练 LR，若 pair_metrics 不存在则先跑 pipeline。

输出：6 组 AUC（Naive k-NN, Naive density, Naive learned(lr), Delta k-NN, Delta density, Delta learned(lr)）。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Learned 攻击器：名称后缀 -> 无参构造器（每次 CV 新建实例）
LEARNED_CLASSIFIERS: List[Tuple[str, Callable[[], Any]]] = [
    ("lr", lambda: LogisticRegression(max_iter=1000, random_state=42)),
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 复用 shadow_pair_metrics 的列划分与 target 加载
from shadow_pair_metrics import (
    _get_round_files_control,
    _load_target_row,
    _split_numeric_categorical,
)


def _get_round_files_member(target_dir: Path) -> List[Tuple[int, Path, Path]]:
    """枚举 target 目录下 member 的 (round_k, in_path, out_path)，排除 control 文件。"""
    in_files = [
        p for p in target_dir.iterdir()
        if p.is_file() and p.name.endswith("_in.csv") and "control" not in p.name
    ]
    out_files = [
        p for p in target_dir.iterdir()
        if p.is_file() and p.name.endswith("_out.csv") and "control" not in p.name
    ]

    def _parse_round(p: Path) -> Optional[int]:
        try:
            core = p.name.split("synthetic_round_", 1)[1]
            num = core.split("_", 1)[0]
            return int(num)
        except Exception:
            return None

    in_map: Dict[int, Path] = {}
    for p in in_files:
        k = _parse_round(p)
        if k is not None:
            in_map[k] = p
    pairs: List[Tuple[int, Path, Path]] = []
    for p in out_files:
        k = _parse_round(p)
        if k is not None and k in in_map:
            pairs.append((k, in_map[k], p))
    pairs.sort(key=lambda x: x[0])
    return pairs

EPS = 1e-12
KNN_K_LIST = [1, 8, 32]
DENSITY_K = 8  # DOMIAS-like 密度估计用的 k

# Learned attacker 与 four_attack_methods 一致的特征列（来自 pair_metrics）
FEATURE_CANDIDATES = [
    "mmd_numeric", "mmd_mixed", "min_dist_in", "min_dist_out", "delta_min_dist",
    "num_mean_diff_mean", "num_mean_diff_max", "num_std_diff_mean", "num_std_diff_max",
    "cat_tv_mean", "cat_tv_max",
]


def _mixed_type_distances(
    target_row: pd.Series,
    df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    mu: Optional[pd.Series] = None,
    sigma: Optional[pd.Series] = None,
) -> np.ndarray:
    """target 到 df 每行的混合型距离，形状 (len(df),)。"""
    n = len(df)
    if n == 0:
        return np.array([], dtype=float)
    if num_cols:
        data_num = df[num_cols].astype(float)
        if mu is None or sigma is None:
            mu_local = data_num.mean()
            sigma_local = data_num.std(ddof=0).replace(0.0, 1.0)
        else:
            mu_local = mu[num_cols]
            sigma_local = sigma[num_cols].replace(0.0, 1.0)
        data_z = ((data_num - mu_local) / sigma_local).to_numpy(dtype=float)
        t_z = ((target_row[num_cols].astype(float) - mu_local) / sigma_local).to_numpy(dtype=float)
        d_num = np.sqrt(((data_z - t_z.reshape(1, -1)) ** 2).sum(axis=1) / max(1, len(num_cols)))
    else:
        d_num = np.zeros(n, dtype=float)
    if cat_cols:
        data_cat = df[cat_cols].astype(str).to_numpy(dtype=object)
        t_cat = target_row[cat_cols].astype(str).to_numpy(dtype=object)
        d_cat = (data_cat != t_cat.reshape(1, -1)).mean(axis=1).astype(float)
    else:
        d_cat = np.zeros(n, dtype=float)
    return (d_num + d_cat).astype(float)


def _avg_dist_k(distances: np.ndarray, k: int) -> float:
    """到最近 k 个点的距离的平均；不足 k 个则用全部。"""
    if distances.size == 0:
        return float("nan")
    k_use = min(k, len(distances))
    topk = np.partition(distances, k_use - 1)[:k_use]
    return float(np.mean(topk))


def _knn_density_at_target(
    distances: np.ndarray,
    k: int,
    d_eff: int,
) -> float:
    """
    DOMIAS 式 k-NN 密度估计：density(x) ∝ k / (r_k^d_eff)。
    返回 log(density)（常数项可略），用于 later 做 log(d_in)-log(d_out)。
    """
    if distances.size < k:
        return float("nan")
    r_k = np.partition(distances, k - 1)[k - 1]
    r_k = max(float(r_k), EPS)
    # log(density) = log(k) - d_eff*log(r_k) + const；这里只保留 -d_eff*log(r_k) 做相对比较
    return -d_eff * np.log(r_k)


def _compute_scores_one_row(
    target_row: pd.Series,
    in_df: pd.DataFrame,
    out_df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    joint_mu: Optional[pd.Series],
    joint_sigma: Optional[pd.Series],
) -> Tuple[Dict, Dict]:
    """
    对单行 (target, in_set, out_set) 计算 k-NN 与 density 所需量。
    返回 (knn_dict, density_dict)。
    knn_dict: "in" -> {k: avg_dist_in_k}, "out" -> {k: avg_dist_out_k}。
    density_dict: "log_density_in", "log_density_out"（DOMIAS 式 k-NN 密度 log）。
    """
    knn_avg_in = {}
    knn_avg_out = {}
    for k in KNN_K_LIST:
        d_in = _mixed_type_distances(target_row, in_df, num_cols, cat_cols, joint_mu, joint_sigma)
        d_out = _mixed_type_distances(target_row, out_df, num_cols, cat_cols, joint_mu, joint_sigma)
        knn_avg_in[k] = _avg_dist_k(d_in, k)
        knn_avg_out[k] = _avg_dist_k(d_out, k)

    d_in = _mixed_type_distances(target_row, in_df, num_cols, cat_cols, joint_mu, joint_sigma)
    d_out = _mixed_type_distances(target_row, out_df, num_cols, cat_cols, joint_mu, joint_sigma)
    d_eff = max(1, len(num_cols) + len(cat_cols))
    k_d = min(DENSITY_K, len(in_df), len(out_df))
    k_d = max(1, k_d)
    log_density_in = _knn_density_at_target(d_in, k_d, d_eff)
    log_density_out = _knn_density_at_target(d_out, k_d, d_eff)

    return (
        {"in": knn_avg_in, "out": knn_avg_out},
        {"log_density_in": log_density_in, "log_density_out": log_density_out},
    )


def _naive_score_knn(avg_in: float, avg_out: float) -> float:
    """Naive k-NN：用 in/out 对比，越大越像 member。d_out - d_in（out 远、in 近则正）。"""
    if np.isnan(avg_in) or np.isnan(avg_out):
        return float("nan")
    return float(avg_out - avg_in)


def _naive_score_density(log_density_in: float, log_density_out: float) -> float:
    """Naive density：log(density_in) - log(density_out)，member 在 in 集密度更高则正。"""
    if np.isnan(log_density_in) or np.isnan(log_density_out):
        return float("nan")
    return float(log_density_in - log_density_out)


def _collect_paired_rows(
    run_dir: Path,
    candidate_df: pd.DataFrame,
) -> Tuple[List[Tuple[int, int, pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]], List[str]]:
    """
    收集所有 (target_idx, round) 且同时有 member in/out 和 control in/out 的配对行。
    返回 list of (target_idx, round, target_row, in_df, out_df, c_in_df, c_out_df)，以及 columns 顺序 list。
    """
    shadow_dir = run_dir / "shadow"
    if not shadow_dir.exists():
        return [], []

    target_indices: List[int] = []
    for d in sorted(shadow_dir.iterdir(), key=lambda x: x.name):
        if not d.is_dir() or not d.name.startswith("target_"):
            continue
        suffix = d.name[7:]
        if suffix.isdigit():
            target_indices.append(int(suffix))

    pairs: List[Tuple[int, int, pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    columns_order: List[str] = []

    for ti in target_indices:
        target_dir = shadow_dir / f"target_{ti}"
        member_rounds = _get_round_files_member(target_dir)
        control_rounds = _get_round_files_control(target_dir)
        if not member_rounds or not control_rounds:
            continue
        try:
            target_row = _load_target_row(run_dir, ti, candidate_df)
        except Exception:
            continue
        control_round_map = {r[0]: (r[1], r[2]) for r in control_rounds}
        for round_k, in_path, out_path in member_rounds:
            if round_k not in control_round_map:
                continue
            c_in_path, c_out_path = control_round_map[round_k]
            in_df = pd.read_csv(in_path)
            out_df = pd.read_csv(out_path)
            out_df = out_df[in_df.columns]
            c_in_df = pd.read_csv(c_in_path)
            c_out_df = pd.read_csv(c_out_path)
            c_out_df = c_out_df[c_in_df.columns]
            if in_df.empty or out_df.empty or c_in_df.empty or c_out_df.empty:
                continue
            if not columns_order:
                columns_order = list(in_df.columns)
            pairs.append((ti, round_k, target_row, in_df, out_df, c_in_df, c_out_df))

    return pairs, columns_order


def _collect_paired_paths(run_dir: Path) -> List[Tuple[int, int, str, str, str, str]]:
    """仅收集 (target_idx, round_k, in_path, out_path, c_in_path, c_out_path) 路径，供多进程 worker 用。"""
    shadow_dir = run_dir / "shadow"
    if not shadow_dir.exists():
        return []
    target_indices = []
    for d in sorted(shadow_dir.iterdir(), key=lambda x: x.name):
        if not d.is_dir() or not d.name.startswith("target_"):
            continue
        suffix = d.name[7:]
        if suffix.isdigit():
            target_indices.append(int(suffix))
    out: List[Tuple[int, int, str, str, str, str]] = []
    for ti in target_indices:
        target_dir = shadow_dir / f"target_{ti}"
        member_rounds = _get_round_files_member(target_dir)
        control_rounds = _get_round_files_control(target_dir)
        if not member_rounds or not control_rounds:
            continue
        control_round_map = {r[0]: (r[1], r[2]) for r in control_rounds}
        for round_k, in_path, out_path in member_rounds:
            if round_k not in control_round_map:
                continue
            c_in_path, c_out_path = control_round_map[round_k]
            out.append((ti, round_k, str(in_path), str(out_path), str(c_in_path), str(c_out_path)))
    return out


def _worker_one_pair(
    run_dir_str: str,
    target_idx: int,
    round_k: int,
    in_path: str,
    out_path: str,
    c_in_path: str,
    c_out_path: str,
) -> Optional[Dict[str, Any]]:
    """
    单 (target, round) 的 worker：读入数据并算 Naive/Delta 的 k-NN 与 density 分数。
    返回 dict: naive_knn {k: (s_m, s_c)}, naive_density (s_m, s_c), delta_knn {k: delta}, delta_density delta。
    """
    run_dir = Path(run_dir_str)
    candidate_df = pd.read_csv(run_dir / "candidate.csv")
    try:
        target_row = _load_target_row(run_dir, target_idx, candidate_df)
    except Exception:
        return None
    in_df = pd.read_csv(in_path)
    out_df = pd.read_csv(out_path)
    out_df = out_df[in_df.columns]
    c_in_df = pd.read_csv(c_in_path)
    c_out_df = pd.read_csv(c_out_path)
    c_out_df = c_out_df[c_in_df.columns]
    if in_df.empty or out_df.empty or c_in_df.empty or c_out_df.empty:
        return None
    num_cols, cat_cols = _split_numeric_categorical(in_df)
    all_num = pd.concat([in_df[num_cols], out_df[num_cols]], axis=0, ignore_index=True).astype(float) if num_cols else None
    joint_mu = all_num.mean() if all_num is not None and len(all_num) else None
    joint_sigma = all_num.std(ddof=0).replace(0.0, 1.0) if all_num is not None and len(all_num) else None
    knn_m, dens_m = _compute_scores_one_row(
        target_row, in_df, out_df, num_cols, cat_cols, joint_mu, joint_sigma
    )
    all_num_c = pd.concat([c_in_df[num_cols], c_out_df[num_cols]], axis=0, ignore_index=True).astype(float) if num_cols else None
    mu_c = all_num_c.mean() if all_num_c is not None and len(all_num_c) else None
    sigma_c = all_num_c.std(ddof=0).replace(0.0, 1.0) if all_num_c is not None and len(all_num_c) else None
    knn_c, dens_c = _compute_scores_one_row(
        target_row, c_in_df, c_out_df, num_cols, cat_cols, mu_c, sigma_c
    )
    naive_knn = {}
    for k in KNN_K_LIST:
        s_m = _naive_score_knn(knn_m["in"][k], knn_m["out"][k])
        s_c = _naive_score_knn(knn_c["in"][k], knn_c["out"][k])
        naive_knn[k] = (s_m, s_c)
    naive_density = (
        _naive_score_density(dens_m["log_density_in"], dens_m["log_density_out"]),
        _naive_score_density(dens_c["log_density_in"], dens_c["log_density_out"]),
    )
    delta_knn = {}
    for k in KNN_K_LIST:
        sig_m = _naive_score_knn(knn_m["in"][k], knn_m["out"][k])
        sig_c = _naive_score_knn(knn_c["in"][k], knn_c["out"][k])
        delta_knn[k] = (sig_m - sig_c) if np.isfinite(sig_m) and np.isfinite(sig_c) else float("nan")
    d_m = _naive_score_density(dens_m["log_density_in"], dens_m["log_density_out"])
    d_c = _naive_score_density(dens_c["log_density_in"], dens_c["log_density_out"])
    delta_density = (d_m - d_c) if np.isfinite(d_m) and np.isfinite(d_c) else float("nan")
    return {
        "naive_knn": naive_knn,
        "naive_density": naive_density,
        "delta_knn": delta_knn,
        "delta_density": delta_density,
    }


def _ensure_pair_metrics(run_dir: Path, n_jobs: int = 4) -> bool:
    """若 run_dir/shadow_pair_metrics 下缺少 pair_metrics.csv 或 pair_metrics_control.csv，则调用 pipeline 生成。"""
    metrics_dir = run_dir / "shadow_pair_metrics"
    pair_path = metrics_dir / "pair_metrics.csv"
    control_path = metrics_dir / "pair_metrics_control.csv"
    if pair_path.exists() and control_path.exists():
        return True
    pipeline = PROJECT_ROOT / "scripts" / "run_shadow_metrics_pipeline.py"
    if not pipeline.exists():
        return False
    cmd = [
        sys.executable,
        str(pipeline),
        "--run-dir", str(run_dir),
        "--n-jobs", str(max(1, n_jobs)),
    ]
    ret = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=3600)
    if ret.returncode != 0:
        return False
    return pair_path.exists() and control_path.exists()


def _compute_learned_aucs(run_dir: Path, cv: int = 5, random_state: int = 42) -> Dict[str, float]:
    """
    从 pair_metrics 计算 Naive/Delta Learned 的 AUC。
    - 按 (target_idx, round) 成对：同一对的两行必须同进 train 或同进 test，用显式按组划分避免泄漏。
    - 先对 merged 打乱顺序，再构造 X/y/groups，避免 group 顺序带来偏差。
    - 标准化用 Pipeline 仅在 train 上拟合。
    """
    out: Dict[str, float] = {}
    pair_path = run_dir / "shadow_pair_metrics" / "pair_metrics.csv"
    control_path = run_dir / "shadow_pair_metrics" / "pair_metrics_control.csv"
    if not pair_path.exists() or not control_path.exists():
        return out
    df_m = pd.read_csv(pair_path)
    df_c = pd.read_csv(control_path)
    if df_m.empty or df_c.empty:
        return out
    id_cols = [c for c in ["target_idx", "round"] if c in df_m.columns and c in df_c.columns]
    if not id_cols:
        for suffix, _ in LEARNED_CLASSIFIERS:
            out[f"auc_naive_learned_{suffix}"] = out[f"auc_delta_learned_{suffix}"] = float("nan")
        return out
    merged = df_m.merge(df_c, on=id_cols, how="inner", suffixes=("_m", "_c"))
    if merged.empty:
        for suffix, _ in LEARNED_CLASSIFIERS:
            out[f"auc_naive_learned_{suffix}"] = out[f"auc_delta_learned_{suffix}"] = float("nan")
        return out
    common = set(df_m.select_dtypes(include=[np.number]).columns) & set(df_c.select_dtypes(include=[np.number]).columns)
    common -= {"target_idx", "round"}
    feature_cols = [x for x in FEATURE_CANDIDATES if x in common] or sorted(common)
    feat_m = [f"{c}_m" for c in feature_cols if f"{c}_m" in merged.columns]
    feat_c = [f"{c}_c" for c in feature_cols if f"{c}_c" in merged.columns]
    if not feat_m or not feat_c:
        for suffix, _ in LEARNED_CLASSIFIERS:
            out[f"auc_naive_learned_{suffix}"] = out[f"auc_delta_learned_{suffix}"] = float("nan")
        return out

    # 打乱 pair 顺序，避免 (target_idx, round) 的原始顺序带来 train/test 分布偏差
    merged = merged.sample(frac=1, random_state=random_state).reset_index(drop=True)
    N = len(merged)
    n_cv = min(cv, max(2, N // 2))

    M = merged[feat_m].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()
    C = merged[feat_c].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()

    # Naive：行 0..N-1 = member 特征，行 N..2N-1 = control 特征；group i 对应 pair i 的两行
    X_naive = np.vstack([M, C])
    y_naive = np.concatenate([np.ones(N), np.zeros(N)])
    groups_naive = np.concatenate([np.arange(N), np.arange(N)])

    # Delta：每对两行 (delta,1) 与 (-delta,0)
    delta = M - C
    X_delta = np.vstack([delta, -delta])
    y_delta = np.concatenate([np.ones(N), np.zeros(N)])
    groups_delta = np.repeat(np.arange(N), 2)

    # 显式按组划分：先打乱 group id，再按 fold 分配，保证同一 group 只出现在 train 或 test 之一
    rng = np.random.default_rng(random_state)
    group_ids = np.arange(N)
    rng.shuffle(group_ids)
    n_per_fold = max(1, N // n_cv)
    splits_naive: List[Tuple[np.ndarray, np.ndarray]] = []
    for f in range(n_cv):
        test_groups = set(
            group_ids[f * n_per_fold : (f + 1) * n_per_fold] if f < n_cv - 1 else group_ids[f * n_per_fold :]
        )
        if f == n_cv - 1 and len(test_groups) == 0:
            test_groups = set(group_ids[(n_cv - 1) * n_per_fold :])
        test_idx = np.array([i for i in range(2 * N) if groups_naive[i] in test_groups], dtype=np.intp)
        train_idx = np.array([i for i in range(2 * N) if groups_naive[i] not in test_groups], dtype=np.intp)
        assert set(groups_naive[train_idx]) & set(groups_naive[test_idx]) == set(), "group leak"
        splits_naive.append((train_idx, test_idx))
    # 若某折 test 为空或单类，则用 GroupKFold 兜底
    if any(len(s[1]) < 4 or len(np.unique(y_naive[s[1]])) < 2 for s in splits_naive):
        gkf = GroupKFold(n_splits=n_cv)
        splits_naive = list(gkf.split(X_naive, y_naive, groups_naive))

    splits_delta: List[Tuple[np.ndarray, np.ndarray]] = []
    for f in range(n_cv):
        test_groups = set(
            group_ids[f * n_per_fold : (f + 1) * n_per_fold] if f < n_cv - 1 else group_ids[f * n_per_fold :]
        )
        if f == n_cv - 1 and len(test_groups) == 0:
            test_groups = set(group_ids[(n_cv - 1) * n_per_fold :])
        test_idx = np.array([i for i in range(2 * N) if groups_delta[i] in test_groups], dtype=np.intp)
        train_idx = np.array([i for i in range(2 * N) if groups_delta[i] not in test_groups], dtype=np.intp)
        assert set(groups_delta[train_idx]) & set(groups_delta[test_idx]) == set(), "group leak"
        splits_delta.append((train_idx, test_idx))
    if any(len(s[1]) < 4 or len(np.unique(y_delta[s[1]])) < 2 for s in splits_delta):
        gkf = GroupKFold(n_splits=n_cv)
        splits_delta = list(gkf.split(X_delta, y_delta, groups_delta))

    def _cv_auc_splits(
        X: np.ndarray,
        y: np.ndarray,
        splits: List[Tuple[np.ndarray, np.ndarray]],
        suffix: str,
        factory: Callable[[], Any],
        prefix_naive: bool,
    ) -> None:
        key = f"auc_{'naive' if prefix_naive else 'delta'}_learned_{suffix}"
        if len(splits) < 2:
            out[key] = float("nan")
            return
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", factory())])
        res = cross_validate(pipe, X, y, cv=splits, scoring="roc_auc", return_train_score=False)
        mean_auc = float(np.mean(res["test_score"]))
        out[key] = max(mean_auc, 1.0 - mean_auc)

    for suffix, factory in LEARNED_CLASSIFIERS:
        _cv_auc_splits(X_naive, y_naive, splits_naive, suffix, factory, prefix_naive=True)
    for suffix, factory in LEARNED_CLASSIFIERS:
        _cv_auc_splits(X_delta, y_delta, splits_delta, suffix, factory, prefix_naive=False)
    return out


def _run_one_run(
    run_dir: Path, n_jobs: int = 1, export_scores_path: Optional[Path] = None
) -> Dict[str, float]:
    """
    对单个 run_dir 计算 4 类方法的 AUC（k-NN 含 k=1,8,32）。
    若 export_scores_path 指定，则写入每个 (target_idx, round) 的 Naive/Delta 分数供画图用。
    返回 dict: auc_naive_knn_k1, ...
    """
    run_dir = Path(run_dir).resolve()
    candidate_path = run_dir / "candidate.csv"
    if not candidate_path.exists():
        return {}
    run_dir_str = str(run_dir)
    path_list = _collect_paired_paths(run_dir)
    if not path_list:
        return {}

    naive_knn_scores: Dict[int, List[float]] = {k: [] for k in KNN_K_LIST}
    naive_knn_labels: Dict[int, List[int]] = {k: [] for k in KNN_K_LIST}
    naive_density_scores: List[float] = []
    naive_density_labels: List[int] = []
    delta_knn_scores: Dict[int, List[float]] = {k: [] for k in KNN_K_LIST}
    delta_knn_labels: Dict[int, List[int]] = {k: [] for k in KNN_K_LIST}
    delta_density_scores: List[float] = []
    delta_density_labels: List[int] = []

    export_rows: List[Dict[str, Any]] = []

    def _process_one(res: Optional[Dict[str, Any]], ti: Optional[int] = None, rk: Optional[int] = None) -> None:
        if res is None:
            return
        for k in KNN_K_LIST:
            s_m, s_c = res["naive_knn"][k]
            if np.isfinite(s_m):
                naive_knn_scores[k].append(s_m)
                naive_knn_labels[k].append(1)
            if np.isfinite(s_c):
                naive_knn_scores[k].append(s_c)
                naive_knn_labels[k].append(0)
        sd_m, sd_c = res["naive_density"]
        if np.isfinite(sd_m):
            naive_density_scores.append(sd_m)
            naive_density_labels.append(1)
        if np.isfinite(sd_c):
            naive_density_scores.append(sd_c)
            naive_density_labels.append(0)
        for k in KNN_K_LIST:
            delta = res["delta_knn"][k]
            if np.isfinite(delta):
                delta_knn_scores[k].append(delta)
                delta_knn_labels[k].append(1)
                delta_knn_scores[k].append(-delta)
                delta_knn_labels[k].append(0)
        delta_d = res["delta_density"]
        if np.isfinite(delta_d):
            delta_density_scores.append(delta_d)
            delta_density_labels.append(1)
            delta_density_scores.append(-delta_d)
            delta_density_labels.append(0)
        if export_scores_path is not None and ti is not None and rk is not None:
            s_m8, s_c8 = res["naive_knn"].get(8, (float("nan"), float("nan")))
            export_rows.append({
                "target_idx": ti,
                "round": rk,
                "naive_knn_k8_member": s_m8,
                "naive_knn_k8_control": s_c8,
                "delta_knn_k8": res["delta_knn"].get(8, float("nan")),
                "naive_density_member": res["naive_density"][0],
                "naive_density_control": res["naive_density"][1],
                "delta_density": res["delta_density"],
            })

    if n_jobs <= 1:
        for ti, rk, in_p, out_p, c_in_p, c_out_p in path_list:
            res = _worker_one_pair(run_dir_str, ti, rk, in_p, out_p, c_in_p, c_out_p)
            _process_one(res, ti, rk)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = {
                ex.submit(_worker_one_pair, run_dir_str, ti, rk, in_p, out_p, c_in_p, c_out_p): (ti, rk)
                for ti, rk, in_p, out_p, c_in_p, c_out_p in path_list
            }
            for fut in as_completed(futures):
                ti, rk = futures[fut]
                _process_one(fut.result(), ti, rk)

    def _auc(scores: List[float], labels: List[int]) -> float:
        if len(scores) < 2 or len(set(labels)) < 2:
            return float("nan")
        s = np.array(scores, dtype=float)
        l = np.array(labels, dtype=int)
        auc = roc_auc_score(l, s)
        return max(float(auc), 1.0 - float(auc))

    out: Dict[str, float] = {}
    for k in KNN_K_LIST:
        out[f"auc_naive_knn_k{k}"] = _auc(naive_knn_scores[k], naive_knn_labels[k])
        out[f"auc_delta_knn_k{k}"] = _auc(delta_knn_scores[k], delta_knn_labels[k])
    out["auc_naive_density"] = _auc(naive_density_scores, naive_density_labels)
    out["auc_delta_density"] = _auc(delta_density_scores, delta_density_labels)

    if export_scores_path and export_rows:
        export_scores_path = Path(export_scores_path)
        export_scores_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(export_rows).to_csv(export_scores_path, index=False)

    # Learned：若 pair_metrics 不存在则先跑 pipeline，再算 Naive/Delta × 多分类器 AUC
    learned_suffixes = [s for s, _ in LEARNED_CLASSIFIERS]
    if _ensure_pair_metrics(run_dir, n_jobs=n_jobs):
        learned = _compute_learned_aucs(run_dir, cv=5)
        out.update(learned)
    else:
        for s in learned_suffixes:
            out[f"auc_naive_learned_{s}"] = float("nan")
            out[f"auc_delta_learned_{s}"] = float("nan")

    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="仅用影子模型原始合成数据做 4 种 MIA (Naive/Delta × k-NN/Density)，输出 AUC。",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="run 目录，如 train_adult_..._control_seed42_20260221_030832",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="可选：输出 CSV 路径；不指定则只打印",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="并行进程数（默认 1）；例如 32 则用 32 进程处理 (target, round) 对",
    )
    parser.add_argument(
        "--export-scores",
        type=str,
        default=None,
        help="可选：导出每个 (target, round) 的 Naive/Delta 分数到 CSV，供 demo 画分布图",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    if not run_dir.exists():
        print(f"错误: run_dir 不存在: {run_dir}", file=sys.stderr)
        return 1

    export_path = Path(args.export_scores) if args.export_scores else None
    result = _run_one_run(run_dir, n_jobs=args.n_jobs, export_scores_path=export_path)
    if not result:
        print("未找到有效的 (target, round) 配对（需同时有 member in/out 与 control in/out）", file=sys.stderr)
        return 1

    learned_suffixes = [s for s, _ in LEARNED_CLASSIFIERS]
    print("AUC (6 组: Naive/Delta × k-NN(k=1,8,32)/density/learned(lr)):")
    groups = [
        ("1. Naive k-NN (k=1,8,32)", [f"auc_naive_knn_k{k}" for k in KNN_K_LIST]),
        ("2. Naive density", ["auc_naive_density"]),
        ("3. Naive learned (lr)", [f"auc_naive_learned_{s}" for s in learned_suffixes]),
        ("4. Delta k-NN (k=1,8,32)", [f"auc_delta_knn_k{k}" for k in KNN_K_LIST]),
        ("5. Delta density", ["auc_delta_density"]),
        ("6. Delta learned (lr)", [f"auc_delta_learned_{s}" for s in learned_suffixes]),
    ]
    for label, keys in groups:
        vals = [result.get(k, float("nan")) for k in keys]
        print(f"  {label}: " + ", ".join(f"{v:.4f}" for v in vals))
    print("  (逐键):")
    for key in sorted(result.keys()):
        print(f"    {key}: {result[key]:.4f}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([result]).to_csv(out_path, index=False)
        print(f"已写入: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
