#!/usr/bin/env python3
"""
仅用影子模型原始合成数据（无 pair_metrics）做 4 种 MIA：2(Naive/Delta) × 2(k-NN/Density)。

思路：
- Naive：不用 control 数据参与**算分**；member/control 两组各自用本组 in/out 算分，用两组得分+标签算 AUC。
- Delta：差分信号。同一 (target, round) 上 member 信号 - control 信号，减去「N-1 伴随集带来的基线」；
  member 行得分 = member_signal - control_signal，control 行得分 = control_signal - member_signal，再算 AUC。
- k-NN：target 到 in 集 / out 集的最近 k 条距离的平均（k=1,8,32）；得分用 in/out 对比（见下）。
- Density：DOMIAS 思路，用 k-NN 密度估计（合成集在 target 处的密度），再 in vs out 比。

用法：
  python scripts/shadow_mia_from_raw.py --run-dir outputs/train_adult_..._control_seed42_20260221_030832
  输出：4 类 AUC（Naive k-NN k=1/8/32, Naive density, Delta k-NN k=1/8/32, Delta density）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

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


def _run_one_run(run_dir: Path) -> Dict[str, float]:
    """
    对单个 run_dir 计算 4 类方法的 AUC（k-NN 含 k=1,8,32）。
    返回 dict: auc_naive_knn_k1, auc_naive_knn_k8, auc_naive_knn_k32, auc_naive_density,
               auc_delta_knn_k1, auc_delta_knn_k8, auc_delta_knn_k32, auc_delta_density。
    """
    run_dir = Path(run_dir).resolve()
    candidate_path = run_dir / "candidate.csv"
    if not candidate_path.exists():
        return {}
    candidate_df = pd.read_csv(candidate_path)

    pairs, _ = _collect_paired_rows(run_dir, candidate_df)
    if not pairs:
        return {}

    # 为 Naive 准备：每个 (target, round) 有 member 一行、control 一行，用「本行 in/out」算分
    naive_knn_scores: Dict[int, List[float]] = {k: [] for k in KNN_K_LIST}
    naive_knn_labels: Dict[int, List[int]] = {k: [] for k in KNN_K_LIST}
    naive_density_scores: List[float] = []
    naive_density_labels: List[int] = []

    delta_knn_scores: Dict[int, List[float]] = {k: [] for k in KNN_K_LIST}
    delta_knn_labels: Dict[int, List[int]] = {k: [] for k in KNN_K_LIST}
    delta_density_scores: List[float] = []
    delta_density_labels: List[int] = []

    for target_idx, round_k, target_row, in_df, out_df, c_in_df, c_out_df in pairs:
        num_cols, cat_cols = _split_numeric_categorical(in_df)
        # 联合标准化：member 用 in+out，control 用 c_in+c_out；为 delta 对齐用同一套特征
        all_num = pd.concat([in_df[num_cols], out_df[num_cols]], axis=0, ignore_index=True).astype(float) if num_cols else None
        joint_mu = all_num.mean() if all_num is not None and len(all_num) else None
        joint_sigma = all_num.std(ddof=0).replace(0.0, 1.0) if all_num is not None and len(all_num) else None

        # Member 行：用 (in_df, out_df) 算分
        knn_m, dens_m = _compute_scores_one_row(
            target_row, in_df, out_df, num_cols, cat_cols, joint_mu, joint_sigma
        )
        # Control 行：用 (c_in_df, c_out_df) 算分，用 control 自己的联合统计
        all_num_c = pd.concat([c_in_df[num_cols], c_out_df[num_cols]], axis=0, ignore_index=True).astype(float) if num_cols else None
        mu_c = all_num_c.mean() if all_num_c is not None and len(all_num_c) else None
        sigma_c = all_num_c.std(ddof=0).replace(0.0, 1.0) if all_num_c is not None and len(all_num_c) else None
        knn_c, dens_c = _compute_scores_one_row(
            target_row, c_in_df, c_out_df, num_cols, cat_cols, mu_c, sigma_c
        )

        # Naive：member 得分、control 得分，不混用对方数据
        for k in KNN_K_LIST:
            s_m = _naive_score_knn(knn_m["in"][k], knn_m["out"][k])
            s_c = _naive_score_knn(knn_c["in"][k], knn_c["out"][k])
            if np.isfinite(s_m):
                naive_knn_scores[k].append(s_m)
                naive_knn_labels[k].append(1)
            if np.isfinite(s_c):
                naive_knn_scores[k].append(s_c)
                naive_knn_labels[k].append(0)
        sd_m = _naive_score_density(dens_m["log_density_in"], dens_m["log_density_out"])
        sd_c = _naive_score_density(dens_c["log_density_in"], dens_c["log_density_out"])
        if np.isfinite(sd_m):
            naive_density_scores.append(sd_m)
            naive_density_labels.append(1)
        if np.isfinite(sd_c):
            naive_density_scores.append(sd_c)
            naive_density_labels.append(0)

        # Delta：差分 = member_signal - control_signal；member 行得 delta，control 行得 -delta
        for k in KNN_K_LIST:
            sig_m = _naive_score_knn(knn_m["in"][k], knn_m["out"][k])
            sig_c = _naive_score_knn(knn_c["in"][k], knn_c["out"][k])
            delta = sig_m - sig_c if np.isfinite(sig_m) and np.isfinite(sig_c) else float("nan")
            if np.isfinite(delta):
                delta_knn_scores[k].append(delta)
                delta_knn_labels[k].append(1)
                delta_knn_scores[k].append(-delta)
                delta_knn_labels[k].append(0)
        d_m = _naive_score_density(dens_m["log_density_in"], dens_m["log_density_out"])
        d_c = _naive_score_density(dens_c["log_density_in"], dens_c["log_density_out"])
        delta_d = d_m - d_c if np.isfinite(d_m) and np.isfinite(d_c) else float("nan")
        if np.isfinite(delta_d):
            delta_density_scores.append(delta_d)
            delta_density_labels.append(1)
            delta_density_scores.append(-delta_d)
            delta_density_labels.append(0)

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
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    if not run_dir.exists():
        print(f"错误: run_dir 不存在: {run_dir}", file=sys.stderr)
        return 1

    result = _run_one_run(run_dir)
    if not result:
        print("未找到有效的 (target, round) 配对（需同时有 member in/out 与 control in/out）", file=sys.stderr)
        return 1

    print("AUC (4 类方法, k-NN 取 k=1,8,32):")
    for key in sorted(result.keys()):
        print(f"  {key}: {result[key]:.4f}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([result]).to_csv(out_path, index=False)
        print(f"已写入: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
