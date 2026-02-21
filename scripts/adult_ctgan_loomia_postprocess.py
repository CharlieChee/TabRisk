#!/usr/bin/env python3
"""
Adult + CTGAN LOO-MIA 后处理（端到端）。

产出论文可用的结果包：
- 统一 run-level 总表（去重 + 完整性检查）
- 三组平衡子集：scaling_balanced / rounds_balanced / iter_balanced
- 图：Fig1 Adv vs train size, Fig2 Adv vs rounds, Fig3 Adv vs iter, Fig4 Adv vs stability, 可选 Fig5 learned attacker
- Table 1：Adv(AUC) vs train size
- postprocess_report.txt

用法:
  python scripts/adult_ctgan_loomia_postprocess.py --outputs outputs --outdir outputs/postprocess_adult_ctgan
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_validate

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 必须存在的文件（缺一即标记 incomplete）
REQUIRED_FILES = ["pair_metrics.csv", "pair_metrics_control.csv", "mmd_component_analysis.json"]
# 可选：control_comparison.json（用于附录指标，不参与完整性）

# 主线 Group A 固定参数
GROUP_A_N_ITER = 1000
GROUP_A_ROUNDS = 20
GROUP_A_CANDIDATE = 100
GROUP_A_TRAIN_ROWS = [200, 500, 1000, 1500, 2000]

# Group B rounds 饱和
GROUP_B_TRAIN_ROWS = 500
GROUP_B_N_ITER = 1000
GROUP_B_ROUNDS = [5, 10, 20, 40, 80, 160]

# Group C iter 强度
GROUP_C_TRAIN_ROWS = 500
GROUP_C_ROUNDS = 20
GROUP_C_N_ITER = [200, 300, 600, 1000, 2000, 4000]

# 主线平衡 seed 集合（K=5）
MAINLINE_SEEDS = [42, 43, 44, 45, 46]

# delta 列优先级（用于 ROC score）
DELTA_COL_CANDIDATES = ("delta_min_dist_mean", "delta_min_dist", "delta_min_dist_median", "delta_mean")

# Learned attacker 特征列
FEATURE_CANDIDATES = [
    "mmd_numeric", "mmd_mixed", "min_dist_in", "min_dist_out", "delta_min_dist",
    "num_mean_diff_mean", "num_mean_diff_max", "num_std_diff_mean", "num_std_diff_max",
    "cat_tv_mean", "cat_tv_max",
]


def parse_run_dir(dirname: str) -> Dict[str, Any]:
    """
    从 run_dir 名称解析：train_rows, n_iter, batch_size, rounds_per_target, candidate, seed, timestamp。
    目录名示例: train_adult_openml_..._train2000_synth2000_ctgan_1000iter_bs256_rounds20_candidate100_control_seed42_20260221_181549
    """
    out = {
        "train_rows": None,
        "n_iter": None,
        "batch_size": None,
        "rounds_per_target": None,
        "candidate": None,
        "seed": None,
        "timestamp": None,
    }
    m = re.search(r"_train(\d+)_synth", dirname)
    if m:
        out["train_rows"] = int(m.group(1))
    m = re.search(r"_(\d+)iter", dirname)
    if m:
        out["n_iter"] = int(m.group(1))
    m = re.search(r"_bs(\d+)(?:_|$)", dirname)
    if m:
        out["batch_size"] = int(m.group(1))
    m = re.search(r"_rounds(\d+)_", f"_{dirname}_")
    if m:
        out["rounds_per_target"] = int(m.group(1))
    m = re.search(r"_candidate(\d+)(?:_|$)", dirname)
    if m:
        out["candidate"] = int(m.group(1))
    m = re.search(r"_seed(\d+)(?:_|$)", dirname)
    if m:
        out["seed"] = int(m.group(1))
    m = re.search(r"_(\d{8}_\d{6})$", dirname)
    if m:
        out["timestamp"] = m.group(1)
    return out


def run_complete(metrics_dir: Path) -> bool:
    """run 是否文件齐全（必须文件都存在）。"""
    if not metrics_dir.is_dir():
        return False
    return all((metrics_dir / f).exists() for f in REQUIRED_FILES)


def config_key(row: Dict[str, Any]) -> Tuple:
    """不含 seed 的 config_key：(model=ctgan, preprocess=monotonic, train_rows, n_iter, batch_size, rounds, candidate)。"""
    return (
        row.get("train_rows"),
        row.get("n_iter"),
        row.get("batch_size"),
        row.get("rounds_per_target"),
        row.get("candidate"),
    )


def run_key(row: Dict[str, Any]) -> Tuple:
    """含 seed 的 run_key：config_key + seed。"""
    return config_key(row) + (row.get("seed"),)


def load_json_safe(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def choose_delta_column(df: pd.DataFrame) -> Optional[str]:
    """优先 delta_min_dist_mean，否则自动选 delta 相关列。"""
    delta_cols = [c for c in df.columns if "delta" in c.lower()]
    if not delta_cols:
        return None
    for cand in DELTA_COL_CANDIDATES:
        if cand in df.columns:
            return cand
    return delta_cols[0]


def compute_auc_adv_ci(
    pair_path: Path,
    pair_control_path: Path,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[str]]:
    """
    member=pair_metrics (label=1), non-member=pair_metrics_control (label=0)。
    score 用 delta 列；返回 (auc_in_out, adv_auc, adv_auc_ci95_low, adv_auc_ci95_high, error_msg)。
    """
    if not pair_path.exists():
        return None, None, None, None, "missing pair_metrics.csv"
    if not pair_control_path.exists():
        return None, None, None, None, "missing pair_metrics_control.csv"
    df_m = pd.read_csv(pair_path)
    df_c = pd.read_csv(pair_control_path)
    if df_m.empty or df_c.empty:
        return None, None, None, None, "empty pair or control CSV"
    delta_col = choose_delta_column(df_m) or choose_delta_column(df_c)
    if not delta_col or delta_col not in df_m.columns or delta_col not in df_c.columns:
        return None, None, None, None, "missing delta column"
    scores_m = df_m[delta_col].astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    scores_c = df_c[delta_col].astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if scores_m.size == 0 or scores_c.size == 0:
        return None, None, None, None, "no valid delta scores"
    scores = np.concatenate([scores_m, scores_c])
    labels = np.concatenate([np.ones(len(scores_m)), np.zeros(len(scores_c))])
    n = len(labels)
    if np.unique(labels).size < 2:
        return None, None, None, None, "single class"
    try:
        auc = float(roc_auc_score(labels, scores))
    except Exception:
        return None, None, None, None, "AUC computation failed"
    adv = auc - 0.5
    rng = np.random.default_rng(seed)
    boot_adv: List[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        a = roc_auc_score(labels[idx], scores[idx])
        if not np.isnan(a):
            boot_adv.append(a - 0.5)
    boot_arr = np.array(boot_adv)
    if boot_arr.size == 0:
        ci_low, ci_high = adv, adv
    else:
        ci_low = float(np.percentile(boot_arr, 2.5))
        ci_high = float(np.percentile(boot_arr, 97.5))
    return auc, adv, ci_low, ci_high, None


def _process_one_run(run_dir_str: str) -> Optional[Dict[str, Any]]:
    """
    单 run 处理（供多进程调用）：解析 + AUC/CI + stability，返回 row 或 None。
    参数为 run_dir 的字符串路径以便 pickle。
    """
    run_dir = Path(run_dir_str)
    metrics_dir = run_dir / "shadow_pair_metrics"
    parsed = parse_run_dir(run_dir.name)
    if parsed.get("rounds_per_target") is None:
        parsed["rounds_per_target"] = _get_rounds_from_config(run_dir)
    if parsed.get("seed") is None:
        parsed["seed"] = _get_seed_from_config(run_dir)

    pair_path = metrics_dir / "pair_metrics.csv"
    pair_control_path = metrics_dir / "pair_metrics_control.csv"
    mmd_path = metrics_dir / "mmd_component_analysis.json"
    ts = parsed.get("timestamp") or ""
    mtime = pair_path.stat().st_mtime if pair_path.exists() else 0

    auc_val, adv_val, ci_low, ci_high, err = compute_auc_adv_ci(pair_path, pair_control_path)
    if err:
        return None
    stab_val, stab_name = extract_stability(mmd_path)

    return {
        "train_rows": parsed.get("train_rows"),
        "n_iter": parsed.get("n_iter"),
        "batch_size": parsed.get("batch_size"),
        "rounds": parsed.get("rounds_per_target"),
        "candidate": parsed.get("candidate"),
        "seed": parsed.get("seed"),
        "timestamp": ts,
        "run_dir": run_dir.name,
        "run_dir_path": str(run_dir.resolve()),
        "auc_in_out": auc_val,
        "adv_auc": adv_val,
        "adv_auc_ci95_low": ci_low,
        "adv_auc_ci95_high": ci_high,
        "stability_metric": stab_val if stab_val is not None else np.nan,
        "stability_metric_name": stab_name if stab_name else "",
        "_mtime": mtime,
    }


def _learned_auc_worker(run_dir_path: str) -> Optional[float]:
    """单 run 的 learned AUC（供多进程调用）。参数为路径字符串。"""
    auc, _ = compute_learned_auc_one_run(run_dir_path)
    return auc


def extract_stability(mmd_path: Path) -> Tuple[Optional[float], Optional[str]]:
    """
    从 mmd_component_analysis.json 提取 stability_metric。
    若存在 overall/total 类键则用；否则对所有数值取 mean，记录来源名。
    """
    if not mmd_path.exists():
        return None, None
    data = load_json_safe(mmd_path)
    if not data:
        return None, None
    scalars: List[Tuple[str, float]] = []

    def collect(d: Any, prefix: str) -> None:
        if isinstance(d, dict):
            for k, v in d.items():
                collect(v, f"{prefix}{k}_")
        elif isinstance(d, (int, float)) and not isinstance(d, bool):
            scalars.append((prefix.rstrip("_"), float(d)))
        elif isinstance(d, list) and d and not isinstance(d[0], (dict, list)):
            for i, x in enumerate(d):
                if isinstance(x, (int, float)) and not isinstance(x, bool):
                    scalars.append((f"{prefix}{i}", float(x)))

    collect(data, "")
    priority = ("mmd_total", "total_mmd", "mmd_overall", "overall", "total", "mean_mixed_minus_numeric")
    for key in priority:
        for name, val in scalars:
            if key in name.lower() or name.lower() == key:
                return val, name
    if not scalars:
        return None, None
    return float(np.mean([v for _, v in scalars])), "mean_of_components"


def scan_and_dedup(outputs_dir: Path, n_jobs: int = 1) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    扫描 outputs 下含 shadow_pair_metrics 的 run（名称需含 adult 与 control），
    解析元信息、计算 run-level 指标，按 run_key 去重（保留时间戳最新）。
    n_jobs>1 时用多进程并行处理每个 run。
    返回 (summary_df, report_stats)。
    """
    outputs_dir = outputs_dir.resolve()
    report: Dict[str, Any] = {
        "total_scanned_dirs": 0,
        "has_metrics_dir": 0,
        "incomplete": 0,
        "complete_before_dedup": 0,
        "duplicate_dropped": 0,
        "final_runs": 0,
    }
    candidates: List[Path] = []
    for d in outputs_dir.iterdir():
        if not d.is_dir():
            continue
        report["total_scanned_dirs"] += 1
        if "adult" not in d.name.lower() or "control" not in d.name.lower():
            continue
        metrics_dir = d / "shadow_pair_metrics"
        if not metrics_dir.is_dir():
            continue
        report["has_metrics_dir"] += 1
        if not run_complete(metrics_dir):
            report["incomplete"] += 1
            continue
        candidates.append(d)

    report["complete_before_dedup"] = len(candidates)
    run_dir_strs = [str(r.resolve()) for r in candidates]

    if n_jobs <= 1 or len(run_dir_strs) == 0:
        rows: List[Dict[str, Any]] = []
        for run_dir in candidates:
            row = _process_one_run(str(run_dir.resolve()))
            if row is not None:
                rows.append(row)
    else:
        rows = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for result in executor.map(_process_one_run, run_dir_strs):
                if result is not None:
                    rows.append(result)

    if not rows:
        df = pd.DataFrame()
        report["final_runs"] = 0
        return df, report

    df = pd.DataFrame(rows)
    # 去重：同一 run_key 只保留时间戳最新（或 _mtime 最大）的一条
    df["_sort"] = df["timestamp"].fillna("") + "_" + df["_mtime"].astype(str)
    dedup = (
        df.sort_values("_sort", ascending=False)
        .groupby(
            ["train_rows", "n_iter", "batch_size", "rounds", "candidate", "seed"],
            dropna=False,
        )
        .first()
        .reset_index()
    )
    report["duplicate_dropped"] = len(df) - len(dedup)
    report["final_runs"] = len(dedup)
    dedup = dedup.drop(columns=["_sort", "_mtime"], errors="ignore")
    return dedup, report


def _get_rounds_from_config(run_dir: Path) -> Optional[int]:
    cfg = run_dir / "train_config.yaml"
    if not cfg.is_file():
        return None
    try:
        from omegaconf import OmegaConf
        v = OmegaConf.select(OmegaConf.load(cfg), "shadow.num_shadow_rounds", default=None)
        return int(v) if v is not None else None
    except Exception:
        return None


def _get_seed_from_config(run_dir: Path) -> Optional[int]:
    cfg = run_dir / "train_config.yaml"
    if not cfg.is_file():
        return None
    try:
        from omegaconf import OmegaConf
        v = OmegaConf.select(OmegaConf.load(cfg), "shadow.random_seed", default=None)
        return int(v) if v is not None else None
    except Exception:
        return None


def build_balanced_subsets(summary: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Group A: n_iter=1000, rounds=20, candidate=100, train_rows in {200,500,1000,1500,2000}, K=5 seeds (42–46).
    Group B: train_rows=500, n_iter=1000, rounds in {5,10,20,40,80,160}, 每 rounds 取 Smin 个 seed.
    Group C: train_rows=500, rounds=20, n_iter in {200,...,4000}, 每 n_iter 取 Imin 个 seed.
    返回 (scaling_balanced, rounds_balanced, iter_balanced, balance_report).
    """
    balance_report: Dict[str, Any] = {"group_a_seeds_used": [], "group_b_smin": None, "group_c_imin": None}

    # A: 主线 scaling；每个 train_rows 必须用相同数量的 seed（K=5 或更小）
    a = summary[
        (summary["n_iter"] == GROUP_A_N_ITER)
        & (summary["rounds"] == GROUP_A_ROUNDS)
        & ((summary["candidate"] == GROUP_A_CANDIDATE) | summary["candidate"].isna())
        & (summary["train_rows"].isin(GROUP_A_TRAIN_ROWS))
    ].copy()
    if not a.empty:
        # 只考虑 MAINLINE_SEEDS 中出现的 seed
        a = a[a["seed"].isin(MAINLINE_SEEDS)]
        # 每个 train_rows 有多少个 MAINLINE_SEEDS 中的 seed
        seed_counts_per_tr = a.groupby("train_rows")["seed"].nunique()
        if len(seed_counts_per_tr) == 0:
            scaling_balanced = pd.DataFrame()
        else:
            k = int(seed_counts_per_tr.min())
            use_seeds = MAINLINE_SEEDS[:k]
            balance_report["group_a_seeds_used"] = use_seeds
            a = a[a["seed"].isin(use_seeds)]
            scaling_balanced = a.drop_duplicates(subset=["train_rows", "seed"], keep="first")
    else:
        scaling_balanced = pd.DataFrame()

    # B: rounds 饱和
    b = summary[
        (summary["train_rows"] == GROUP_B_TRAIN_ROWS)
        & (summary["n_iter"] == GROUP_B_N_ITER)
        & (summary["rounds"].isin(GROUP_B_ROUNDS))
    ].copy()
    if not b.empty:
        smin = int(b.groupby("rounds")["seed"].nunique().min())
        balance_report["group_b_smin"] = smin
        keep_seeds_per_rounds = b.groupby("rounds")["seed"].apply(lambda x: sorted(x.unique())[:smin]).to_dict()
        rows_b = []
        for r, seeds in keep_seeds_per_rounds.items():
            sub = b[(b["rounds"] == r) & (b["seed"].isin(seeds))]
            sub = sub.drop_duplicates(subset=["seed"], keep="first")
            rows_b.append(sub)
        rounds_balanced = pd.concat(rows_b, ignore_index=True) if rows_b else pd.DataFrame()
    else:
        rounds_balanced = pd.DataFrame()
        balance_report["group_b_smin"] = 0

    # C: iter 强度
    c = summary[
        (summary["train_rows"] == GROUP_C_TRAIN_ROWS)
        & (summary["rounds"] == GROUP_C_ROUNDS)
        & (summary["n_iter"].isin(GROUP_C_N_ITER))
    ].copy()
    if not c.empty:
        imin = int(c.groupby("n_iter")["seed"].nunique().min())
        balance_report["group_c_imin"] = imin
        keep_seeds_per_iter = c.groupby("n_iter")["seed"].apply(lambda x: sorted(x.unique())[:imin]).to_dict()
        rows_c = []
        for it, seeds in keep_seeds_per_iter.items():
            sub = c[(c["n_iter"] == it) & (c["seed"].isin(seeds))]
            sub = sub.drop_duplicates(subset=["seed"], keep="first")
            rows_c.append(sub)
        iter_balanced = pd.concat(rows_c, ignore_index=True) if rows_c else pd.DataFrame()
    else:
        iter_balanced = pd.DataFrame()
        balance_report["group_c_imin"] = 0

    return scaling_balanced, rounds_balanced, iter_balanced, balance_report


def compute_learned_auc_one_run(run_dir_path: str, cv: int = 5) -> Tuple[Optional[float], Optional[str]]:
    """单 run：pair_metrics (label=1) + pair_metrics_control (label=0)，LR 5-fold CV AUC。"""
    run_dir = Path(run_dir_path)
    m = run_dir / "shadow_pair_metrics"
    pair_path = m / "pair_metrics.csv"
    pair_control_path = m / "pair_metrics_control.csv"
    if not pair_path.exists() or not pair_control_path.exists():
        return None, "missing pair CSV"
    df_m = pd.read_csv(pair_path)
    df_c = pd.read_csv(pair_control_path)
    if df_m.empty or df_c.empty:
        return None, "empty CSV"
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
        return None, "no valid rows"
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
    return float(np.mean(res["test_score"])), None


# ---------- 作图 ----------
def fig1_adv_vs_train(scaling_balanced: pd.DataFrame, out_path: Path, jitter_seed: int = 42) -> None:
    """Fig 1: x=train_rows, y=adv_auc；每 seed 散点（jitter）+ mean + errorbar；y=0 虚线。"""
    df = scaling_balanced.dropna(subset=["train_rows", "adv_auc"]).copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    rng = np.random.default_rng(jitter_seed)
    x_vals = df["train_rows"].astype(float)
    y_vals = df["adv_auc"].astype(float)
    x_range = (x_vals.max() - x_vals.min()) or 1.0
    jitter = (rng.random(len(x_vals)) - 0.5) * x_range * 0.03
    ax.scatter(x_vals + jitter, y_vals, alpha=0.7, s=50, c="C0", edgecolors="k", linewidths=0.5, label="per seed")
    for tr in sorted(df["train_rows"].unique()):
        block = df[df["train_rows"] == tr]["adv_auc"]
        mean_y = block.mean()
        std_y = block.std()
        if pd.isna(std_y) or std_y <= 0:
            std_y = 0
        ax.errorbar(tr, mean_y, yerr=std_y, fmt="s", color="C1", markersize=10, capsize=4, capthick=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Train size")
    ax.set_ylabel("Adv(AUC) = AUC − 0.5")
    ax.set_title("Adv(AUC) vs Train Size (Scaling)")
    ax.set_xticks(sorted(df["train_rows"].unique()))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig2_adv_vs_rounds(rounds_balanced: pd.DataFrame, out_path: Path, jitter_seed: int = 42) -> None:
    """Fig 2: x=rounds, y=adv_auc；seed 散点 + mean±std；y=0 虚线。"""
    df = rounds_balanced.dropna(subset=["rounds", "adv_auc"]).copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    rng = np.random.default_rng(jitter_seed)
    x_vals = df["rounds"].astype(float)
    y_vals = df["adv_auc"].astype(float)
    x_range = (x_vals.max() - x_vals.min()) or 1.0
    jitter = (rng.random(len(x_vals)) - 0.5) * x_range * 0.03
    ax.scatter(x_vals + jitter, y_vals, alpha=0.7, s=50, c="C0", edgecolors="k", linewidths=0.5)
    for r in sorted(df["rounds"].unique()):
        block = df[df["rounds"] == r]["adv_auc"]
        mean_y = block.mean()
        std_y = block.std()
        if pd.isna(std_y) or std_y <= 0:
            std_y = 0
        ax.errorbar(r, mean_y, yerr=std_y, fmt="s", color="C1", markersize=10, capsize=4, capthick=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Shadow rounds per target")
    ax.set_ylabel("Adv(AUC)")
    ax.set_title("Adv(AUC) vs Rounds (Saturation)")
    ax.set_xticks(sorted(df["rounds"].unique()))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig3_adv_vs_iter(iter_balanced: pd.DataFrame, out_path: Path, jitter_seed: int = 42) -> None:
    """Fig 3: x=n_iter, y=adv_auc；seed 散点 + mean±std；y=0 虚线。"""
    df = iter_balanced.dropna(subset=["n_iter", "adv_auc"]).copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    rng = np.random.default_rng(jitter_seed)
    x_vals = df["n_iter"].astype(float)
    y_vals = df["adv_auc"].astype(float)
    x_range = (x_vals.max() - x_vals.min()) or 1.0
    jitter = (rng.random(len(x_vals)) - 0.5) * x_range * 0.02
    ax.scatter(x_vals + jitter, y_vals, alpha=0.7, s=50, c="C0", edgecolors="k", linewidths=0.5)
    for it in sorted(df["n_iter"].unique()):
        block = df[df["n_iter"] == it]["adv_auc"]
        mean_y = block.mean()
        std_y = block.std()
        if pd.isna(std_y) or std_y <= 0:
            std_y = 0
        ax.errorbar(it, mean_y, yerr=std_y, fmt="s", color="C1", markersize=10, capsize=4, capthick=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Training iterations (n_iter)")
    ax.set_ylabel("Adv(AUC)")
    ax.set_title("Adv(AUC) vs Training Iterations")
    ax.set_xticks(sorted(df["n_iter"].unique()))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig4_adv_vs_stability(scaling_balanced: pd.DataFrame, out_path: Path) -> None:
    """Fig 4: x=stability_metric, y=adv_auc；颜色=train_rows；线性拟合，Pearson r & Spearman ρ + p。"""
    df = scaling_balanced.dropna(subset=["stability_metric", "adv_auc"]).copy()
    if df.empty or len(df) < 2:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    train_sizes = sorted(df["train_rows"].unique())
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, max(len(train_sizes), 1)))
    for i, ts in enumerate(train_sizes):
        sub = df[df["train_rows"] == ts]
        ax.scatter(
            sub["stability_metric"], sub["adv_auc"],
            c=[colors[i % len(colors)]], s=60, alpha=0.8, label=f"train_rows={int(ts)}",
        )
    x = df["stability_metric"].to_numpy(dtype=float)
    y = df["adv_auc"].to_numpy(dtype=float)
    if x.size > 2 and np.isfinite(x).all() and np.isfinite(y).all():
        slope, intercept, r_pearson, p_pearson, _ = stats.linregress(x, y)
        rho, p_spearman = stats.spearmanr(x, y)
        xx = np.linspace(x.min(), x.max(), 100)
        ax.plot(xx, slope * xx + intercept, "k-", linewidth=1.5, label="linear fit")
        ax.set_title(
            f"Adv(AUC) vs Stability\nPearson r = {r_pearson:.3f} (p = {p_pearson:.4f}), Spearman ρ = {rho:.3f} (p = {p_spearman:.4f})"
        )
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Stability metric")
    ax.set_ylabel("Adv(AUC)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig5a_learned_scaling(df_with_learned: pd.DataFrame, out_path: Path) -> None:
    """Fig 5A: learned_adv vs train_rows（主线）。"""
    df = df_with_learned.dropna(subset=["train_rows", "learned_adv"]).copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    rng = np.random.default_rng(42)
    x_vals = df["train_rows"].astype(float)
    y_vals = df["learned_adv"].astype(float)
    jitter = (rng.random(len(x_vals)) - 0.5) * (x_vals.max() - x_vals.min() or 1) * 0.03
    ax.scatter(x_vals + jitter, y_vals, alpha=0.7, s=50, c="C0", edgecolors="k", linewidths=0.5)
    for tr in sorted(df["train_rows"].unique()):
        block = df[df["train_rows"] == tr]["learned_adv"]
        mean_y = block.mean()
        std_y = block.std()
        if pd.isna(std_y) or std_y <= 0:
            std_y = 0
        ax.errorbar(tr, mean_y, yerr=std_y, fmt="s", color="C1", markersize=10, capsize=4, capthick=2)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Train size")
    ax.set_ylabel("Learned attacker (5-fold CV AUC)")
    ax.set_title("Learned Attacker vs Train Size (Scaling)")
    ax.set_xticks(sorted(df["train_rows"].unique()))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig5b_learned_saturation(df_with_learned: pd.DataFrame, out_path: Path) -> None:
    """Fig 5B: learned_adv vs rounds（饱和）。"""
    df = df_with_learned.dropna(subset=["rounds", "learned_adv"]).copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    rng = np.random.default_rng(42)
    x_vals = df["rounds"].astype(float)
    y_vals = df["learned_adv"].astype(float)
    jitter = (rng.random(len(x_vals)) - 0.5) * (x_vals.max() - x_vals.min() or 1) * 0.02
    ax.scatter(x_vals + jitter, y_vals, alpha=0.7, s=50, c="C0", edgecolors="k", linewidths=0.5)
    for r in sorted(df["rounds"].unique()):
        block = df[df["rounds"] == r]["learned_adv"]
        mean_y = block.mean()
        std_y = block.std()
        if pd.isna(std_y) or std_y <= 0:
            std_y = 0
        ax.errorbar(r, mean_y, yerr=std_y, fmt="s", color="C1", markersize=10, capsize=4, capthick=2)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Shadow rounds per target")
    ax.set_ylabel("Learned attacker (5-fold CV AUC)")
    ax.set_title("Learned Attacker vs Rounds (Saturation)")
    ax.set_xticks(sorted(df["rounds"].unique()))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_table1(scaling_balanced: pd.DataFrame, out_path: Path) -> None:
    """Table 1: 按 train_rows 聚合 mean adv_auc, std, 95% CI (seed-level), n_seeds。"""
    df = scaling_balanced.dropna(subset=["train_rows", "adv_auc"]).copy()
    if df.empty:
        return
    rows = []
    for tr in sorted(df["train_rows"].unique()):
        block = df[df["train_rows"] == tr]["adv_auc"]
        mean_adv = block.mean()
        std_adv = block.std()
        n_seeds = len(block)
        if n_seeds >= 2 and not pd.isna(std_adv):
            se = std_adv / np.sqrt(n_seeds)
            ci_low = mean_adv - 1.96 * se
            ci_high = mean_adv + 1.96 * se
        else:
            ci_low = ci_high = mean_adv
        rows.append({
            "train_rows": int(tr),
            "mean_adv_auc": mean_adv,
            "std_adv_auc": std_adv if not pd.isna(std_adv) else np.nan,
            "adv_auc_ci95_low": ci_low,
            "adv_auc_ci95_high": ci_high,
            "n_seeds": n_seeds,
        })
    out_df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")


def write_report(
    report_path: Path,
    scan_report: Dict[str, Any],
    balance_report: Dict[str, Any],
    summary: pd.DataFrame,
    scaling_balanced: pd.DataFrame,
    rounds_balanced: pd.DataFrame,
    iter_balanced: pd.DataFrame,
) -> None:
    """写入 postprocess_report.txt。"""
    lines = [
        "Adult+CTGAN LOO-MIA Postprocess Report",
        "========================================",
        "",
        "Run completeness: pair_metrics.csv, pair_metrics_control.csv, mmd_component_analysis.json",
        "",
        "Scan & dedup:",
        f"  total_scanned_dirs: {scan_report.get('total_scanned_dirs', 0)}",
        f"  has_metrics_dir (adult+control): {scan_report.get('has_metrics_dir', 0)}",
        f"  incomplete (missing required files): {scan_report.get('incomplete', 0)}",
        f"  complete_before_dedup: {scan_report.get('complete_before_dedup', 0)}",
        f"  duplicate_dropped: {scan_report.get('duplicate_dropped', 0)}",
        f"  final_runs (unique run_key): {scan_report.get('final_runs', 0)}",
        "",
        "Balanced subsets:",
        f"  Group A (scaling) rows: {len(scaling_balanced)}",
        f"  Group B (rounds saturation) rows: {len(rounds_balanced)}",
        f"  Group C (iter strength) rows: {len(iter_balanced)}",
        f"  Group A seeds used: {balance_report.get('group_a_seeds_used', [])}",
        f"  Group B Smin (seeds per rounds): {balance_report.get('group_b_smin')}",
        f"  Group C Imin (seeds per n_iter): {balance_report.get('group_c_imin')}",
        "",
    ]
    if not summary.empty:
        lines.append("Config keys in summary (sample):")
        for _, r in summary.head(5).iterrows():
            lines.append(f"  train_rows={r.get('train_rows')} n_iter={r.get('n_iter')} rounds={r.get('rounds')} candidate={r.get('candidate')} seed={r.get('seed')}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Adult+CTGAN LOO-MIA 端到端后处理：总表、平衡子集、图、Table1、报告")
    parser.add_argument("--outputs", type=Path, default=PROJECT_ROOT / "outputs", help="outputs 根目录")
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "outputs" / "postprocess_adult_ctgan", help="输出目录（CSV/图/报告）")
    parser.add_argument("--with-fig5", action="store_true", help="是否计算并绘制 Fig5 learned attacker")
    parser.add_argument("--n-jobs", type=int, default=1, help="并行进程数（扫描+AUC 与 Fig5 均生效）；1=单进程")
    args = parser.parse_args()

    outputs_dir = args.outputs.resolve()
    if not outputs_dir.is_dir():
        print(f"outputs 目录不存在: {outputs_dir}")
        return 1

    # 1) 扫描 + 去重 → 总表
    summary, scan_report = scan_and_dedup(outputs_dir, n_jobs=args.n_jobs)
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    summary_path = outdir / "adult_ctgan_run_summary.csv"
    if summary.empty:
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print("未找到任何完整 run，已写出空总表。")
        write_report(outdir / "postprocess_report.txt", scan_report, {}, summary, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        return 0

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"总表已写入: {summary_path} (rows={len(summary)})")

    # 2) 平衡子集
    scaling_balanced, rounds_balanced, iter_balanced, balance_report = build_balanced_subsets(summary)
    scaling_balanced.to_csv(outdir / "scaling_balanced.csv", index=False, encoding="utf-8-sig")
    rounds_balanced.to_csv(outdir / "rounds_balanced.csv", index=False, encoding="utf-8-sig")
    iter_balanced.to_csv(outdir / "iter_balanced.csv", index=False, encoding="utf-8-sig")
    print(f"scaling_balanced: {outdir / 'scaling_balanced.csv'} (rows={len(scaling_balanced)})")
    print(f"rounds_balanced: {outdir / 'rounds_balanced.csv'} (rows={len(rounds_balanced)})")
    print(f"iter_balanced: {outdir / 'iter_balanced.csv'} (rows={len(iter_balanced)})")

    # 3) 图
    fig_dir = outdir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if not scaling_balanced.empty:
        fig1_adv_vs_train(scaling_balanced, fig_dir / "fig1_adv_vs_train.png")
        print(f"Fig1: {fig_dir / 'fig1_adv_vs_train.png'}")
        fig4_adv_vs_stability(scaling_balanced, fig_dir / "fig4_adv_vs_stability.png")
        print(f"Fig4: {fig_dir / 'fig4_adv_vs_stability.png'}")
    if not rounds_balanced.empty:
        fig2_adv_vs_rounds(rounds_balanced, fig_dir / "fig2_adv_vs_rounds.png")
        print(f"Fig2: {fig_dir / 'fig2_adv_vs_rounds.png'}")
    if not iter_balanced.empty:
        fig3_adv_vs_iter(iter_balanced, fig_dir / "fig3_adv_vs_iter.png")
        print(f"Fig3: {fig_dir / 'fig3_adv_vs_iter.png'}")

    # 4) Table 1
    if not scaling_balanced.empty:
        write_table1(scaling_balanced, outdir / "table1_scaling.csv")
        print(f"Table1: {outdir / 'table1_scaling.csv'}")

    # 5) 可选 Fig5 learned attacker
    if args.with_fig5:
        n_jobs_fig5 = max(1, args.n_jobs)
        scaling_with_learned = scaling_balanced.copy()
        paths_a = scaling_with_learned["run_dir_path"].tolist()
        if n_jobs_fig5 <= 1:
            learned_vals = [_learned_auc_worker(p) for p in paths_a]
        else:
            with ProcessPoolExecutor(max_workers=n_jobs_fig5) as executor:
                learned_vals = list(executor.map(_learned_auc_worker, paths_a))
        scaling_with_learned["learned_adv"] = [np.nan if v is None else v for v in learned_vals]
        if not scaling_with_learned["learned_adv"].isna().all():
            fig5a_learned_scaling(scaling_with_learned, fig_dir / "fig5a_learned_scaling.png")
            print(f"Fig5A: {fig_dir / 'fig5a_learned_scaling.png'}")
        rounds_with_learned = rounds_balanced.copy()
        paths_b = rounds_with_learned["run_dir_path"].tolist()
        if n_jobs_fig5 <= 1:
            learned_vals_b = [_learned_auc_worker(p) for p in paths_b]
        else:
            with ProcessPoolExecutor(max_workers=n_jobs_fig5) as executor:
                learned_vals_b = list(executor.map(_learned_auc_worker, paths_b))
        rounds_with_learned["learned_adv"] = [np.nan if v is None else v for v in learned_vals_b]
        if not rounds_with_learned["learned_adv"].isna().all():
            fig5b_learned_saturation(rounds_with_learned, fig_dir / "fig5b_learned_saturation.png")
            print(f"Fig5B: {fig_dir / 'fig5b_learned_saturation.png'}")

    # 6) 报告
    write_report(
        outdir / "postprocess_report.txt",
        scan_report,
        balance_report,
        summary,
        scaling_balanced,
        rounds_balanced,
        iter_balanced,
    )
    print(f"报告: {outdir / 'postprocess_report.txt'}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
