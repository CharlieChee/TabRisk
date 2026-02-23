#!/usr/bin/env python3
"""
四种 MIA 攻击方法的成功率（AUC）及随轮次变化曲线。

攻击方法：
1. naive_nn：仅用最近邻距离 min_dist_in 作为信号（无 control，无 delta）；score = -min_dist_in（越小越像 member）
2. delta：用 delta_min_dist（in/out 对比）作为信号
3. density：用 density 分数 log(d_in)-log(d_out) 作为信号
4. learned：Learned attacker，LR 5-fold CV

数据筛选：outputs 下 rounds=20 的 run，且每个 (model, train_size) setting 至少 5 个 seed。
输出：四种方法的成功率表（AUC@round=20）；一张图 4 子图（每方法一子图），x=轮次(1..20)，y=AUC，不同模型用不同 legend 画在同一图中。
并行：按 run 并行计算。
"""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DENSITY_EPS = 1e-10
FEATURE_CANDIDATES = [
    "mmd_numeric", "mmd_mixed", "min_dist_in", "min_dist_out", "delta_min_dist",
    "num_mean_diff_mean", "num_mean_diff_max", "num_std_diff_mean", "num_std_diff_max",
    "cat_tv_mean", "cat_tv_max",
]


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
    if "ddpm" in dirname.lower() or "tabddpm" in dirname.lower():
        out["model"] = "ddpm"
    elif "ctgan" in dirname.lower():
        out["model"] = "ctgan"
    return out


def _auc_opt(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC 或 1-AUC 取大（攻击成功率 >= 0.5）。"""
    if scores.size == 0 or labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    auc = roc_auc_score(labels, scores)
    return max(float(auc), 1.0 - float(auc))


# ---------- 1. Naive NN：仅 min_dist_in，无 delta ----------
def _auc_naive_nn_at_k(df_m: pd.DataFrame, df_c: pd.DataFrame, k: int) -> float:
    """用前 k 轮：score = -min_dist_in（越小越像 member）。"""
    if "min_dist_in" not in df_m.columns or "min_dist_in" not in df_c.columns:
        return float("nan")
    if "round" not in df_m.columns or "round" not in df_c.columns:
        return float("nan")
    m = df_m[df_m["round"] < k]
    c = df_c[df_c["round"] < k]
    if m.empty or c.empty:
        return float("nan")
    sm = -m["min_dist_in"].astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    sc = -c["min_dist_in"].astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if sm.size == 0 or sc.size == 0:
        return float("nan")
    scores = np.concatenate([sm, sc])
    labels = np.concatenate([np.ones(len(sm)), np.zeros(len(sc))])
    return _auc_opt(scores, labels)


# ---------- 2. Delta ----------
def _choose_delta_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("delta_min_dist_mean", "delta_min_dist", "delta_min_dist_median"):
        if c in df.columns:
            return c
    for c in df.columns:
        if "delta" in c.lower():
            return c
    return None


def _auc_delta_at_k(df_m: pd.DataFrame, df_c: pd.DataFrame, k: int) -> float:
    col = _choose_delta_col(df_m) or _choose_delta_col(df_c)
    if not col or col not in df_m.columns or col not in df_c.columns:
        return float("nan")
    m = df_m[df_m["round"] < k]
    c = df_c[df_c["round"] < k]
    if m.empty or c.empty:
        return float("nan")
    sm = m[col].astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    sc = c[col].astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if sm.size == 0 or sc.size == 0:
        return float("nan")
    scores = np.concatenate([sm, sc])
    labels = np.concatenate([np.ones(len(sm)), np.zeros(len(sc))])
    return _auc_opt(scores, labels)


# ---------- 3. Density ----------
def _density_score(df: pd.DataFrame) -> Optional[np.ndarray]:
    if "min_dist_in" not in df.columns or "min_dist_out" not in df.columns:
        return None
    din = df["min_dist_in"].astype(float).replace([np.inf, -np.inf], np.nan)
    dout = df["min_dist_out"].astype(float).replace([np.inf, -np.inf], np.nan)
    din = np.maximum(din.to_numpy(), DENSITY_EPS)
    dout = np.maximum(dout.to_numpy(), DENSITY_EPS)
    return np.log(din) - np.log(dout)


def _auc_density_at_k(df_m: pd.DataFrame, df_c: pd.DataFrame, k: int) -> float:
    m = df_m[df_m["round"] < k]
    c = df_c[df_c["round"] < k]
    if m.empty or c.empty:
        return float("nan")
    dm = _density_score(m)
    dc = _density_score(c)
    if dm is None or dc is None:
        return float("nan")
    dm = dm[np.isfinite(dm)]
    dc = dc[np.isfinite(dc)]
    if dm.size == 0 or dc.size == 0:
        return float("nan")
    scores = np.concatenate([dm, dc])
    labels = np.concatenate([np.ones(len(dm)), np.zeros(len(dc))])
    return _auc_opt(scores, labels)


# ---------- 4. Learned ----------
def _auc_learned_at_k(df_m: pd.DataFrame, df_c: pd.DataFrame, k: int, cv: int = 5) -> float:
    m = df_m[df_m["round"] < k]
    c = df_c[df_c["round"] < k]
    if m.empty or c.empty or len(m) + len(c) < 10:
        return float("nan")
    numeric_m = set(m.select_dtypes(include=[np.number]).columns)
    numeric_c = set(c.select_dtypes(include=[np.number]).columns)
    common = numeric_m & numeric_c - {"target_idx", "round"}
    feature_cols = [x for x in FEATURE_CANDIDATES if x in common] or sorted(common)
    if not feature_cols:
        return float("nan")
    Xm = m[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()
    Xc = c[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()
    X = np.vstack([Xm, Xc])
    y = np.concatenate([np.ones(len(Xm)), np.zeros(len(Xc))])
    if np.unique(y).size < 2 or len(y) < cv:
        return float("nan")
    X = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    res = cross_validate(clf, X, y, cv=min(cv, len(y) // 2), scoring="roc_auc", return_train_score=False)
    auc = float(np.mean(res["test_score"]))
    return max(auc, 1.0 - auc)


def compute_one_run_curves(run_dir_str: str) -> Optional[Dict[str, Any]]:
    """
    单 run：读 pair_metrics / pair_metrics_control，对 k=1..20 和 4 种方法算 AUC。
    返回 dict: model, train_size, seed, run_dir, curves { method: [auc_k1, ..., auc_k20] }。
    """
    run_dir = Path(run_dir_str)
    metrics_dir = run_dir / "shadow_pair_metrics"
    pair_path = metrics_dir / "pair_metrics.csv"
    pair_control_path = metrics_dir / "pair_metrics_control.csv"
    if not pair_path.exists() or not pair_control_path.exists():
        return None
    df_m = pd.read_csv(pair_path)
    df_c = pd.read_csv(pair_control_path)
    if df_m.empty or df_c.empty:
        return None
    if "round" not in df_m.columns or "round" not in df_c.columns:
        return None
    parsed = parse_run_dir(run_dir.name)
    if parsed.get("rounds") != 20:
        return None
    model = parsed.get("model")
    if not model:
        return None
    max_round = max(
        int(df_m["round"].max()) if "round" in df_m.columns else 0,
        int(df_c["round"].max()) if "round" in df_c.columns else 0,
    )
    if max_round < 19:
        return None
    curves = {"naive_nn": [], "delta": [], "density": [], "learned": []}
    for k in range(1, 21):
        curves["naive_nn"].append(_auc_naive_nn_at_k(df_m, df_c, k))
        curves["delta"].append(_auc_delta_at_k(df_m, df_c, k))
        curves["density"].append(_auc_density_at_k(df_m, df_c, k))
        curves["learned"].append(_auc_learned_at_k(df_m, df_c, k))
    return {
        "model": model,
        "train_size": parsed.get("train_size"),
        "seed": parsed.get("seed"),
        "run_dir": run_dir.name,
        "curves": curves,
    }


def scan_runs(outputs_dir: Path) -> List[Path]:
    """扫描 outputs 下含 control、shadow_pair_metrics 且 pair_metrics.csv / pair_metrics_control.csv 存在的 run。"""
    run_dirs = []
    outputs_dir = outputs_dir.resolve()
    if not outputs_dir.is_dir():
        return []
    for d in outputs_dir.iterdir():
        if not d.is_dir() or "control" not in d.name:
            continue
        m = d / "shadow_pair_metrics"
        if m.is_dir() and (m / "pair_metrics.csv").exists() and (m / "pair_metrics_control.csv").exists():
            run_dirs.append(d)
    for sub in outputs_dir.iterdir():
        if not sub.is_dir() or sub.name.startswith("."):
            continue
        for d in sub.iterdir():
            if not d.is_dir() or "control" not in d.name:
                continue
            m = d / "shadow_pair_metrics"
            if m.is_dir() and (m / "pair_metrics.csv").exists() and (m / "pair_metrics_control.csv").exists():
                run_dirs.append(d)
    return sorted(set(run_dirs), key=lambda p: (p.name, str(p)))


def filter_rounds20_min5seeds(run_dirs: List[Path]) -> List[Path]:
    """只保留 rounds=20，且每个 (model, train_size) 至少 5 个 seed 的 run。"""
    rows = []
    for d in run_dirs:
        p = parse_run_dir(d.name)
        if p.get("rounds") != 20 or not p.get("model"):
            continue
        rows.append({
            "run_dir": str(d.resolve()),
            "model": p["model"],
            "train_size": p.get("train_size"),
            "seed": p.get("seed"),
            "timestamp": p.get("timestamp") or "",
        })
    if not rows:
        return []
    df = pd.DataFrame(rows)
    # 每个 (model, train_size) 至少 5 seeds：先算 seed 数，再筛出合格 setting
    seed_counts = df.groupby(["model", "train_size"])["seed"].nunique().reset_index()
    seed_counts.columns = ["model", "train_size", "seed_count"]
    valid_settings = seed_counts[seed_counts["seed_count"] >= 5][["model", "train_size"]]
    if valid_settings.empty:
        return []
    df_valid = df.merge(valid_settings, on=["model", "train_size"], how="inner")
    # 每个 (model, train_size, seed) 保留一条（timestamp 最新）
    df_valid = df_valid.sort_values("timestamp", ascending=True, na_position="last")
    df_valid = df_valid.drop_duplicates(subset=["model", "train_size", "seed"], keep="last")
    return [Path(p) for p in df_valid["run_dir"].tolist()]


def run_parallel(run_dirs: List[Path], n_jobs: int) -> List[Dict[str, Any]]:
    results = []
    if n_jobs <= 1:
        for d in run_dirs:
            r = compute_one_run_curves(str(d.resolve()))
            if r is not None:
                results.append(r)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = {ex.submit(compute_one_run_curves, str(d.resolve())): d for d in run_dirs}
            for fut in as_completed(futures):
                r = fut.result()
                if r is not None:
                    results.append(r)
    return results


def aggregate_curves_by_model(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[float]]]:
    """按 (model, method) 对 AUC(k) 求平均。"""
    # results: list of { model, curves: { method: [auc_1..auc_20] } }
    by_model_method: Dict[Tuple[str, str], List[List[float]]] = {}
    for r in results:
        model = r["model"]
        for method, curve in r["curves"].items():
            key = (model, method)
            if key not in by_model_method:
                by_model_method[key] = []
            by_model_method[key].append(curve)
    out = {}
    for (model, method), curves in by_model_method.items():
        if model not in out:
            out[model] = {}
        arr = np.array(curves)
        out[model][method] = np.nanmean(arr, axis=0).tolist()
    return out


def print_success_rates(by_model: Dict[str, Dict[str, List[float]]]) -> None:
    """打印四种方法在 round=20 时的成功率（AUC）。"""
    rounds_idx = 19  # k=20 -> index 19
    print("\n========== 四种攻击方法成功率 (AUC @ 20 rounds) ==========\n")
    methods = ["naive_nn", "delta", "density", "learned"]
    for model in sorted(by_model.keys()):
        print(f"  [{model}]")
        for method in methods:
            curve = by_model[model].get(method, [])
            auc20 = curve[rounds_idx] if len(curve) > rounds_idx else float("nan")
            print(f"    {method}: {auc20:.4f}")
        print()
    # 表格式
    print("Summary table (AUC @ round=20):")
    print("-" * 50)
    header = "Model     " + " ".join(f"{m:>10}" for m in methods)
    print(header)
    for model in sorted(by_model.keys()):
        row = f"{model:10}"
        for method in methods:
            curve = by_model[model].get(method, [])
            auc20 = curve[rounds_idx] if len(curve) > rounds_idx else float("nan")
            row += f" {auc20:10.4f}"
        print(row)


def plot_auc_vs_rounds(by_model: Dict[str, Dict[str, List[float]]], out_path: Path) -> None:
    """4 子图（每方法一个），每子图内不同模型同图，legend 区分。"""
    methods = ["naive_nn", "delta", "density", "learned"]
    method_labels = {"naive_nn": "Naive NN (min_dist only)", "delta": "Delta", "density": "Density", "learned": "Learned attacker"}
    rounds = list(range(1, 21))
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    axes = axes.flatten()
    colors = {"ctgan": "C0", "ddpm": "C1"}
    for i, method in enumerate(methods):
        ax = axes[i]
        for model in sorted(by_model.keys()):
            curve = by_model[model].get(method, [])
            if len(curve) < 20:
                continue
            ax.plot(rounds, curve[:20], "-o", label=model.upper(), color=colors.get(model, "gray"), markersize=3)
        ax.set_xlabel("Number of rounds (k)")
        ax.set_ylabel("AUC")
        ax.set_title(method_labels.get(method, method))
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.45, 1.0)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    fig.suptitle("AUC vs rounds (rounds=20, ≥5 seeds per setting)", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="四种 MIA 方法 (naive_nn, delta, density, learned) 成功率及 AUC 随轮次变化，rounds=20 且 ≥5 seeds"
    )
    parser.add_argument("--outputs", type=Path, default=PROJECT_ROOT / "outputs", help="outputs 目录")
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "outputs" / "four_attack_methods", help="输出目录（表+图）")
    parser.add_argument("--n-jobs", type=int, default=4, help="并行进程数")
    args = parser.parse_args()

    run_dirs = scan_runs(args.outputs)
    run_dirs = filter_rounds20_min5seeds(run_dirs)
    print(f"Runs (rounds=20, ≥5 seeds per setting): {len(run_dirs)}")
    if not run_dirs:
        print("No runs found. Exit.")
        return 1

    results = run_parallel(run_dirs, args.n_jobs)
    print(f"Valid runs with curves: {len(results)}")
    if not results:
        print("No valid results. Exit.")
        return 1

    by_model = aggregate_curves_by_model(results)
    args.outdir.mkdir(parents=True, exist_ok=True)

    print_success_rates(by_model)
    summary_path = args.outdir / "four_methods_auc_at_20rounds.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        # 简单把表格再写一份
        methods = ["naive_nn", "delta", "density", "learned"]
        f.write("AUC @ round=20\n")
        for model in sorted(by_model.keys()):
            for method in methods:
                curve = by_model[model].get(method, [])
                auc20 = curve[19] if len(curve) > 19 else float("nan")
                f.write(f"{model}\t{method}\t{auc20:.4f}\n")
    print(f"Summary written: {summary_path}")

    plot_auc_vs_rounds(by_model, args.outdir / "auc_vs_rounds_four_methods.png")

    # 写入完整曲线 CSV：model, method, round_k, auc_mean
    rows = []
    for model in sorted(by_model.keys()):
        for method in ["naive_nn", "delta", "density", "learned"]:
            curve = by_model[model].get(method, [])
            for k, auc in enumerate(curve, start=1):
                rows.append({"model": model, "method": method, "round_k": k, "auc": auc})
    pd.DataFrame(rows).to_csv(args.outdir / "auc_vs_rounds_curves.csv", index=False)
    print(f"Curves CSV: {args.outdir / 'auc_vs_rounds_curves.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
