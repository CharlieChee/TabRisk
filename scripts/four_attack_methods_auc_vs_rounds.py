#!/usr/bin/env python3
"""
四种 MIA 攻击方法的成功率（AUC）及随 train size (N) 变化曲线。

rounds=20：每个 target 有 20 个影子模型（20 次不同伴随数据集），用于降方差；不是 x 轴。
x 轴：train_size（训练集大小 N），从目录名解析：200, 500, 1000, 1500, 2000 等。

攻击方法：
1. naive_nn：仅用 min_dist_in；2. delta：delta_min_dist；3. density：log(d_in)-log(d_out)；4. learned：LR 5-fold CV。

数据筛选：outputs 下 rounds=20 的 run，且每个 (model, train_size) 至少 5 个 seed。
输出：四种方法成功率表；一张图 4 子图，x=train_size(N)，y=AUC，不同模型同图用 legend 区分，带 error bar。
并行：按 run 并行。
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


# ---------- 1. Naive NN：仅 min_dist_in，全量数据 ----------
def _auc_naive_nn_full(df_m: pd.DataFrame, df_c: pd.DataFrame) -> float:
    """用全部行：score = -min_dist_in。"""
    if "min_dist_in" not in df_m.columns or "min_dist_in" not in df_c.columns:
        return float("nan")
    sm = -df_m["min_dist_in"].astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    sc = -df_c["min_dist_in"].astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
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


def _auc_delta_full(df_m: pd.DataFrame, df_c: pd.DataFrame) -> float:
    col = _choose_delta_col(df_m) or _choose_delta_col(df_c)
    if not col or col not in df_m.columns or col not in df_c.columns:
        return float("nan")
    sm = df_m[col].astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    sc = df_c[col].astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
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


def _auc_density_full(df_m: pd.DataFrame, df_c: pd.DataFrame) -> float:
    dm = _density_score(df_m)
    dc = _density_score(df_c)
    if dm is None or dc is None:
        return float("nan")
    dm = dm[np.isfinite(dm)]
    dc = dc[np.isfinite(dc)]
    if dm.size == 0 or dc.size == 0:
        return float("nan")
    scores = np.concatenate([dm, dc])
    labels = np.concatenate([np.ones(len(dm)), np.zeros(len(dc))])
    return _auc_opt(scores, labels)


# ---------- 4. Learned：全量数据 ----------
def _auc_learned_full(df_m: pd.DataFrame, df_c: pd.DataFrame, cv: int = 5) -> float:
    if len(df_m) + len(df_c) < 10:
        return float("nan")
    numeric_m = set(df_m.select_dtypes(include=[np.number]).columns)
    numeric_c = set(df_c.select_dtypes(include=[np.number]).columns)
    common = numeric_m & numeric_c - {"target_idx", "round"}
    feature_cols = [x for x in FEATURE_CANDIDATES if x in common] or sorted(common)
    if not feature_cols:
        return float("nan")
    Xm = df_m[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()
    Xc = df_c[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()
    X = np.vstack([Xm, Xc])
    y = np.concatenate([np.ones(len(Xm)), np.zeros(len(Xc))])
    if np.unique(y).size < 2 or len(y) < cv:
        return float("nan")
    X = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    res = cross_validate(clf, X, y, cv=min(cv, len(y) // 2), scoring="roc_auc", return_train_score=False)
    auc = float(np.mean(res["test_score"]))
    return max(auc, 1.0 - auc)


def compute_one_run_aucs(run_dir_str: str) -> Optional[Dict[str, Any]]:
    """
    单 run：用该 run 的全部数据（20 个影子模型的所有 target×round 行）算 4 种方法的 AUC 各一个。
    返回 dict: model, train_size, seed, run_dir, auc_naive_nn, auc_delta, auc_density, auc_learned。
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
    parsed = parse_run_dir(run_dir.name)
    if parsed.get("rounds") != 20 or not parsed.get("model"):
        return None
    return {
        "model": parsed["model"],
        "train_size": parsed.get("train_size"),
        "seed": parsed.get("seed"),
        "run_dir": run_dir.name,
        "auc_naive_nn": _auc_naive_nn_full(df_m, df_c),
        "auc_delta": _auc_delta_full(df_m, df_c),
        "auc_density": _auc_density_full(df_m, df_c),
        "auc_learned": _auc_learned_full(df_m, df_c),
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
            r = compute_one_run_aucs(str(d.resolve()))
            if r is not None:
                results.append(r)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = {ex.submit(compute_one_run_aucs, str(d.resolve())): d for d in run_dirs}
            for fut in as_completed(futures):
                r = fut.result()
                if r is not None:
                    results.append(r)
    return results


def results_to_df(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """每条 run 一行：model, train_size, seed, auc_naive_nn, auc_delta, auc_density, auc_learned。"""
    return pd.DataFrame(results)


def aggregate_by_model_train_size(df: pd.DataFrame) -> pd.DataFrame:
    """按 (model, train_size) 聚合成 mean ± std，用于画图。"""
    methods = ["auc_naive_nn", "auc_delta", "auc_density", "auc_learned"]
    means = df.groupby(["model", "train_size"], as_index=False)[methods].mean()
    stds = df.groupby(["model", "train_size"], as_index=False)[methods].std()
    stds = stds.rename(columns={m: f"{m}_std" for m in methods})
    agg = means.merge(stds, on=["model", "train_size"])
    return agg


def print_success_rates(df: pd.DataFrame, agg: pd.DataFrame) -> None:
    """打印四种方法成功率：按 (model, train_size) 及整体。"""
    methods = ["auc_naive_nn", "auc_delta", "auc_density", "auc_learned"]
    method_short = ["naive_nn", "delta", "density", "learned"]
    print("\n========== 四种攻击方法成功率 (AUC, 每 run 用全部 20 影子模型数据) ==========\n")
    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model]
        print(f"  [{model.upper()}] mean AUC over runs:")
        for m, name in zip(methods, method_short):
            mean_auc = sub[m].mean()
            print(f"    {name}: {mean_auc:.4f}")
        print()
    cols_no_std = [c for c in agg.columns if not c.endswith("_std")]
    print("Per (model, train_size) — mean AUC:")
    print(agg[cols_no_std].to_string(index=False))
    print("\nSummary table (mean AUC per model):")
    print("-" * 55)
    header = "Model     " + " ".join(f"{n:>10}" for n in method_short)
    print(header)
    for model in sorted(df["model"].unique()):
        row = f"{model:10}"
        for m in methods:
            row += f" {df[df['model']==model][m].mean():10.4f}"
        print(row)


def plot_auc_vs_train_size(agg: pd.DataFrame, out_path: Path) -> None:
    """4 子图（每方法一个），x=train_size(N)，y=mean AUC，不同模型同图、legend 区分，带 error bar。"""
    methods = ["auc_naive_nn", "auc_delta", "auc_density", "auc_learned"]
    method_labels = {
        "auc_naive_nn": "Naive NN (min_dist only)",
        "auc_delta": "Delta",
        "auc_density": "Density",
        "auc_learned": "Learned attacker",
    }
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    axes = axes.flatten()
    colors = {"ctgan": "C0", "ddpm": "C1"}
    for i, method in enumerate(methods):
        ax = axes[i]
        std_col = f"{method}_std"
        for model in sorted(agg["model"].unique()):
            sub = agg[agg["model"] == model].sort_values("train_size")
            if sub.empty:
                continue
            x = sub["train_size"].to_numpy()
            y = sub[method].to_numpy()
            yerr = sub[std_col].to_numpy() if std_col in sub.columns else None
            if yerr is not None:
                yerr = np.where(np.isfinite(yerr), yerr, 0)
            ax.errorbar(
                x, y, yerr=yerr, marker="o", label=model.upper(),
                color=colors.get(model, "gray"), capsize=4, capthick=1,
            )
        ax.set_xlabel("Train size (N)")
        ax.set_ylabel("AUC")
        ax.set_title(method_labels.get(method, method))
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.45, 1.0)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    fig.suptitle("AUC vs Train size (N); rounds=20 shadow models per target, ≥5 seeds per setting", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_path}")


# 四种方法 × 两种模型：区分度高的配色（方法用颜色，模型用线型+标记）
METHOD_COLORS = {
    "auc_naive_nn": "#2E86AB",    # 钢蓝
    "auc_delta": "#E94F37",       # 朱红
    "auc_density": "#44AF69",    # 青绿
    "auc_learned": "#7B2D8E",    # 紫
}
METHOD_LABELS_SHORT = {
    "auc_naive_nn": "Naive NN",
    "auc_delta": "Delta",
    "auc_density": "Density",
    "auc_learned": "Learned",
}


def plot_auc_vs_train_size_combined(agg: pd.DataFrame, out_path: Path) -> None:
    """所有 8 条线（4 方法 × 2 模型）画在同一张图里，配色清晰、有区分度。"""
    methods = ["auc_naive_nn", "auc_delta", "auc_density", "auc_learned"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # 模型区分：CTGAN 实线+圆点，DDPM 虚线+方点
    model_style = {"ctgan": ("-", "o", 6), "ddpm": ("--", "s", 5)}
    for method in methods:
        color = METHOD_COLORS.get(method, "#333333")
        mlabel = METHOD_LABELS_SHORT.get(method, method)
        for model in sorted(agg["model"].unique()):
            sub = agg[agg["model"] == model].sort_values("train_size")
            if sub.empty:
                continue
            ls, marker, ms = model_style.get(model, ("-", "o", 5))
            std_col = f"{method}_std"
            x = sub["train_size"].to_numpy()
            y = sub[method].to_numpy()
            yerr = sub[std_col].to_numpy() if std_col in sub.columns else None
            if yerr is not None:
                yerr = np.where(np.isfinite(yerr), yerr, 0)
            label = f"{mlabel} ({model.upper()})"
            ax.errorbar(
                x, y, yerr=yerr, linestyle=ls, marker=marker, markersize=ms,
                color=color, capsize=2.5, capthick=1, label=label,
            )
    ax.set_xlabel("Train size (N)", fontsize=11)
    ax.set_ylabel("AUC", fontsize=11)
    ax.set_ylim(0.48, 1.0)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.grid(True, alpha=0.35, linestyle="-")
    ax.legend(loc="upper right", fontsize=9, ncol=2, framealpha=0.95)
    ax.set_title("AUC vs Train size (N)\n(rounds=20 shadow models per target, ≥5 seeds)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="四种 MIA 方法 (naive_nn, delta, density, learned) 成功率及 AUC 随 train size (N) 变化，rounds=20 且 ≥5 seeds"
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
    print(f"Valid runs: {len(results)}")
    if not results:
        print("No valid results. Exit.")
        return 1

    df = results_to_df(results)
    agg = aggregate_by_model_train_size(df)
    args.outdir.mkdir(parents=True, exist_ok=True)

    print_success_rates(df, agg)
    summary_path = args.outdir / "four_methods_auc_by_train_size.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("model\ttrain_size\tnaive_nn\tdelta\tdensity\tlearned\n")
        for _, r in agg.iterrows():
            ts = r["train_size"]
            ts_str = str(int(ts)) if pd.notna(ts) and np.isfinite(ts) else ""
            f.write(f"{r['model']}\t{ts_str}\t{r['auc_naive_nn']:.4f}\t{r['auc_delta']:.4f}\t{r['auc_density']:.4f}\t{r['auc_learned']:.4f}\n")
    print(f"Summary written: {summary_path}")

    plot_auc_vs_train_size(agg, args.outdir / "auc_vs_train_size_four_methods.png")
    plot_auc_vs_train_size_combined(agg, args.outdir / "auc_vs_train_size_combined.png")

    df.to_csv(args.outdir / "per_run_aucs.csv", index=False)
    agg.to_csv(args.outdir / "auc_vs_train_size_aggregated.csv", index=False)
    print(f"CSV: {args.outdir / 'per_run_aucs.csv'}, {args.outdir / 'auc_vs_train_size_aggregated.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
