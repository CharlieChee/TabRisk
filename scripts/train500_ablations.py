#!/usr/bin/env python3
"""
Train=500 消融与饱和曲线：总汇总表 + 训练强度消融(Adv vs iter) + Shadow 饱和(Adv vs rounds)。

Step 1: 扫描 outputs 下 train500_synth500_ctgan + control 的 run，去重后生成 adult_train500_full_summary.csv
Step 2: 画 Adv(AUC) vs n_iter（固定 rounds=20）
Step 3: 画 Adv(AUC) vs rounds（固定 iter=1000）

用法:
  python scripts/train500_ablations.py build-summary --outputs outputs --out outputs/adult_train500_full_summary.csv
  python scripts/train500_ablations.py plot-iter --summary outputs/adult_train500_full_summary.csv --out outputs/figures/adv_auc_vs_iter_train500_rounds20.png
  python scripts/train500_ablations.py plot-rounds --summary outputs/adult_train500_full_summary.csv --out outputs/figures/adv_auc_vs_rounds_train500_iter1000.png
  python scripts/train500_ablations.py all --outputs outputs --outdir outputs
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
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 目录名匹配：只处理 train500_synth500_ctgan 且含 control
NAME_MUST_CONTAIN = ["train500_synth500_ctgan", "control"]


def _parse_dirname(dirname: str) -> Dict[str, Any]:
    """从 run 目录名解析 train_rows, n_iter, batch_size, rounds_per_target, seed。"""
    out = {
        "train_rows": None,
        "n_iter": None,
        "batch_size": None,
        "rounds_per_target": None,
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
    m = re.search(r"_seed(\d+)(?:_|$)", dirname)
    if m:
        out["seed"] = int(m.group(1))
    # 时间戳：末尾 _YYYYMMDD_HHMMSS，用于“最新”排序
    m = re.search(r"_(\d{8}_\d{6})$", dirname)
    if m:
        out["timestamp"] = m.group(1)
    return out


def _get_rounds_from_config(run_dir: Path) -> Optional[int]:
    """从 run_dir/train_config.yaml 读取 shadow.num_shadow_rounds。"""
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


def _get_seed_from_config(run_dir: Path) -> Optional[int]:
    """从 run_dir/train_config.yaml 读取 shadow.random_seed。"""
    cfg_path = run_dir / "train_config.yaml"
    if not cfg_path.is_file():
        return None
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(cfg_path)
        v = OmegaConf.select(cfg, "shadow.random_seed", default=None)
        return int(v) if v is not None else None
    except Exception:
        return None


def _choose_delta_column(df: pd.DataFrame) -> Optional[str]:
    delta_cols = [c for c in df.columns if "delta" in c.lower()]
    if not delta_cols:
        return None
    for cand in ("delta_min_dist_mean", "delta_min_dist", "delta_min_dist_median", "delta_mean"):
        if cand in df.columns:
            return cand
    return delta_cols[0]


def _auc_adv_from_pair_files(pair_path: Path, pair_control_path: Path) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """计算 AUC 和 adv_auc。返回 (auc, adv_auc, error_msg)。"""
    if not pair_path.exists():
        return None, None, "missing pair_metrics.csv"
    if not pair_control_path.exists():
        return None, None, "missing pair_metrics_control.csv"
    df_m = pd.read_csv(pair_path)
    df_c = pd.read_csv(pair_control_path)
    if df_m.empty or df_c.empty:
        return None, None, "empty CSV"
    delta_col = _choose_delta_column(df_m) or _choose_delta_column(df_c)
    if not delta_col or delta_col not in df_m.columns or delta_col not in df_c.columns:
        return None, None, "missing delta column"
    scores_m = df_m[delta_col].astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    scores_c = df_c[delta_col].astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if scores_m.size == 0 or scores_c.size == 0:
        return None, None, "no valid delta scores"
    scores = np.concatenate([scores_m, scores_c])
    labels = np.concatenate([np.ones(len(scores_m)), np.zeros(len(scores_c))])
    if np.unique(labels).size < 2:
        return None, None, "single class"
    try:
        auc = float(roc_auc_score(labels, scores))
    except Exception:
        return None, None, "AUC failed"
    return auc, auc - 0.5, None


def _load_json_safe(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        import json
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_stability_from_mmd(path: Path) -> Optional[float]:
    """从 mmd_component_analysis.json 提取一个标量。"""
    data = _load_json_safe(path)
    if not data:
        return None
    scalars: List[float] = []

    def collect(d: Any) -> None:
        if isinstance(d, dict):
            for v in d.values():
                collect(v)
        elif isinstance(d, (int, float)) and not isinstance(d, bool):
            scalars.append(float(d))
        elif isinstance(d, list) and d and not isinstance(d[0], (dict, list)):
            for x in d:
                if isinstance(x, (int, float)) and not isinstance(x, bool):
                    scalars.append(float(x))
    collect(data)
    return float(np.mean(scalars)) if scalars else None


def _run_complete(run_dir: Path) -> bool:
    """run 是否文件完整：pair_metrics, pair_metrics_control, mmd_component_analysis。"""
    m = run_dir / "shadow_pair_metrics"
    if not m.is_dir():
        return False
    return (
        (m / "pair_metrics.csv").exists()
        and (m / "pair_metrics_control.csv").exists()
        and (m / "mmd_component_analysis.json").exists()
    )


def build_full_summary(outputs_dir: Path, out_path: Path) -> pd.DataFrame:
    """
    Step 1: 扫描 outputs 下 train500_synth500_ctgan + control 的 run，
    解析参数、计算 AUC/adv_auc、提取 stability；
    同一 (train_rows, n_iter, rounds_per_target, batch_size, seed) 只保留文件完整且时间戳最新的。
    输出表列: train_rows, n_iter, rounds, batch_size, seed, adv_auc, auc, stability_metric, run_dir_name, run_dir_path
    """
    outputs_dir = outputs_dir.resolve()
    if not outputs_dir.is_dir():
        raise FileNotFoundError(f"outputs 目录不存在: {outputs_dir}")

    candidates: List[Path] = []
    for d in outputs_dir.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if not all(x in name for x in NAME_MUST_CONTAIN):
            continue
        if not _run_complete(d):
            continue
        candidates.append(d)

    rows: List[Dict[str, Any]] = []
    for run_dir in candidates:
        name = run_dir.name
        parsed = _parse_dirname(name)
        rounds = parsed.get("rounds_per_target")
        if rounds is None:
            rounds = _get_rounds_from_config(run_dir)
        seed = parsed.get("seed")
        if seed is None:
            seed = _get_seed_from_config(run_dir)
        # 用时间戳字符串排序，无则用 mtime
        ts = parsed.get("timestamp") or ""
        mtime = (run_dir / "shadow_pair_metrics" / "pair_metrics.csv").stat().st_mtime if (run_dir / "shadow_pair_metrics" / "pair_metrics.csv").exists() else 0

        metrics_dir = run_dir / "shadow_pair_metrics"
        pair_path = metrics_dir / "pair_metrics.csv"
        pair_control_path = metrics_dir / "pair_metrics_control.csv"
        mmd_path = metrics_dir / "mmd_component_analysis.json"

        auc_val, adv_val, err = _auc_adv_from_pair_files(pair_path, pair_control_path)
        if err:
            continue
        stability = _extract_stability_from_mmd(mmd_path)

        rows.append({
            "train_rows": parsed.get("train_rows"),
            "n_iter": parsed.get("n_iter"),
            "rounds": rounds,
            "batch_size": parsed.get("batch_size"),
            "seed": seed if seed is not None else -1,
            "adv_auc": adv_val,
            "auc": auc_val,
            "stability_metric": stability if stability is not None else np.nan,
            "_timestamp": ts,
            "_mtime": mtime,
            "run_dir_name": name,
            "run_dir_path": str(run_dir.resolve()),
        })

    if not rows:
        df = pd.DataFrame(columns=["train_rows", "n_iter", "rounds", "batch_size", "seed", "adv_auc", "auc", "stability_metric", "run_dir_name", "run_dir_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print("未找到任何符合条件的 run，已写出空表。")
        return df

    df = pd.DataFrame(rows)

    # 去重：同一 (train_rows, n_iter, rounds, batch_size, seed) 只保留时间戳最新（或 mtime 最大）的一条
    key_cols = ["train_rows", "n_iter", "rounds", "batch_size", "seed"]
    for c in key_cols:
        if c in df.columns:
            df[c] = df[c].fillna(-1)
    df["_sort"] = df["_timestamp"].fillna("") + "_" + df["_mtime"].astype(str)
    dedup = (
        df.sort_values("_sort", ascending=False)
        .groupby(key_cols, dropna=False)
        .first()
        .reset_index()
    )
    for c in key_cols:
        dedup[c] = dedup[c].replace(-1, np.nan)
    out_cols = ["train_rows", "n_iter", "rounds", "batch_size", "seed", "adv_auc", "auc", "stability_metric", "run_dir_name", "run_dir_path"]
    dedup = dedup[[c for c in out_cols if c in dedup.columns]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dedup.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"已写入 {len(dedup)} 行到 {out_path}")
    return dedup


def plot_adv_auc_vs_iter(summary_path: Path, out_path: Path) -> None:
    """
    Step 2: 从汇总表筛选 train_rows=500, rounds=20，以 n_iter 为 x，adv_auc 为 y，
    每个 iter 画 seed 散点 + mean ± std error bar。
    """
    df = pd.read_csv(summary_path)
    df = df[(df["train_rows"] == 500) & (df["rounds"] == 20)].dropna(subset=["n_iter", "adv_auc"])
    if df.empty:
        print("无 train_rows=500 且 rounds=20 的数据，跳过画图")
        return
    df = df.astype({"n_iter": int, "adv_auc": float})
    iters = sorted(df["n_iter"].unique())
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, n_iter in enumerate(iters):
        sub = df[df["n_iter"] == n_iter]
        x = np.full(len(sub), n_iter) + np.random.uniform(-0.15, 0.15, len(sub))
        ax.scatter(x, sub["adv_auc"], alpha=0.6, s=40, color="C0", edgecolors="k", linewidths=0.5)
        mean_y = sub["adv_auc"].mean()
        std_y = sub["adv_auc"].std()
        std_y = std_y if pd.notna(std_y) and std_y > 0 else 0
        ax.errorbar(n_iter, mean_y, yerr=std_y, fmt="o", color="C1", markersize=10, capsize=5, capthick=2, label=None if i else "mean ± std")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("n_iter")
    ax.set_ylabel("Adv(AUC)")
    ax.set_title("Train=500, rounds=20: Adv(AUC) vs n_iter (training intensity ablation)")
    ax.set_xticks(iters)
    if iters:
        ax.legend(loc="best")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {out_path}")


def plot_adv_auc_vs_rounds(summary_path: Path, out_path: Path) -> None:
    """
    Step 3: 从汇总表筛选 train_rows=500, n_iter=1000，以 rounds 为 x，adv_auc 为 y，
    每个 rounds 画 seed 散点 + mean ± std。
    """
    df = pd.read_csv(summary_path)
    df = df[(df["train_rows"] == 500) & (df["n_iter"] == 1000)].dropna(subset=["rounds", "adv_auc"])
    if df.empty:
        print("无 train_rows=500 且 n_iter=1000 的数据，跳过画图")
        return
    df = df.astype({"rounds": int, "adv_auc": float})
    rounds_vals = sorted(df["rounds"].unique())
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, r in enumerate(rounds_vals):
        sub = df[df["rounds"] == r]
        x = np.full(len(sub), r) + np.random.uniform(-0.15, 0.15, len(sub))
        ax.scatter(x, sub["adv_auc"], alpha=0.6, s=40, color="C0", edgecolors="k", linewidths=0.5)
        mean_y = sub["adv_auc"].mean()
        std_y = sub["adv_auc"].std()
        std_y = std_y if pd.notna(std_y) and std_y > 0 else 0
        ax.errorbar(r, mean_y, yerr=std_y, fmt="o", color="C1", markersize=10, capsize=5, capthick=2, label=None if i else "mean ± std")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("rounds_per_target")
    ax.set_ylabel("Adv(AUC)")
    ax.set_title("Train=500, iter=1000: Adv(AUC) vs rounds (shadow saturation)")
    ax.set_xticks(rounds_vals)
    if rounds_vals:
        ax.legend(loc="best")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train=500 消融与饱和：汇总表 + Adv vs iter + Adv vs rounds")
    sub = parser.add_subparsers(dest="command", required=True)

    # build-summary
    p1 = sub.add_parser("build-summary", help="Step 1: 生成 adult_train500_full_summary.csv")
    p1.add_argument("--outputs", type=Path, default=PROJECT_ROOT / "outputs", help="outputs 根目录")
    p1.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "adult_train500_full_summary.csv", help="输出 CSV 路径")

    # plot-iter
    p2 = sub.add_parser("plot-iter", help="Step 2: Adv(AUC) vs n_iter (train500, rounds=20)")
    p2.add_argument("--summary", type=Path, default=PROJECT_ROOT / "outputs" / "adult_train500_full_summary.csv")
    p2.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "figures" / "adv_auc_vs_iter_train500_rounds20.png")

    # plot-rounds
    p3 = sub.add_parser("plot-rounds", help="Step 3: Adv(AUC) vs rounds (train500, iter=1000)")
    p3.add_argument("--summary", type=Path, default=PROJECT_ROOT / "outputs" / "adult_train500_full_summary.csv")
    p3.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "figures" / "adv_auc_vs_rounds_train500_iter1000.png")

    # all
    p4 = sub.add_parser("all", help="依次执行 build-summary, plot-iter, plot-rounds")
    p4.add_argument("--outputs", type=Path, default=PROJECT_ROOT / "outputs")
    p4.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "outputs", help="汇总表与图所在根目录")

    args = parser.parse_args()
    cmd = args.command

    if cmd == "build-summary":
        build_full_summary(args.outputs, args.out)
    elif cmd == "plot-iter":
        plot_adv_auc_vs_iter(args.summary, args.out)
    elif cmd == "plot-rounds":
        plot_adv_auc_vs_rounds(args.summary, args.out)
    elif cmd == "all":
        summary_path = args.outdir / "adult_train500_full_summary.csv"
        fig_dir = args.outdir / "figures"
        build_full_summary(args.outputs, summary_path)
        plot_adv_auc_vs_iter(summary_path, fig_dir / "adv_auc_vs_iter_train500_rounds20.png")
        plot_adv_auc_vs_rounds(summary_path, fig_dir / "adv_auc_vs_rounds_train500_iter1000.png")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
