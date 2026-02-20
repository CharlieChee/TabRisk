#!/usr/bin/env python3
"""
批量扫描 outputs 下带 control 的 run，汇总 shadow_pair_metrics 的 JSON 与可选 CSV，
生成汇总表、四张核心图（论文级）及聚合表。

用法:
  python scripts/make_shadow_figures.py --outputs outputs --outdir outputs/figures
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 从 JSON 需要读的字段（展平后 key）
GDS_KEYS = [
    "n", "mean_delta", "median_delta", "std_delta", "pos_rate",
    "ci95_low", "ci95_high", "cohens_d",
    "t_test_t_stat", "t_test_p_value_two_sided",
    "wilcoxon_statistic", "wilcoxon_p_value_two_sided",
]
CC_KEYS = [
    "n", "mean_delta_target", "mean_delta_control", "mean_diff", "var_diff",
    "t_test_paired_t_stat", "t_test_paired_p_value_two_sided",
    "t_test_paired_p_value_one_sided_target_gt_control", "cohens_d_paired",
]


def parse_params_from_dirname(dirname: str) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """从 run 目录名解析 train_size, synth_size, iter, batchsize。"""
    train_size = None
    synth_size = None
    iters = None
    batchsize = None
    m = re.search(r"_train(\d+)_", dirname)
    if m:
        train_size = int(m.group(1))
    m = re.search(r"_synth(\d+)(?:_|$)", dirname)
    if m:
        synth_size = int(m.group(1))
    m = re.search(r"_(\d+)iter", dirname)
    if m:
        iters = int(m.group(1))
    m = re.search(r"_bs(\d+)(?:_|$)", dirname)
    if m:
        batchsize = int(m.group(1))
    return (train_size, synth_size, iters, batchsize)


def flatten_json(obj: Any, prefix: str = "") -> Dict[str, Any]:
    """嵌套 dict 展平为一层，key 用 prefix 与下划线连接。"""
    out: Dict[str, Any] = {}
    if not isinstance(obj, dict):
        out[prefix.rstrip("_")] = obj
        return out
    for k, v in obj.items():
        new_key = f"{prefix}{k}"
        if isinstance(v, dict) and v and not any(isinstance(x, (dict, list)) for x in (v.values() if isinstance(v, dict) else [])):
            for k2, v2 in v.items():
                out[f"{new_key}_{k2}"] = v2
        else:
            out[new_key] = v if not isinstance(v, (dict, list)) else json.dumps(v, ensure_ascii=False)
    return out


def load_json_safe(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def scan_and_build_summary(outputs_dir: Path) -> Tuple[pd.DataFrame, int, int, List[Path]]:
    """
    扫描 outputs 下名称包含 control 的 run，判定完整 run，构建汇总表。
    返回 (summary_df, n_control_dirs, n_complete, list_of_incomplete_paths)。
    """
    outputs_dir = outputs_dir.resolve()
    control_dirs: List[Path] = []
    for d in outputs_dir.iterdir():
        if not d.is_dir() or "control" not in d.name:
            continue
        metrics_dir = d / "shadow_pair_metrics"
        if not metrics_dir.is_dir():
            continue
        control_dirs.append(d)

    n_control = len(control_dirs)
    key_counts: Dict[Tuple[int, int, int], int] = {}
    rows: List[Dict[str, Any]] = []
    incomplete: List[Path] = []

    for run_dir in sorted(control_dirs):
        metrics_dir = run_dir / "shadow_pair_metrics"
        gds_path = metrics_dir / "global_delta_stats.json"
        cc_path = metrics_dir / "control_comparison.json"
        if not gds_path.exists() or not cc_path.exists():
            incomplete.append(run_dir)
            continue

        dirname = run_dir.name
        train_size, synth_size, iters, batchsize = parse_params_from_dirname(dirname)
        key = (
            train_size if train_size is not None else -1,
            iters if iters is not None else -1,
            batchsize if batchsize is not None else -1,
        )
        key_counts[key] = key_counts.get(key, 0) + 1
        run_id = key_counts[key]

        gds_data = load_json_safe(gds_path)
        cc_data = load_json_safe(cc_path)
        if not gds_data or not cc_data:
            incomplete.append(run_dir)
            continue

        flat_gds = flatten_json(gds_data, prefix="gds_")
        flat_cc = flatten_json(cc_data, prefix="cc_")

        row = {
            "train_size": train_size,
            "synth_size": synth_size,
            "iter": iters,
            "batchsize": batchsize,
            "run_id": run_id,
            "run_dir_name": dirname,
            "run_dir_path": str(run_dir.resolve()),
            "shadow_pair_metrics_path": str(metrics_dir.resolve()),
        }
        for k in GDS_KEYS:
            row[f"gds_{k}"] = flat_gds.get(f"gds_{k}", np.nan)
        for k in CC_KEYS:
            flat_k = "cc_" + k
            row[flat_k] = flat_cc.get(flat_k, np.nan)
        rows.append(row)

    df = pd.DataFrame(rows)
    n_complete = len(df)
    return df, n_control, n_complete, incomplete


def save_summary_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")


def fig1_train_size_vs_paired_d(df: pd.DataFrame, outdir: Path) -> None:
    """图1: x=train_size, y=cc_cohens_d_paired, 按 iter 分组，同配置 jitter + 均值点，y=0 参考线。"""
    col = "cc_cohens_d_paired"
    if col not in df.columns:
        return
    df = df.dropna(subset=["train_size", "iter", col])
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    iters = sorted(df["iter"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(iters), 1)))
    for i, it in enumerate(iters):
        sub = df[df["iter"] == it]
        x = sub["train_size"].astype(float)
        y = sub[col].astype(float)
        jitter = (np.random.RandomState(42).rand(len(x)) - 0.5) * 40
        ax.scatter(x + jitter, y, alpha=0.7, s=40, c=[colors[i]], label=f"iter={int(it)}", edgecolors="k", linewidths=0.5)
        # 同一 (train_size, iter) 的均值
        for ts in sub["train_size"].unique():
            mean_y = sub[sub["train_size"] == ts][col].mean()
            ax.scatter([ts], [mean_y], s=120, c=[colors[i]], marker="s", edgecolors="k", linewidths=1.5, zorder=5)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Train size")
    ax.set_ylabel("Paired Cohen's d (target − control)")
    ax.set_title("Train Size vs Membership Strength (paired Cohen's d)")
    ax.legend()
    ax.set_xticks(sorted(df["train_size"].unique()))
    plt.tight_layout()
    fig.savefig(outdir / "fig1_train_size_vs_paired_d.png", dpi=200)
    plt.close(fig)


def fig2_iter_vs_paired_d(df: pd.DataFrame, outdir: Path) -> None:
    """图2: x=iter, y=mean(cc_cohens_d_paired), 三条线对应 train_size, 误差条为同配置 std。"""
    col = "cc_cohens_d_paired"
    if col not in df.columns:
        return
    df = df.dropna(subset=["train_size", "iter", col])
    if df.empty:
        return
    agg = df.groupby(["train_size", "iter"], as_index=False).agg(
        mean_d=(col, "mean"),
        std_d=(col, "std"),
        count=(col, "count"),
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    train_sizes = sorted(agg["train_size"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(train_sizes), 1)))
    for i, ts in enumerate(train_sizes):
        sub = agg[agg["train_size"] == ts].sort_values("iter")
        x = sub["iter"].values
        y = sub["mean_d"].values
        err = sub["std_d"].values
        has_err = sub["count"].values > 1
        err_plot = np.where(has_err, err, 0)
        ax.errorbar(x, y, yerr=err_plot, marker="o", capsize=4, label=f"train_size={int(ts)}", color=colors[i])
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Paired Cohen's d (target − control)")
    ax.set_title("Iteration vs Membership (mean ± std over replicates)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(outdir / "fig2_iter_vs_paired_d.png", dpi=200)
    plt.close(fig)


def _get_delta_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if "delta" in c.lower() and "mean" in c.lower():
            return c
    for c in df.columns:
        if "delta" in c.lower():
            return c
    return None


def fig3_delta_distribution(summary_df: pd.DataFrame, outputs_dir: Path, outdir: Path) -> None:
    """
    图3: Delta 分布对比。优先切片 iter=1000, batchsize=256；缺则每 train_size 选 iter 最大且最常见 batch。
    数据来源：shadow_pair_metrics/summary_by_target.csv 的 delta 列，否则 pair_metrics 聚合。
    """
    # 选代表 run：优先 iter=1000, bs=256；否则每 train_size 选 iter 最大的一组
    candidates = summary_df.dropna(subset=["train_size", "iter", "batchsize"])
    if candidates.empty:
        return
    slice_1000_256 = candidates[(candidates["iter"] == 1000) & (candidates["batchsize"] == 256)]
    if slice_1000_256.empty:
        # 每 train_size 取 iter 最大的一组（第一个）
        reps = []
        for ts in candidates["train_size"].unique():
            sub = candidates[candidates["train_size"] == ts].sort_values("iter", ascending=False)
            if not sub.empty:
                reps.append(sub.iloc[0])
        if not reps:
            return
        slice_df = pd.DataFrame(reps)
    else:
        slice_df = slice_1000_256.drop_duplicates("train_size", keep="first")

    all_deltas: Dict[int, np.ndarray] = {}
    outputs_dir = Path(outputs_dir)
    for _, r in slice_df.iterrows():
        run_path = Path(r["run_dir_path"])
        metrics_dir = run_path / "shadow_pair_metrics"
        ts = int(r["train_size"]) if r["train_size"] is not None else 0
        summary_path = metrics_dir / "summary_by_target.csv"
        if summary_path.exists():
            try:
                sdf = pd.read_csv(summary_path)
                delta_col = _get_delta_column(sdf)
                if delta_col:
                    vals = sdf[delta_col].dropna().astype(float).values
                    if vals.size > 0:
                        all_deltas[ts] = vals
                        continue
            except Exception:
                pass
        pair_path = metrics_dir / "pair_metrics.csv"
        if pair_path.exists():
            try:
                pdf = pd.read_csv(pair_path)
                dc = _get_delta_column(pdf)
                if dc:
                    per_target = pdf.groupby("target_idx")[dc].mean()
                    vals = per_target.dropna().astype(float).values
                    if vals.size > 0:
                        all_deltas[ts] = vals
            except Exception:
                pass

    if not all_deltas:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    train_sizes = sorted(all_deltas.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(train_sizes), 1)))
    for i, ts in enumerate(train_sizes):
        vals = all_deltas[ts]
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(vals)
            xmin, xmax = float(vals.min()), float(vals.max())
            pad = (xmax - xmin) * 0.1 or 0.5
            xx = np.linspace(xmin - pad, xmax + pad, 200)
            ax.plot(xx, kde(xx), color=colors[i], label=f"train_size={ts}", linewidth=2)
        except Exception:
            ax.hist(vals, bins=25, alpha=0.5, density=True, color=colors[i], label=f"train_size={ts}")
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Delta (min dist)")
    ax.set_ylabel("Density")
    ax.set_title("Delta distribution by train size (slice: iter=1000, bs=256 or fallback)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(outdir / "fig3_delta_distribution.png", dpi=200)
    plt.close(fig)


def fig4_replicate_stability(df: pd.DataFrame, outdir: Path) -> None:
    """图4: 仅同配置多 run_id 的配置，x=配置名，y=cc_cohens_d_paired，每 run 一点+均值。"""
    col = "cc_cohens_d_paired"
    if col not in df.columns:
        return
    multi = df.groupby(["train_size", "iter", "batchsize"]).agg(n_run=("run_id", "count")).reset_index()
    multi = multi[multi["n_run"] > 1]
    if multi.empty:
        return
    merged = df.merge(multi[["train_size", "iter", "batchsize"]], on=["train_size", "iter", "batchsize"], how="inner")
    merged = merged.dropna(subset=[col])
    if merged.empty:
        return
    merged["config_name"] = (
        "train" + merged["train_size"].astype(int).astype(str)
        + "-iter" + merged["iter"].astype(int).astype(str)
        + "-bs" + merged["batchsize"].astype(int).astype(str)
    )
    config_order = merged.groupby("config_name")[col].mean().sort_values().index.tolist()
    merged["config_name"] = pd.Categorical(merged["config_name"], categories=config_order, ordered=True)
    merged = merged.sort_values("config_name")

    fig, ax = plt.subplots(figsize=(max(8, len(config_order) * 0.8), 5))
    for i, cfg in enumerate(config_order):
        sub = merged[merged["config_name"] == cfg]
        x_jitter = np.random.RandomState(42).rand(len(sub)) * 0.4 - 0.2
        ax.scatter(i + x_jitter, sub[col].values, alpha=0.7, s=50, c="steelblue", edgecolors="k", linewidths=0.5)
        ax.scatter([i], [sub[col].mean()], s=150, marker="*", c="orange", edgecolors="k", linewidths=1, zorder=5)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(config_order)))
    ax.set_xticklabels(config_order, rotation=45, ha="right")
    ax.set_ylabel("Paired Cohen's d (target − control)")
    ax.set_title("Replicate variability (multiple runs per config)")
    plt.tight_layout()
    fig.savefig(outdir / "fig4_replicate_stability.png", dpi=200)
    plt.close(fig)


def save_agg_table(df: pd.DataFrame, out_path: Path) -> None:
    """按 (train_size, iter, batchsize) 聚合：mean(std) cc_cohens_d_paired, mean p_value, mean gds_mean_delta。"""
    group = ["train_size", "iter", "batchsize"]
    if df.empty or not all(c in df.columns for c in group):
        return
    agg_cols = []
    if "cc_cohens_d_paired" in df.columns:
        agg_cols.append(("cc_cohens_d_paired_mean", "cc_cohens_d_paired", "mean"))
        agg_cols.append(("cc_cohens_d_paired_std", "cc_cohens_d_paired", "std"))
    if "cc_t_test_paired_p_value_two_sided" in df.columns:
        agg_cols.append(("cc_t_test_paired_p_value_two_sided_mean", "cc_t_test_paired_p_value_two_sided", "mean"))
    if "gds_mean_delta" in df.columns:
        agg_cols.append(("gds_mean_delta_mean", "gds_mean_delta", "mean"))
    if not agg_cols:
        return
    agg_spec = {new_name: (col, func) for new_name, col, func in agg_cols}
    out = df.groupby(group, as_index=False).agg(**agg_spec)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="扫描 control run、汇总表、四张核心图与聚合表")
    parser.add_argument("--outputs", type=Path, default=PROJECT_ROOT / "outputs", help="outputs 根目录")
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "outputs" / "figures", help="图片输出目录")
    args = parser.parse_args()

    outputs_dir = args.outputs.resolve()
    if not outputs_dir.is_dir():
        print(f"目录不存在: {outputs_dir}")
        return

    df, n_control, n_complete, incomplete = scan_and_build_summary(outputs_dir)
    print(f"总共扫描到的 control run 数量: {n_control}")
    print(f"完整 run 数量: {n_complete}")
    print(f"缺失 json 的 run 数量: {len(incomplete)}")
    if incomplete:
        for p in incomplete:
            print(f"  缺失: {p}")

    if df.empty:
        print("无完整 run，退出")
        return

    summary_path = outputs_dir / "control_shadow_metrics_summary_auto.csv"
    save_summary_csv(df, summary_path)
    print(f"汇总表已保存: {summary_path}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    fig1_train_size_vs_paired_d(df, args.outdir)
    fig2_iter_vs_paired_d(df, args.outdir)
    fig3_delta_distribution(df, outputs_dir, args.outdir)
    fig4_replicate_stability(df, args.outdir)
    print(f"四张图已保存到: {args.outdir}")

    agg_path = outputs_dir / "control_shadow_metrics_summary_agg.csv"
    save_agg_table(df, agg_path)
    print(f"聚合表已保存: {agg_path}")


if __name__ == "__main__":
    main()
