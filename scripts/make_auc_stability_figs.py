#!/usr/bin/env python3
"""
批量扫描 outputs 下带 control 的 run，要求 shadow_pair_metrics 含 pair_metrics.csv、
control_comparison.json、mmd_component_analysis.json。计算 per-run AUC/adv_auc（基于 pair+control 行级）、
从 mmd_component_analysis.json 提取 stability metric，输出汇总表与两张图。

用法:
  python scripts/make_auc_stability_figs.py --outputs outputs --outdir outputs/figures
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
from scipy import stats
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 与 make_shadow_figures 对齐的 JSON 展平字段（可选，用于合并到汇总表）
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


def parse_params_from_dirname(dirname: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """从 run 目录名解析 train_size, iter, batchsize。正则: _train(\\d+)_, _(\\d+)iter_, _bs(\\d+)_。"""
    train_size = None
    iters = None
    batchsize = None
    m = re.search(r"_train(\d+)_", dirname)
    if m:
        train_size = int(m.group(1))
    m = re.search(r"_(\d+)iter", dirname)
    if m:
        iters = int(m.group(1))
    m = re.search(r"_bs(\d+)(?:_|$)", dirname)
    if m:
        batchsize = int(m.group(1))
    return (train_size, iters, batchsize)


def flatten_json(obj: Any, prefix: str = "") -> Dict[str, Any]:
    """嵌套 dict 展平为一层。"""
    out: Dict[str, Any] = {}
    if not isinstance(obj, dict):
        out[prefix.rstrip("_")] = obj
        return out
    for k, v in obj.items():
        new_key = f"{prefix}{k}"
        if isinstance(v, dict) and v and not any(
            isinstance(x, (dict, list)) for x in (v.values() if isinstance(v, dict) else [])
        ):
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


def _choose_delta_column(df: pd.DataFrame) -> Optional[str]:
    """从 pair_metrics 中选一个 delta 分数列：优先 delta_min_dist_mean，否则自动选择唯一/最合理的 delta 列。"""
    delta_cols = [c for c in df.columns if "delta" in c.lower()]
    if not delta_cols:
        return None
    for cand in ("delta_min_dist_mean", "delta_min_dist", "delta_min_dist_median", "delta_mean"):
        if cand in df.columns:
            return cand
    return delta_cols[0]


def _auc_from_scores_labels(scores: np.ndarray, labels: np.ndarray) -> float:
    """计算 AUC（1 = member, 0 = non-member；score 越大越像 member）。"""
    if scores.size == 0 or labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return float("nan")


def compute_auc_and_ci(
    pair_path: Path,
    pair_control_path: Path,
    delta_col: Optional[str],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[str]]:
    """
    从 pair_metrics.csv（member, label=1）与 pair_metrics_control.csv（non-member, label=0）计算 AUC 与 bootstrap CI。
    返回 (auc_in_out, adv_auc, adv_auc_ci95_low, adv_auc_ci95_high, error_msg)。
    error_msg 非空表示缺失 label/文件等，应跳过 AUC。
    """
    if not pair_path.exists():
        return None, None, None, None, "missing pair_metrics.csv"
    if not pair_control_path.exists():
        return None, None, None, None, "missing pair_metrics_control.csv (cannot define non-member label)"

    df_m = pd.read_csv(pair_path)
    df_c = pd.read_csv(pair_control_path)
    if df_m.empty or df_c.empty:
        return None, None, None, None, "empty pair or control CSV"

    if delta_col is None:
        delta_col = _choose_delta_column(df_m)
    if delta_col is None:
        delta_col = _choose_delta_column(df_c)
    if delta_col not in df_m.columns or delta_col not in df_c.columns:
        return None, None, None, None, f"missing delta column (e.g. delta_min_dist)"

    scores_m = df_m[delta_col].astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    scores_c = df_c[delta_col].astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if scores_m.size == 0 or scores_c.size == 0:
        return None, None, None, None, "no valid delta scores after dropna"

    # 每个 pair 一行：member=1, non-member=0；score 用 delta（越大越可能 member）
    scores = np.concatenate([scores_m, scores_c])
    labels = np.concatenate([np.ones(len(scores_m)), np.zeros(len(scores_c))])
    n = len(labels)
    auc = _auc_from_scores_labels(scores, labels)
    if np.isnan(auc):
        return None, None, None, None, "AUC computation failed (e.g. single class)"

    adv = auc - 0.5
    rng = np.random.default_rng(seed)
    boot_adv: List[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        s, l = scores[idx], labels[idx]
        a = _auc_from_scores_labels(s, l)
        if not np.isnan(a):
            boot_adv.append(a - 0.5)
    boot_adv_arr = np.array(boot_adv)
    if boot_adv_arr.size == 0:
        ci_low, ci_high = adv, adv
    else:
        ci_low = float(np.percentile(boot_adv_arr, 2.5))
        ci_high = float(np.percentile(boot_adv_arr, 97.5))
    return auc, adv, ci_low, ci_high, None


def extract_stability_from_mmd_json(path: Path) -> Tuple[Optional[float], Optional[str]]:
    """
    从 mmd_component_analysis.json 提取一个稳定性标量。优先 total 类键，否则对全部数值取 mean。
    返回 (stability_metric, stability_metric_name)。higher value = more shift / less stable。
    """
    if not path.exists():
        return None, None
    data = load_json_safe(path)
    if not data:
        return None, None

    # 递归收集所有 (key_path, numeric_value)
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

    # 优先：最能代表整体 MMD / total shift 的键
    priority_keys = (
        "mmd_total", "total_mmd", "mmd_overall", "overall", "total",
        "mean_mixed_minus_numeric", "var_mixed_minus_numeric",
    )
    for key in priority_keys:
        for name, val in scalars:
            if key in name.lower() or name.lower() == key:
                return val, name

    if not scalars:
        return None, None
    # 否则用所有数值的 mean 作为整体标量，并记录为 "mean_of_components"
    vals = [v for _, v in scalars]
    return float(np.mean(vals)), "mean_of_components"


def scan_and_build_summary_with_auc_stability(
    outputs_dir: Path,
) -> Tuple[pd.DataFrame, int, List[Path], List[Tuple[Path, str]]]:
    """
    只处理名称包含 control 的 run；完整 run 需 shadow_pair_metrics 且含
    pair_metrics.csv, control_comparison.json, mmd_component_analysis.json。
    返回 (summary_df, n_auc_ok, list_incomplete, list_missing_label) 。
    """
    outputs_dir = outputs_dir.resolve()
    control_dirs: List[Path] = []
    for d in outputs_dir.iterdir():
        if not d.is_dir() or "control" not in d.name:
            continue
        metrics_dir = d / "shadow_pair_metrics"
        if not metrics_dir.is_dir():
            continue
        if not (metrics_dir / "pair_metrics.csv").exists():
            continue
        if not (metrics_dir / "control_comparison.json").exists():
            continue
        if not (metrics_dir / "mmd_component_analysis.json").exists():
            continue
        control_dirs.append(d)

    key_counts: Dict[Tuple[int, int, int], int] = {}
    rows: List[Dict[str, Any]] = []
    incomplete: List[Path] = []
    missing_label: List[Tuple[Path, str]] = []
    n_auc_ok = 0

    for run_dir in sorted(control_dirs):
        metrics_dir = run_dir / "shadow_pair_metrics"
        dirname = run_dir.name
        train_size, iters, batchsize = parse_params_from_dirname(dirname)
        key = (
            train_size if train_size is not None else -1,
            iters if iters is not None else -1,
            batchsize if batchsize is not None else -1,
        )
        key_counts[key] = key_counts.get(key, 0) + 1
        run_id = key_counts[key]

        row: Dict[str, Any] = {
            "train_size": train_size,
            "iter": iters,
            "batchsize": batchsize,
            "run_id": run_id,
            "run_dir_name": dirname,
            "run_dir_path": str(run_dir.resolve()),
        }

        # 可选：gds / cc 列（与 summary_auto 对齐）
        gds_path = metrics_dir / "global_delta_stats.json"
        cc_path = metrics_dir / "control_comparison.json"
        if gds_path.exists():
            gds_data = load_json_safe(gds_path)
            if gds_data:
                flat = flatten_json(gds_data, prefix="gds_")
                for k in GDS_KEYS:
                    row[f"gds_{k}"] = flat.get(f"gds_{k}", np.nan)
        for k in GDS_KEYS:
            if f"gds_{k}" not in row:
                row[f"gds_{k}"] = np.nan
        if cc_path.exists():
            cc_data = load_json_safe(cc_path)
            if cc_data:
                flat = flatten_json(cc_data, prefix="cc_")
                for k in CC_KEYS:
                    row[f"cc_{k}"] = flat.get(f"cc_{k}", np.nan)
        for k in CC_KEYS:
            if f"cc_{k}" not in row:
                row[f"cc_{k}"] = np.nan

        # AUC（per-pair 行级：member=pair_metrics, non-member=pair_metrics_control）
        pair_path = metrics_dir / "pair_metrics.csv"
        pair_control_path = metrics_dir / "pair_metrics_control.csv"
        auc_val, adv_val, ci_low, ci_high, err = compute_auc_and_ci(
            pair_path, pair_control_path, None
        )
        if err:
            row["auc_in_out"] = np.nan
            row["adv_auc"] = np.nan
            row["adv_auc_ci95_low"] = np.nan
            row["adv_auc_ci95_high"] = np.nan
            missing_label.append((run_dir, err))
        else:
            row["auc_in_out"] = auc_val
            row["adv_auc"] = adv_val
            row["adv_auc_ci95_low"] = ci_low
            row["adv_auc_ci95_high"] = ci_high
            n_auc_ok += 1

        # stability 来自 mmd_component_analysis.json
        mmd_path = metrics_dir / "mmd_component_analysis.json"
        stab_val, stab_name = extract_stability_from_mmd_json(mmd_path)
        row["stability_metric"] = stab_val if stab_val is not None else np.nan
        row["stability_metric_name"] = stab_name if stab_name else ""

        rows.append(row)

    df = pd.DataFrame(rows)
    return df, n_auc_ok, incomplete, missing_label


def fig_adv_auc_vs_train_size(df: pd.DataFrame, out_path: Path, jitter_seed: int = 42) -> None:
    """图 A: x=train_size, y=adv_auc；按 iter 区分颜色；同配置 jitter + 均值点；y=0；error bar 用 bootstrap CI。"""
    df = df.dropna(subset=["train_size", "adv_auc"]).copy()
    if df.empty:
        return
    df["iter"] = df["iter"].fillna(-1).astype(int)
    iters = sorted(df["iter"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(iters), 1)))
    fig, ax = plt.subplots(figsize=(8, 5))
    rng = np.random.default_rng(jitter_seed)
    for i, it in enumerate(iters):
        sub = df[df["iter"] == it]
        x = sub["train_size"].astype(float)
        y = sub["adv_auc"].astype(float)
        x_range = (x.max() - x.min()) or 1.0
        jitter = (rng.random(len(x)) - 0.5) * x_range * 0.03
        ax.scatter(
            x + jitter, y, alpha=0.7, s=50, c=[colors[i % len(colors)]],
            label=f"iter={int(it) if it >= 0 else '?'}", edgecolors="k", linewidths=0.5
        )
        # 同一 (train_size, iter) 均值点 + error bar（多 run 用 replicate std，否则用 bootstrap CI）
        for ts in sub["train_size"].unique():
            block = sub[sub["train_size"] == ts]
            mean_y = block["adv_auc"].mean()
            if len(block) > 1:
                err = block["adv_auc"].std()
                if np.isnan(err) or err <= 0:
                    err = 0
                ax.errorbar([ts], [mean_y], yerr=err, fmt="none", color=colors[i % len(colors)], capsize=3)
            elif "adv_auc_ci95_low" in block.columns and block["adv_auc_ci95_low"].notna().any():
                ci_lo = block["adv_auc_ci95_low"].iloc[0]
                ci_hi = block["adv_auc_ci95_high"].iloc[0]
                ax.errorbar([ts], [mean_y], yerr=[[mean_y - ci_lo], [ci_hi - mean_y]], fmt="none",
                            color=colors[i % len(colors)], capsize=3)
            ax.scatter([ts], [mean_y], s=140, c=[colors[i % len(colors)]], marker="s",
                       edgecolors="k", linewidths=1.5, zorder=5)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Train size")
    ax.set_ylabel("Adv(AUC) = AUC − 0.5")
    ax.set_title("Adv(AUC) vs Train Size (by iter)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_adv_auc_vs_stability(df: pd.DataFrame, out_path: Path) -> None:
    """图 B: x=stability_metric, y=adv_auc；颜色 train_size，marker iter；线性回归，Pearson r/p；y=0。"""
    df = df.dropna(subset=["stability_metric", "adv_auc"]).copy()
    if df.empty:
        return
    df["train_size"] = df["train_size"].fillna(-1).astype(int)
    df["iter"] = df["iter"].fillna(-1).astype(int)
    train_sizes = sorted(df["train_size"].unique())
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, max(len(train_sizes), 1)))
    markers = ["o", "s", "^", "D", "v"][:5]
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, ts in enumerate(train_sizes):
        sub = df[df["train_size"] == ts]
        for j, it in enumerate(sub["iter"].unique()):
            block = sub[sub["iter"] == it]
            m = markers[j % len(markers)]
            ax.scatter(
                block["stability_metric"], block["adv_auc"],
                c=[colors[i % len(colors)]], marker=m, s=60, alpha=0.8,
                label=f"train={int(ts) if ts >= 0 else '?'}, iter={int(it) if it >= 0 else '?'}"
            )
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    x = df["stability_metric"].to_numpy(dtype=float)
    y = df["adv_auc"].to_numpy(dtype=float)
    if x.size > 2 and np.isfinite(x).all() and np.isfinite(y).all():
        slope, intercept, r, p, _ = stats.linregress(x, y)
        yy_pred = slope * x + intercept
        r2 = 1 - np.sum((y - yy_pred) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12)
        xx = np.linspace(x.min(), x.max(), 100)
        ax.plot(xx, slope * xx + intercept, "k-", linewidth=1.5, label="linear fit")
        ax.set_title(
            f"Adv(AUC) vs Stability (higher stability_metric = more shift / less stable)\n"
            f"Pearson r = {r:.3f}, p = {p:.4f}, R² = {r2:.4f}"
        )
    else:
        ax.set_title("Adv(AUC) vs Stability (higher stability_metric = more shift / less stable)")
    ax.set_xlabel("Stability metric (from mmd_component_analysis.json)")
    ax.set_ylabel("Adv(AUC)")
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 control run 的 pair_metrics + control + mmd_component_analysis 计算 AUC/adv_auc 与 stability，输出汇总表与两张图",
    )
    parser.add_argument("--outputs", type=Path, default=PROJECT_ROOT / "outputs", help="outputs 根目录")
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "outputs" / "figures", help="图片输出目录")
    parser.add_argument("--summary-name", type=str, default="control_shadow_metrics_summary_with_auc_stability.csv",
                        help="汇总表文件名（放在 outputs 下）")
    args = parser.parse_args()

    outputs_dir = args.outputs.resolve()
    if not outputs_dir.is_dir():
        print(f"outputs 目录不存在: {outputs_dir}")
        return

    df, n_auc_ok, incomplete, missing_label = scan_and_build_summary_with_auc_stability(outputs_dir)
    if df.empty:
        print("没有满足条件的完整 run（需 pair_metrics.csv + control_comparison.json + mmd_component_analysis.json），退出")
        return

    summary_path = outputs_dir / args.summary_name
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"汇总表已写入: {summary_path}")

    print(f"成功计算 AUC 的 run 数: {n_auc_ok} / {len(df)}")
    if missing_label:
        print("缺失 label / 缺失文件的 run 列表:")
        for run_dir, err in missing_label:
            print(f"  {run_dir.name}: {err}")
    if incomplete:
        print(f"因缺少必要文件未纳入的 control 目录数: {len(incomplete)}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    fig_a = args.outdir / "fig_adv_auc_vs_train_size.png"
    fig_b = args.outdir / "fig_adv_auc_vs_stability.png"
    fig_adv_auc_vs_train_size(df, fig_a)
    fig_adv_auc_vs_stability(df, fig_b)
    print(f"图 A 已保存: {fig_a}")
    print(f"图 B 已保存: {fig_b}")

    # 统计摘要：每个 train_size 的 adv_auc 平均值
    adv = df["adv_auc"].dropna()
    if not adv.empty:
        print("\n按 train_size 的 adv_auc 平均值:")
        for ts in sorted(df["train_size"].dropna().unique()):
            sub = df[df["train_size"] == ts]["adv_auc"].dropna()
            if not sub.empty:
                print(f"  train_size={int(ts)}: mean(adv_auc) = {sub.mean():.4f} (n={len(sub)})")


if __name__ == "__main__":
    main()
