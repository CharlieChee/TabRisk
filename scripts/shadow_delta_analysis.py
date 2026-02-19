#!/usr/bin/env python3
"""
对 shadow_pair_metrics 的结果做全局与对照分析，包括：

任务 1：delta_min_dist_mean 的总体统计检验（global_delta_stats.json）
任务 2：delta 分布可视化（delta_distribution.png）
任务 3：Control 对照（target vs control 的 delta，对应 control_comparison.json）
任务 4：numeric-only vs mixed MMD 对照（mmd_component_analysis.json）
任务 5：异常 target 分析，输出 round 级 delta 列表（outlier_targets.json）

默认从：
  run_dir/shadow_pair_metrics/summary_by_target.csv
  run_dir/shadow_pair_metrics/pair_metrics.csv
读取数据，可通过参数覆盖。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _float(v, ndigits: int = 4) -> float:
    """辅助：统一四位小数输出。"""
    try:
        return round(float(v), ndigits)
    except Exception:
        return float("nan")


def task1_global_delta_stats(
    summary_df: pd.DataFrame,
    out_path: Path,
    delta_col: str = "delta_min_dist_mean",
) -> None:
    """任务 1：对 summary_by_target.csv 中的 delta_min_dist_mean 做统计检验。"""
    if delta_col not in summary_df.columns:
        raise KeyError(f"summary_by_target.csv 不含列 {delta_col}")

    delta = summary_df[delta_col].dropna().to_numpy(dtype=float)
    n = delta.size
    if n == 0:
        raise ValueError("delta 序列为空，无法做统计检验")

    mean = float(delta.mean())
    median = float(np.median(delta))
    std = float(delta.std(ddof=1)) if n > 1 else float("nan")
    pos_rate = float((delta > 0).mean())

    # 95% CI for mean (t 分布)
    if n > 1 and std > 0:
        se = std / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n - 1)
        ci_low = mean - t_crit * se
        ci_high = mean + t_crit * se
    else:
        ci_low = float("nan")
        ci_high = float("nan")

    # one-sample t-test, H0: mean = 0
    if n > 1:
        t_res = stats.ttest_1samp(delta, 0.0, alternative="two-sided")
        t_stat = float(t_res.statistic)
        p_value = float(t_res.pvalue)
    else:
        t_stat = float("nan")
        p_value = float("nan")

    # Wilcoxon signed-rank test
    try:
        if n > 0:
            w_res = stats.wilcoxon(delta, alternative="two-sided", zero_method="pratt")
            w_stat = float(w_res.statistic)
            w_p = float(w_res.pvalue)
        else:
            w_stat = float("nan")
            w_p = float("nan")
    except Exception:
        w_stat = float("nan")
        w_p = float("nan")

    # Cohen's d
    if n > 1 and std > 0:
        cohend = mean / std
    else:
        cohend = float("nan")

    out = {
        "n": int(n),
        "mean_delta": _float(mean),
        "median_delta": _float(median),
        "std_delta": _float(std),
        "pos_rate": _float(pos_rate),
        "ci95_low": _float(ci_low),
        "ci95_high": _float(ci_high),
        "t_test": {
            "t_stat": _float(t_stat),
            "p_value_two_sided": _float(p_value),
        },
        "wilcoxon": {
            "statistic": _float(w_stat),
            "p_value_two_sided": _float(w_p),
        },
        "cohens_d": _float(cohend),
    }

    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


def task2_delta_visualization(
    summary_df: pd.DataFrame,
    out_path: Path,
    delta_col: str = "delta_min_dist_mean",
) -> None:
    """任务 2：绘制 delta 的直方图 / 箱线图 / 小提琴图。"""
    if delta_col not in summary_df.columns:
        raise KeyError(f"summary_by_target.csv 不含列 {delta_col}")

    delta = summary_df[delta_col].dropna().to_numpy(dtype=float)
    if delta.size == 0:
        raise ValueError("delta 序列为空，无法绘图")

    mean = float(delta.mean())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 直方图
    ax = axes[0]
    ax.hist(delta, bins=30, color="skyblue", edgecolor="black", alpha=0.7)
    ax.axvline(0.0, color="red", linestyle="--", label="0")
    ax.axvline(mean, color="green", linestyle="-", label=f"mean={mean:.3f}")
    ax.set_title("Histogram of delta_min_dist_mean")
    ax.set_xlabel("delta")
    ax.set_ylabel("count")
    ax.legend()

    # 箱线图
    ax = axes[1]
    ax.boxplot(delta, vert=True, showmeans=True)
    ax.axhline(0.0, color="red", linestyle="--")
    ax.set_title("Boxplot of delta_min_dist_mean")
    ax.set_ylabel("delta")

    # 小提琴图
    ax = axes[2]
    ax.violinplot(delta, showmeans=True, showmedians=True)
    ax.axhline(0.0, color="red", linestyle="--")
    ax.axhline(mean, color="green", linestyle="-", label="mean")
    ax.set_title("Violin plot of delta_min_dist_mean")
    ax.set_ylabel("delta")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def task3_control_comparison(
    summary_real: pd.DataFrame,
    summary_ctrl: pd.DataFrame,
    out_path: Path,
    id_col: str = "target_idx_",
    delta_col: str = "delta_min_dist_mean",
) -> None:
    """
    任务 3：Control 对照实验统计（假定 control 的 delta 已经算好，位于另一份 summary_by_target_control.csv）。

    比较：
      delta_target vs delta_control（成对样本，按 target_idx 对齐）。
    """
    if id_col not in summary_real.columns or id_col not in summary_ctrl.columns:
        raise KeyError(f"summary_real/ctrl 必须均包含 id 列 {id_col}")
    if delta_col not in summary_real.columns or delta_col not in summary_ctrl.columns:
        raise KeyError(f"summary_real/ctrl 必须均包含列 {delta_col}")

    merged = pd.merge(
        summary_real[[id_col, delta_col]].rename(columns={delta_col: "delta_target"}),
        summary_ctrl[[id_col, delta_col]].rename(columns={delta_col: "delta_control"}),
        on=id_col,
        how="inner",
    )
    if merged.empty:
        raise ValueError("real 与 control 没有共同的 target_idx，无法对照")

    delta_t = merged["delta_target"].to_numpy(dtype=float)
    delta_c = merged["delta_control"].to_numpy(dtype=float)
    diff = delta_t - delta_c
    n = diff.size

    mean_t = float(delta_t.mean())
    mean_c = float(delta_c.mean())
    mean_diff = float(diff.mean())
    var_diff = float(diff.var(ddof=1)) if n > 1 else float("nan")

    # t-test: H0: mean_diff = 0, Ha: mean_diff > 0 (target > control?)
    if n > 1:
        t_res = stats.ttest_rel(delta_t, delta_c, alternative="two-sided")
        t_stat = float(t_res.statistic)
        p_two = float(t_res.pvalue)
        # 单侧 p（target > control）
        if t_stat > 0:
            p_one = p_two / 2.0
        else:
            p_one = 1.0 - p_two / 2.0
    else:
        t_stat = float("nan")
        p_two = float("nan")
        p_one = float("nan")

    # Cohen's d（配对）：mean_diff / std(diff)
    if n > 1 and var_diff > 0:
        cohend = mean_diff / np.sqrt(var_diff)
    else:
        cohend = float("nan")

    out = {
        "n": int(n),
        "mean_delta_target": _float(mean_t),
        "mean_delta_control": _float(mean_c),
        "mean_diff": _float(mean_diff),
        "var_diff": _float(var_diff),
        "t_test_paired": {
            "t_stat": _float(t_stat),
            "p_value_two_sided": _float(p_two),
            "p_value_one_sided_target_gt_control": _float(p_one),
        },
        "cohens_d_paired": _float(cohend),
    }
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


def task4_mmd_component_analysis(
    summary_df: pd.DataFrame,
    out_path: Path,
    mmd_numeric_col: str = "mmd_numeric_mean",
    mmd_mixed_col: str = "mmd_mixed_mean",
) -> None:
    """任务 4：numeric-only vs mixed MMD 对照分析。"""
    if mmd_numeric_col not in summary_df.columns or mmd_mixed_col not in summary_df.columns:
        raise KeyError(f"summary_by_target.csv 需要包含 {mmd_numeric_col} 与 {mmd_mixed_col}")

    num = summary_df[mmd_numeric_col].to_numpy(dtype=float)
    mix = summary_df[mmd_mixed_col].to_numpy(dtype=float)
    if num.size != mix.size:
        raise ValueError("mmd_numeric_mean 与 mmd_mixed_mean 长度不一致")

    diff = mix - num
    n = diff.size
    mean_diff = float(diff.mean())
    var_diff = float(diff.var(ddof=1)) if n > 1 else float("nan")

    if n > 1:
        t_res = stats.ttest_1samp(diff, 0.0, alternative="greater")
        t_stat = float(t_res.statistic)
        p_one = float(t_res.pvalue)
    else:
        t_stat = float("nan")
        p_one = float("nan")

    out = {
        "n": int(n),
        "mean_mixed_minus_numeric": _float(mean_diff),
        "var_mixed_minus_numeric": _float(var_diff),
        "t_test_one_sided_mixed_gt_numeric": {
            "t_stat": _float(t_stat),
            "p_value": _float(p_one),
        },
    }
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


def task5_outlier_targets(
    summary_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    out_path: Path,
    id_col_summary: str = "target_idx_",
    id_col_pair: str = "target_idx",
    delta_mean_col: str = "delta_min_dist_mean",
    delta_pair_col: str = "delta_min_dist",
    top_k: int = 5,
) -> None:
    """任务 5：找 |delta| 最大的前 top_k 个 target，并输出其 round 级 delta 列表。"""
    if id_col_summary not in summary_df.columns:
        raise KeyError(f"summary_by_target.csv 不含列 {id_col_summary}")
    if delta_mean_col not in summary_df.columns:
        raise KeyError(f"summary_by_target.csv 不含列 {delta_mean_col}")
    if id_col_pair not in pair_df.columns:
        raise KeyError(f"pair_metrics.csv 不含列 {id_col_pair}")
    if delta_pair_col not in pair_df.columns:
        raise KeyError(f"pair_metrics.csv 不含列 {delta_pair_col}")

    sub = summary_df[[id_col_summary, delta_mean_col]].copy()
    sub["abs_delta"] = sub[delta_mean_col].abs()
    sub = sub.sort_values("abs_delta", ascending=False).head(top_k)

    out: Dict[str, Dict[str, object]] = {}
    for _, row in sub.iterrows():
        tid = int(row[id_col_summary])
        mean_delta = float(row[delta_mean_col])
        df_t = pair_df[pair_df[id_col_pair] == tid]
        deltas_round = df_t[delta_pair_col].dropna().to_numpy(dtype=float).tolist()
        out[str(tid)] = {
            "delta_mean": _float(mean_delta),
            "delta_rounds": [_float(x) for x in deltas_round],
        }

    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="对 shadow_pair_metrics 的 summary_by_target/pair_metrics 做统计分析与可视化。",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="单次训练 run 目录（含 shadow_pair_metrics/*）",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default=None,
        help="summary_by_target.csv 路径（默认：run_dir/shadow_pair_metrics/summary_by_target.csv）",
    )
    parser.add_argument(
        "--pair-path",
        type=str,
        default=None,
        help="pair_metrics.csv 路径（默认：run_dir/shadow_pair_metrics/pair_metrics.csv）",
    )
    parser.add_argument(
        "--control-summary-path",
        type=str,
        default=None,
        help="control 对照的 summary_by_target_control.csv 路径（若提供则执行任务 3）",
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir 不存在: {run_dir}")

    default_dir = run_dir / "shadow_pair_metrics"
    summary_path = Path(args.summary_path) if args.summary_path else default_dir / "summary_by_target.csv"
    pair_path = Path(args.pair_path) if args.pair_path else default_dir / "pair_metrics.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"summary_by_target.csv 不存在: {summary_path}")
    if not pair_path.exists():
        raise FileNotFoundError(f"pair_metrics.csv 不存在: {pair_path}")

    summary_df = pd.read_csv(summary_path)
    pair_df = pd.read_csv(pair_path)

    # summary 的 target id 列：pandas groupby 展平后可能是 target_idx 或 target_idx_
    id_col_summary = "target_idx_" if "target_idx_" in summary_df.columns else "target_idx"

    # 任务 1：global_delta_stats.json
    task1_path = default_dir / "global_delta_stats.json"
    task1_global_delta_stats(summary_df, task1_path)

    # 任务 2：delta_distribution.png
    task2_path = default_dir / "delta_distribution.png"
    task2_delta_visualization(summary_df, task2_path)

    # 任务 3：control_comparison.json（仅当提供 control summary 时）
    if args.control_summary_path:
        ctrl_path = Path(args.control_summary_path)
        if not ctrl_path.exists():
            raise FileNotFoundError(f"control summary 文件不存在: {ctrl_path}")
        summary_ctrl = pd.read_csv(ctrl_path)
        task3_path = default_dir / "control_comparison.json"
        task3_control_comparison(summary_df, summary_ctrl, task3_path, id_col=id_col_summary)

    # 任务 4：mmd_component_analysis.json
    task4_path = default_dir / "mmd_component_analysis.json"
    task4_mmd_component_analysis(summary_df, task4_path)

    # 任务 5：outlier_targets.json
    task5_path = default_dir / "outlier_targets.json"
    task5_outlier_targets(summary_df, pair_df, task5_path, id_col_summary=id_col_summary)

    print(f"global_delta_stats.json 已写入: {task1_path}")
    print(f"delta_distribution.png 已写入: {task2_path}")
    if args.control_summary_path:
        print(f"control_comparison.json 已写入: {task3_path}")
    print(f"mmd_component_analysis.json 已写入: {task4_path}")
    print(f"outlier_targets.json 已写入: {task5_path}")


if __name__ == "__main__":
    main()

