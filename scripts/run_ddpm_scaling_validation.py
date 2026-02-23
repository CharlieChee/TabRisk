#!/usr/bin/env python3
"""
验证 DDPM 反弹是否统计显著且稳健。

输入：run_level_adv_abs.csv, run_level_adv_abs_density.csv, run_level_adv_abs_clf.csv（仅用 model=="ddpm"）。
输出：ddpm_scaling_significance_summary.csv, ddpm_bootstrap_by_N.csv, ddpm_leave_one_out_slopes.csv,
      ddpm_permutation_test.csv, fig_ddpm_bootstrap_scaling.png，及终端判定与总结。

用法:
  python scripts/run_ddpm_scaling_validation.py --indir outputs/figures --outdir outputs/figures
  --n_jobs N  并行进程数（PART 2/3/4 的 bootstrap、LOO、permutation）
"""

from __future__ import annotations

import argparse
import multiprocessing
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
N_BOOTSTRAP = 2000
N_PERM = 5000
ALPHA = 0.05

ATTACK_CONFIG = [
    ("NN", "adv_abs", "run_level_adv_abs.csv"),
    ("density", "adv_abs_density", "run_level_adv_abs_density.csv"),
    ("classifier", "adv_abs_clf", "run_level_adv_abs_clf.csv"),
]


def load_ddpm_data(indir: Path) -> pd.DataFrame:
    """合并三个 run_level 表，只保留 model=='ddpm'，列 run_id, train_size, adv_abs, adv_abs_density, adv_abs_clf。"""
    indir = Path(indir)
    adv_path = indir / "run_level_adv_abs.csv"
    if not adv_path.exists():
        raise FileNotFoundError(f"需要 {adv_path}")
    df = pd.read_csv(adv_path)
    df = df[df["model"] == "ddpm"].copy()
    if df.empty:
        return df
    df = df[["run_id", "train_size", "adv_abs"]].copy()
    df = df.rename(columns={"adv_abs": "adv_abs_NN"})

    for _attack_name, col_val, fname in [("density", "adv_abs_density", "run_level_adv_abs_density.csv"), ("classifier", "adv_abs_clf", "run_level_adv_abs_clf.csv")]:
        path = indir / fname
        if not path.exists():
            df[col_val] = np.nan
            continue
        other = pd.read_csv(path)
        if "model" not in other.columns or col_val not in other.columns:
            df[col_val] = np.nan
            continue
        other = other[other["model"] == "ddpm"][["run_id", "train_size", col_val]]
        df = df.merge(other, on=["run_id", "train_size"], how="left")
    df["adv_abs"] = df["adv_abs_NN"]
    return df


def get_attack_series(df: pd.DataFrame, value_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (train_size, value) 且 dropna。"""
    sub = df[["train_size", value_col]].dropna()
    return sub["train_size"].to_numpy(dtype=float), sub[value_col].to_numpy(dtype=float)


def part1_significance(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    """PART 1: 线性、1/sqrt、Spearman，输出 ddpm_scaling_significance_summary.csv。"""
    rows = []
    for attack_type, value_col, _ in ATTACK_CONFIG:
        if value_col not in df.columns:
            continue
        x, y = get_attack_series(df, value_col)
        n = len(x)
        if n < 3:
            rows.append({
                "attack_type": attack_type,
                "slope_linear": np.nan, "p_linear": np.nan,
                "slope_inv_sqrt": np.nan, "p_inv_sqrt": np.nan,
                "spearman_r": np.nan, "spearman_p": np.nan,
                "n_runs": n,
            })
            continue
        log_n = np.log(x)
        inv_sqrt_n = 1.0 / np.sqrt(x)
        slope_lin, intercept_lin, r_lin, p_lin, _ = stats.linregress(log_n, y)
        slope_inv, _, r_inv, p_inv, _ = stats.linregress(inv_sqrt_n, y)
        sp_r, sp_p = stats.spearmanr(x, y)
        rows.append({
            "attack_type": attack_type,
            "slope_linear": slope_lin, "p_linear": p_lin,
            "slope_inv_sqrt": slope_inv, "p_inv_sqrt": p_inv,
            "spearman_r": sp_r if np.isfinite(sp_r) else np.nan,
            "spearman_p": sp_p if np.isfinite(sp_p) else np.nan,
            "n_runs": n,
        })
        sig_lin = slope_lin > 0 and p_lin < ALPHA
        sig_sp = np.isfinite(sp_p) and sp_p < ALPHA
        status = "" if (sig_lin or sig_sp) else "NOT statistically significant"
        print(f"  [{attack_type}] slope_linear={slope_lin:.4f} p={p_lin:.4f} (slope>0 & p<0.05: {sig_lin}); Spearman p={sp_p:.4f} (p<0.05: {sig_sp}) {status}")
    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "ddpm_scaling_significance_summary.csv", index=False)
    return summary


def _worker_bootstrap(args: Tuple[str, int, List[float], int, int]) -> Dict[str, Any]:
    """(attack_type, train_size, values_list, n_bootstrap, seed) -> one row."""
    attack_type, train_size, values_list, n_bootstrap, seed = args
    pool = np.array(values_list, dtype=float)
    n = len(pool)
    if n == 0:
        return {"train_size": train_size, "attack_type": attack_type, "mean": np.nan, "CI_low": np.nan, "CI_high": np.nan}
    rng = np.random.default_rng(seed)
    boot_means = [np.mean(pool[rng.integers(0, n, size=n)]) for _ in range(n_bootstrap)]
    boot_means = np.array(boot_means)
    return {
        "train_size": int(train_size),
        "attack_type": attack_type,
        "mean": float(np.mean(pool)),
        "CI_low": float(np.percentile(boot_means, 2.5)),
        "CI_high": float(np.percentile(boot_means, 97.5)),
    }


def part2_bootstrap(df: pd.DataFrame, outdir: Path, n_jobs: int = 1) -> pd.DataFrame:
    """PART 2: 按 (attack_type, train_size) bootstrap 2000 次，得 mean 与 95% CI。n_jobs>1 时多进程。"""
    tasks = []
    for attack_type, value_col, _ in ATTACK_CONFIG:
        if value_col not in df.columns:
            continue
        sub = df[["run_id", "train_size", value_col]].dropna()
        if sub.empty:
            continue
        for train_size in sub["train_size"].unique():
            pool = sub[sub["train_size"] == train_size][value_col].tolist()
            if len(pool) == 0:
                continue
            tasks.append((attack_type, int(train_size), pool, N_BOOTSTRAP, 42 + len(tasks)))
    if n_jobs is None or n_jobs <= 1:
        rows = [_worker_bootstrap(t) for t in tasks]
    else:
        with multiprocessing.Pool(processes=n_jobs) as pool:
            rows = pool.map(_worker_bootstrap, tasks)
    boot_df = pd.DataFrame(rows)
    boot_df.to_csv(outdir / "ddpm_bootstrap_by_N.csv", index=False)
    return boot_df


def plot_bootstrap(boot_df: pd.DataFrame, outdir: Path) -> None:
    """fig_ddpm_bootstrap_scaling.png: 横轴 log(N)，纵轴 adv_abs，每攻击族一条线 + error bar。"""
    if boot_df.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for attack_type in boot_df["attack_type"].unique():
        sub = boot_df[boot_df["attack_type"] == attack_type].sort_values("train_size")
        x = sub["train_size"].to_numpy(dtype=float)
        y = sub["mean"].to_numpy(dtype=float)
        lo = sub["CI_low"].to_numpy(dtype=float)
        hi = sub["CI_high"].to_numpy(dtype=float)
        ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o-", capsize=4, label=attack_type)
    ax.set_xscale("log")
    ax.set_xlabel("train_size N (log)")
    ax.set_ylabel("adv_abs")
    ax.set_title("DDPM scaling: bootstrap 95% CI by attack type")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(outdir / "fig_ddpm_bootstrap_scaling.png", dpi=200)
    plt.close(fig)


def part2_judge(boot_df: pd.DataFrame) -> None:
    """相邻 N 的 CI 是否重叠；最大 N 是否显著高于最小 N。"""
    if boot_df.empty:
        return
    print("\n  [Bootstrap] 按 attack_type 检查相邻 N 的 CI 重叠与 max vs min N：")
    for attack_type in boot_df["attack_type"].unique():
        sub = boot_df[boot_df["attack_type"] == attack_type].sort_values("train_size")
        if len(sub) < 2:
            continue
        train_sizes = sub["train_size"].to_numpy()
        ci_lo = sub["CI_low"].to_numpy()
        ci_hi = sub["CI_high"].to_numpy()
        overlaps = []
        for i in range(len(sub) - 1):
            overlap = not (ci_hi[i] < ci_lo[i + 1] or ci_hi[i + 1] < ci_lo[i])
            overlaps.append(overlap)
        max_higher_than_min = ci_lo[-1] > ci_hi[0]
        print(f"    {attack_type}: 相邻 N CI 重叠={overlaps}; 最大 N 下限 > 最小 N 上限 (单调上升支持) = {max_higher_than_min}")


def _worker_loo(args: Tuple[str, Any, List[Any], List[float], List[float]]) -> Dict[str, Any]:
    """(attack_type, removed_run_id, run_ids, train_sizes, values) -> one row."""
    attack_type, removed_run_id, run_ids, train_sizes, values = args
    run_ids = list(run_ids)
    train_sizes = np.array(train_sizes, dtype=float)
    values = np.array(values, dtype=float)
    mask = np.array([r != removed_run_id for r in run_ids])
    x = train_sizes[mask]
    y = values[mask]
    if len(x) < 3:
        return {"attack_type": attack_type, "removed_run_id": removed_run_id, "slope": np.nan, "p_value": np.nan}
    try:
        slope, _, _, p, _ = stats.linregress(np.log(x), y)
        return {"attack_type": attack_type, "removed_run_id": removed_run_id, "slope": float(slope), "p_value": float(p)}
    except Exception:
        return {"attack_type": attack_type, "removed_run_id": removed_run_id, "slope": np.nan, "p_value": np.nan}


def part3_loo(df: pd.DataFrame, outdir: Path, n_jobs: int = 1) -> pd.DataFrame:
    """PART 3: 每个 run 删除后重新拟合 slope，输出 ddpm_leave_one_out_slopes.csv。n_jobs>1 时多进程。"""
    tasks = []
    for attack_type, value_col, _ in ATTACK_CONFIG:
        if value_col not in df.columns:
            continue
        sub = df[["run_id", "train_size", value_col]].dropna()
        if len(sub) < 4:
            continue
        run_ids = sub["run_id"].tolist()
        train_sizes = sub["train_size"].tolist()
        values = sub[value_col].tolist()
        for removed in sub["run_id"].unique():
            tasks.append((attack_type, removed, run_ids, train_sizes, values))
    if n_jobs is None or n_jobs <= 1:
        rows = [_worker_loo(t) for t in tasks]
    else:
        with multiprocessing.Pool(processes=n_jobs) as pool:
            rows = pool.map(_worker_loo, tasks)
    loo_df = pd.DataFrame(rows)
    loo_df.to_csv(outdir / "ddpm_leave_one_out_slopes.csv", index=False)
    return loo_df


def part3_judge(loo_df: pd.DataFrame) -> None:
    """slope 是否所有 LOO 仍 >0；符号是否稳定；p 是否持续显著；是否被单 run 驱动。"""
    if loo_df.empty:
        return
    print("\n  [Leave-One-Out] 按 attack_type：")
    for attack_type in loo_df["attack_type"].unique():
        sub = loo_df[loo_df["attack_type"] == attack_type].dropna(subset=["slope"])
        if sub.empty:
            continue
        all_positive = (sub["slope"] > 0).all()
        sign_stable = (sub["slope"] > 0).all() or (sub["slope"] < 0).all()
        all_sig = (sub["p_value"] < ALPHA).all()
        driven = not all_positive or not all_sig
        print(f"    {attack_type}: slope 全部>0={all_positive}, 符号稳定={sign_stable}, p 持续<0.05={all_sig}; Driven by specific run={driven}")


def _worker_permutation(args: Tuple[str, List[float], List[float], int, int]) -> Dict[str, Any]:
    """(attack_type, x_list, y_list, n_perm, seed) -> one row with true_slope, null_mean, null_std, empirical_p_value."""
    attack_type, x_list, y_list, n_perm, seed = args
    x = np.array(x_list, dtype=float)
    y = np.array(y_list, dtype=float)
    n = len(x)
    if n < 3:
        return {"attack_type": attack_type, "true_slope": np.nan, "null_mean": np.nan, "null_std": np.nan, "empirical_p_value": np.nan}
    rng = np.random.default_rng(seed)
    log_n = np.log(x)
    true_slope, _, _, _, _ = stats.linregress(log_n, y)
    null_slopes = []
    for _ in range(n_perm):
        perm_idx = rng.permutation(n)
        try:
            s, _, _, _, _ = stats.linregress(np.log(x[perm_idx]), y)
            null_slopes.append(s)
        except Exception:
            pass
    null_slopes = np.array(null_slopes)
    null_mean = float(np.mean(null_slopes))
    null_std = float(np.std(null_slopes))
    empirical_p = float(np.mean(null_slopes >= true_slope))
    if empirical_p == 0:
        empirical_p = 1.0 / (n_perm + 1)
    return {
        "attack_type": attack_type,
        "true_slope": true_slope,
        "null_mean": null_mean,
        "null_std": null_std,
        "empirical_p_value": empirical_p,
    }


def part4_permutation(df: pd.DataFrame, outdir: Path, n_jobs: int = 1) -> pd.DataFrame:
    """PART 4: 打乱 train_size 5000 次，得 null distribution，empirical p。n_jobs>1 时按 attack_type 并行。"""
    tasks = []
    for attack_type, value_col, _ in ATTACK_CONFIG:
        if value_col not in df.columns:
            continue
        x, y = get_attack_series(df, value_col)
        if len(x) < 3:
            tasks.append((attack_type, [], [], 0, 42))
            continue
        tasks.append((attack_type, x.tolist(), y.tolist(), N_PERM, 42 + len(tasks)))
    if n_jobs is None or n_jobs <= 1:
        rows = [_worker_permutation(t) for t in tasks]
    else:
        with multiprocessing.Pool(processes=min(n_jobs, len(tasks)) or 1) as pool:
            rows = pool.map(_worker_permutation, tasks)
    perm_df = pd.DataFrame(rows)
    perm_df.to_csv(outdir / "ddpm_permutation_test.csv", index=False)
    for _, r in perm_df.iterrows():
        if np.isfinite(r.get("empirical_p_value", np.nan)):
            reject = r["empirical_p_value"] < ALPHA
            print(f"  [{r['attack_type']}] true_slope={r['true_slope']:.4f} empirical_p={r['empirical_p_value']:.4f} -> {'DDPM positive scaling unlikely due to random assignment' if reject else 'Cannot reject null hypothesis of random fluctuation'}")
    return perm_df


def part5_summary(
    summary_df: pd.DataFrame,
    boot_df: pd.DataFrame,
    loo_df: pd.DataFrame,
    perm_df: pd.DataFrame,
) -> None:
    """PART 5: 汇总并打印 Conclusion。"""
    print("\n" + "=" * 60)
    print("DDPM scaling 验证总结")
    print("=" * 60)
    all_positive = True
    all_sig = True
    bootstrap_ok = True
    loo_ok = True
    perm_ok = True
    if not summary_df.empty:
        for _, r in summary_df.iterrows():
            if np.isfinite(r.get("slope_linear", np.nan)):
                if r["slope_linear"] <= 0:
                    all_positive = False
                if r["p_linear"] >= ALPHA:
                    all_sig = False
        print(f"  1) 是否所有攻击族 slope 为正: {all_positive}")
        print(f"  2) 是否统计显著 (p_linear < 0.05): {all_sig}")
    if not boot_df.empty:
        for attack_type in boot_df["attack_type"].unique():
            sub = boot_df[boot_df["attack_type"] == attack_type].sort_values("train_size")
            if len(sub) < 2:
                continue
            if sub["CI_low"].iloc[-1] <= sub["CI_high"].iloc[0]:
                bootstrap_ok = False
                break
        print(f"  3) Bootstrap CI 是否支持单调上升 (最大 N 下限 > 最小 N 上限): {bootstrap_ok}")
    if not loo_df.empty:
        for attack_type in loo_df["attack_type"].unique():
            s = loo_df[loo_df["attack_type"] == attack_type]
            if (s["slope"] <= 0).any() or (s["p_value"] >= ALPHA).any():
                loo_ok = False
                break
        print(f"  4) Leave-one-out 是否稳健 (所有 LOO slope>0 且 p<0.05): {loo_ok}")
    if not perm_df.empty:
        for _, r in perm_df.iterrows():
            if np.isfinite(r.get("empirical_p_value", np.nan)) and r["empirical_p_value"] >= ALPHA:
                perm_ok = False
                break
        print(f"  5) Permutation test 是否支持 (empirical_p < 0.05): {perm_ok}")
    # Conclusion
    real_supported = all_positive and all_sig and perm_ok
    if real_supported and loo_ok and bootstrap_ok:
        confidence = "HIGH"
    elif real_supported:
        confidence = "MODERATE"
    else:
        confidence = "LOW"
    conclusion = "REAL" if real_supported else "NOT statistically supported"
    print("\n  Conclusion:")
    print(f"  - DDPM scaling {conclusion}")
    print(f"  - Confidence level: {confidence}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="DDPM 反弹统计显著性与稳健性验证")
    parser.add_argument("--indir", type=Path, default=PROJECT_ROOT / "outputs" / "figures", help="输入目录（含 run_level_adv_abs 等 CSV）")
    parser.add_argument("--outdir", type=Path, default=None, help="输出目录，默认同 indir")
    parser.add_argument("--n_jobs", type=int, default=1, help="并行进程数，>1 时 PART 2/3/4 多进程")
    args = parser.parse_args()
    args.outdir = args.outdir or args.indir
    args.outdir.mkdir(parents=True, exist_ok=True)
    n_jobs = args.n_jobs or 1
    print("DDPM scaling 验证（仅 model==ddpm）")
    print(f"  输入: {args.indir}, 输出: {args.outdir}, n_jobs: {n_jobs}")
    df = load_ddpm_data(args.indir)
    if df.empty:
        print("无 DDPM 数据，请先生成 run_level_adv_abs.csv 等并包含 model=='ddpm' 行")
        return
    print(f"  DDPM runs: {len(df)}")
    print("\n--- PART 1: Slope 显著性 ---")
    summary_df = part1_significance(df, args.outdir)
    print("\n--- PART 2: Bootstrap ---")
    boot_df = part2_bootstrap(df, args.outdir, n_jobs=n_jobs)
    plot_bootstrap(boot_df, args.outdir)
    part2_judge(boot_df)
    print("\n--- PART 3: Leave-One-Out ---")
    loo_df = part3_loo(df, args.outdir, n_jobs=n_jobs)
    part3_judge(loo_df)
    print("\n--- PART 4: Permutation Test ---")
    perm_df = part4_permutation(df, args.outdir, n_jobs=n_jobs)
    part5_summary(summary_df, boot_df, loo_df, perm_df)
    print("\n已写出: ddpm_scaling_significance_summary.csv, ddpm_bootstrap_by_N.csv, ddpm_leave_one_out_slopes.csv, ddpm_permutation_test.csv, fig_ddpm_bootstrap_scaling.png")


if __name__ == "__main__":
    main()
