#!/usr/bin/env python3
"""
稳健 bridge 分析：在 run_level_bridge_table.csv 基础上补齐稳健稳定性、方向一致性、SNR、
control excess、partial correlation、outlier 敏感性分析与对应图。

输入：run_level_bridge_table.csv（含 adv_abs, s_run, run_dir_path 等）、各 run 的
  shadow_pair_metrics/pair_metrics.csv，若有 control 则 pair_metrics_control.csv。

输出：
  bridge_robust_stability_table.csv
  bridge_with_control_excess.csv（若有 control）
  partial_corr_summary.csv
  sensitivity_remove_outliers.csv
  figA / figA2 / figB / figC / figD（见下）

用法:
  python scripts/build_bridge_robust_analysis.py --table outputs/run_level_bridge_table.csv --outdir outputs/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _choose_delta_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("delta_min_dist", "delta_min_dist_mean"):
        if c in df.columns:
            return c
    return next((c for c in df.columns if "delta" in c.lower()), None)


def _choose_mmd_col(df: pd.DataFrame) -> Optional[str]:
    return "mmd_mixed" if "mmd_mixed" in df.columns else ("mmd_numeric" if "mmd_numeric" in df.columns else None)


def compute_robust_stability_and_direction(
    pair_path: Path,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], int]:
    """
    从 pair_metrics.csv 计算：s_run_mean, s_run_median, s_run_trim95, s_run_p95, n_pairs,
    sign_consistency, sign_bias, delta_mean, delta_std, snr_delta。
    返回 (s_mean, s_median, s_trim95, s_p95, sign_consistency, sign_bias, delta_mean, delta_std, n_pairs)。
    snr_delta 由调用方用 delta_mean/(delta_std+1e-12) 计算。
    """
    if not pair_path.exists():
        return None, None, None, None, None, None, None, None, 0
    df = pd.read_csv(pair_path)
    if df.empty:
        return None, None, None, None, None, None, None, None, 0
    mmd_col = _choose_mmd_col(df)
    delta_col = _choose_delta_col(df)
    if mmd_col is None:
        s_mean = s_median = s_trim95 = s_p95 = None
        n_pairs = 0
    else:
        M = df[mmd_col].astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        n_pairs = len(M)
        if n_pairs == 0:
            s_mean = s_median = s_trim95 = s_p95 = None
        else:
            s_mean = float(np.mean(M))
            s_median = float(np.median(M))
            sorted_m = np.sort(M)
            lo = max(0, int(np.ceil(0.025 * n_pairs)))
            hi = max(lo, min(n_pairs, int(np.floor(0.975 * n_pairs))))
            if hi > lo:
                s_trim95 = float(np.mean(sorted_m[lo:hi]))
            else:
                s_trim95 = s_median
            s_p95 = float(np.percentile(M, 95))
    if delta_col is None or delta_col not in df.columns:
        sign_consistency = sign_bias = delta_mean = delta_std = None
    else:
        D = df[delta_col].astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if len(D) == 0:
            sign_consistency = sign_bias = delta_mean = delta_std = None
        else:
            n_pos = (D > 0).sum()
            sign_consistency = float(n_pos / len(D))
            sign_bias = float(abs(sign_consistency - 0.5))
            delta_mean = float(np.mean(D))
            delta_std = float(np.std(D, ddof=1)) if len(D) > 1 else 0.0
    return s_mean, s_median, s_trim95, s_p95, sign_consistency, sign_bias, delta_mean, delta_std, n_pairs


def compute_control_excess(
    pair_path: Path,
    pair_control_path: Path,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """mean_delta_target, mean_delta_control, excess_signal = mean_target - mean_control。"""
    if not pair_path.exists() or not pair_control_path.exists():
        return None, None, None
    df_t = pd.read_csv(pair_path)
    df_c = pd.read_csv(pair_control_path)
    delta_col = _choose_delta_col(df_t) or _choose_delta_col(df_c)
    if not delta_col or delta_col not in df_t.columns or delta_col not in df_c.columns:
        return None, None, None
    mt = df_t[delta_col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    mc = df_c[delta_col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if mt.empty or mc.empty:
        return None, None, None
    mean_target = float(mt.mean())
    mean_control = float(mc.mean())
    excess = mean_target - mean_control
    return mean_target, mean_control, excess


def build_robust_table(
    bridge_table: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    从 run_level_bridge_table 和每个 run 的 pair_metrics（及可选的 control）构建：
    - robust_df: bridge_robust_stability_table（含 trim95/median/p95, sign_consistency, sign_bias, delta_mean, delta_std, snr_delta, tail_ratio）
    - control_df: bridge_with_control_excess（仅含有 control 的 run）
    """
    robust_rows: List[Dict[str, Any]] = []
    control_rows: List[Dict[str, Any]] = []

    for _, row in bridge_table.iterrows():
        run_dir = Path(row.get("run_dir_path") or row.get("run_dir", ""))
        if not run_dir.exists():
            continue
        pair_path = run_dir / "shadow_pair_metrics" / "pair_metrics.csv"
        pair_control_path = run_dir / "shadow_pair_metrics" / "pair_metrics_control.csv"

        s_mean, s_median, s_trim95, s_p95, sign_cons, sign_bias, delta_mean, delta_std, n_pairs = compute_robust_stability_and_direction(pair_path)
        snr_delta = (delta_mean / (delta_std + 1e-12)) if delta_mean is not None and delta_std is not None else None
        tail_ratio = (s_p95 / (s_median + 1e-12)) if s_p95 is not None and s_median is not None else None

        r = {
            "run_dir": str(run_dir),
            "run_dir_name": row.get("run_dir_name", run_dir.name),
            "model": row.get("model"),
            "train_size": row.get("train_size"),
            "seed": row.get("seed"),
            "adv_abs": row.get("adv_abs", np.nan),
            "s_run_mean": s_mean if s_mean is not None else row.get("s_run", np.nan),
            "s_run_median": s_median,
            "s_run_trim95": s_trim95,
            "s_run_p95": s_p95,
            "n_pairs": n_pairs,
            "sign_consistency": sign_cons,
            "sign_bias": sign_bias,
            "delta_mean": delta_mean,
            "delta_std": delta_std,
            "snr_delta": snr_delta,
            "tail_ratio": tail_ratio,
        }
        robust_rows.append(r)

        if pair_control_path.exists():
            mean_target, mean_control, excess = compute_control_excess(pair_path, pair_control_path)
            if mean_target is not None:
                control_rows.append({
                    **{k: r[k] for k in ["run_dir", "run_dir_name", "model", "train_size", "seed", "adv_abs"]},
                    "mean_delta_target": mean_target,
                    "mean_delta_control": mean_control,
                    "excess_signal": excess,
                })

    robust_df = pd.DataFrame(robust_rows)
    control_df = pd.DataFrame(control_rows) if control_rows else pd.DataFrame()
    return robust_df, control_df


def partial_spearman_pvalue(rho: float, n: int) -> float:
    """H0: partial correlation = 0, t = r * sqrt((n-3)/(1-r^2)), df = n-3."""
    if n <= 3 or not np.isfinite(rho) or abs(rho) >= 1.0:
        return np.nan
    t = rho * np.sqrt((n - 3) / (1 - rho**2))
    return float(2 * stats.t.sf(abs(t), n - 3))


def partial_spearman_xy_given_z(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> Tuple[float, float]:
    """Partial Spearman: corr(x, y | z). 用秩做 Pearson partial。"""
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    n = len(x)
    if n < 4:
        return np.nan, np.nan
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rz = stats.rankdata(z)
    r_xy = np.corrcoef(rx, ry)[0, 1]
    r_xz = np.corrcoef(rx, rz)[0, 1]
    r_yz = np.corrcoef(ry, rz)[0, 1]
    denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    if denom <= 0:
        return np.nan, np.nan
    partial = (r_xy - r_xz * r_yz) / denom
    p = partial_spearman_pvalue(partial, n)
    return float(partial), float(p)


def compute_partial_corr_summary(robust_df: pd.DataFrame, control_df: pd.DataFrame) -> pd.DataFrame:
    """对 all / ctgan / ddpm 分别算 partial_spearman(adv_abs, x | train_size)，x in [trim95, sign_bias, snr_delta, excess_signal]。"""
    rows: List[Dict[str, Any]] = []
    subsets = [
        ("all", robust_df),
        ("ctgan", robust_df[robust_df["model"] == "ctgan"]),
        ("ddpm", robust_df[robust_df["model"] == "ddpm"]),
    ]
    metrics = [
        ("s_run_trim95", "trim95"),
        ("sign_bias", "sign_bias"),
        ("snr_delta", "snr_delta"),
    ]
    for subset_name, sub in subsets:
        if len(sub) < 4:
            continue
        a = sub["adv_abs"].to_numpy(dtype=float)
        z = sub["train_size"].to_numpy(dtype=float)
        for col, label in metrics:
            if col not in sub.columns:
                continue
            x = sub[col].to_numpy(dtype=float)
            rho, p = partial_spearman_xy_given_z(a, x, z)
            rows.append({
                "subset": subset_name,
                "x_metric": label,
                "partial_spearman_rho": rho,
                "p_value": p,
                "n_runs": len(sub),
            })
    if not control_df.empty and "excess_signal" in control_df.columns:
        for subset_name, sub_robust in subsets:
            run_dirs = set(sub_robust["run_dir"].astype(str))
            sub = control_df[control_df["run_dir"].astype(str).isin(run_dirs)]
            if len(sub) < 4:
                continue
            a = sub["adv_abs"].to_numpy(dtype=float)
            z = sub["train_size"].to_numpy(dtype=float)
            x = sub["excess_signal"].to_numpy(dtype=float)
            rho, p = partial_spearman_xy_given_z(a, x, z)
            rows.append({
                "subset": subset_name,
                "x_metric": "excess_signal",
                "partial_spearman_rho": rho,
                "p_value": p,
                "n_runs": len(sub),
            })
    return pd.DataFrame(rows)


def compute_sensitivity_remove_outliers(robust_df: pd.DataFrame) -> pd.DataFrame:
    """
    异常 run：tail_ratio 在 ddpm 内部 top 10% 或 s_run_p95 > 0.5。
    去掉后重算 Spearman(adv_abs, s_run_trim95) 和 Spearman(adv_abs, sign_bias)。
    """
    ddpm = robust_df[robust_df["model"] == "ddpm"].copy()
    if ddpm.empty or "tail_ratio" not in ddpm.columns or ddpm["tail_ratio"].isna().all():
        outlier_mask = (robust_df["s_run_p95"] > 0.5) if "s_run_p95" in robust_df.columns else pd.Series(False, index=robust_df.index)
    else:
        thresh_tail = ddpm["tail_ratio"].quantile(0.9)
        outlier_mask = ((robust_df["model"] == "ddpm") & (robust_df["tail_ratio"] >= thresh_tail)) | (
            (robust_df["s_run_p95"] > 0.5) if "s_run_p95" in robust_df.columns else False
        )
    clean = robust_df[~outlier_mask].dropna(subset=["adv_abs", "s_run_trim95", "sign_bias"])
    if len(clean) < 3:
        return pd.DataFrame()
    rows = []
    for x_col, label in [("s_run_trim95", "s_run_trim95"), ("sign_bias", "sign_bias")]:
        if x_col not in clean.columns:
            continue
        x = clean[x_col].to_numpy(dtype=float)
        y = clean["adv_abs"].to_numpy(dtype=float)
        rho, p = stats.spearmanr(x, y, nan_policy="omit")
        rows.append({
            "x_metric": label,
            "spearman_rho_after_remove_outliers": rho if np.isfinite(rho) else np.nan,
            "p_value": p if np.isfinite(p) else np.nan,
            "n_runs_remaining": len(clean),
            "n_outliers_removed": int(outlier_mask.sum()),
        })
    return pd.DataFrame(rows)


def _scatter_fit(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    out_path: Path,
    title: str,
    xlabel: str,
) -> None:
    df = df.dropna(subset=[x_col, y_col])
    if df.empty or len(df) < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for model in sorted(df["model"].dropna().unique()):
        sub = df[df["model"] == model]
        ax.scatter(sub[x_col], sub[y_col], label=model, alpha=0.8, s=50, edgecolors="k", linewidths=0.5)
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    if x.size > 2 and np.isfinite(x).all() and np.isfinite(y).all():
        r_pearson, p_pearson = stats.linregress(x, y)[2:4]
        rho, p_spearman = stats.spearmanr(x, y, nan_policy="omit")
        xx = np.linspace(x.min(), x.max(), 100)
        ax.plot(xx, stats.linregress(x, y)[0] * xx + stats.linregress(x, y)[1], "k-", linewidth=1.5, label="linear fit")
        sub_r = f"Pearson r={r_pearson:.3f} (p={p_pearson:.4f})"
        sub_s = f"Spearman ρ={rho:.3f} (p={p_spearman:.4f})" if np.isfinite(rho) else "Spearman N/A"
        ax.set_title(f"{title}\n{sub_r}; {sub_s}")
    else:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("adv_abs = |AUC − 0.5|")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="稳健 bridge 分析：稳健稳定性、方向性、SNR、partial corr、敏感性、图")
    parser.add_argument("--table", type=Path, default=PROJECT_ROOT / "outputs" / "run_level_bridge_table.csv", help="run_level_bridge_table.csv")
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "outputs" / "figures", help="输出目录（表与图）")
    args = parser.parse_args()

    table_path = args.table.resolve()
    if not table_path.exists():
        print(f"未找到桥接表: {table_path}")
        return
    bridge_table = pd.read_csv(table_path)
    if bridge_table.empty:
        print("桥接表为空")
        return

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) 稳健稳定性 + 方向一致性 + SNR → bridge_robust_stability_table.csv
    robust_df, control_df = build_robust_table(bridge_table)
    robust_path = outdir / "bridge_robust_stability_table.csv"
    robust_df.to_csv(robust_path, index=False, float_format="%.6f")
    print(f"已写: {robust_path}")

    # 2) Control excess → bridge_with_control_excess.csv
    if not control_df.empty:
        control_path = outdir / "bridge_with_control_excess.csv"
        control_df.to_csv(control_path, index=False, float_format="%.6f")
        print(f"已写: {control_path}")
    else:
        control_path = None

    # 3) Partial correlation
    partial_df = compute_partial_corr_summary(robust_df, control_df)
    partial_path = outdir / "partial_corr_summary.csv"
    partial_df.to_csv(partial_path, index=False, float_format="%.6f")
    print(f"已写: {partial_path}")

    # 4) Sensitivity remove outliers
    sens_df = compute_sensitivity_remove_outliers(robust_df)
    sens_path = outdir / "sensitivity_remove_outliers.csv"
    sens_df.to_csv(sens_path, index=False, float_format="%.6f")
    print(f"已写: {sens_path}")

    # 5) 图 A / A2 / B / C / D
    _scatter_fit(
        robust_df, "s_run_trim95", "adv_abs",
        outdir / "figA_adv_abs_vs_s_run_trim95.png",
        "adv_abs vs s_run_trim95 (robust stability)",
        "s_run_trim95",
    )
    _scatter_fit(
        robust_df, "s_run_median", "adv_abs",
        outdir / "figA2_adv_abs_vs_s_run_median.png",
        "adv_abs vs s_run_median (robust stability)",
        "s_run_median",
    )
    _scatter_fit(
        robust_df, "sign_bias", "adv_abs",
        outdir / "figB_adv_abs_vs_sign_bias.png",
        "adv_abs vs sign_bias (direction consistency)",
        "sign_bias = |sign_consistency − 0.5|",
    )
    _scatter_fit(
        robust_df, "snr_delta", "adv_abs",
        outdir / "figC_adv_abs_vs_snr_delta.png",
        "adv_abs vs snr_delta (signal vs noise)",
        "snr_delta",
    )
    if not control_df.empty and "excess_signal" in control_df.columns:
        _scatter_fit(
            control_df, "excess_signal", "adv_abs",
            outdir / "figD_adv_abs_vs_excess_signal.png",
            "adv_abs vs excess_signal (membership-specific)",
            "excess_signal",
        )
        print("已写: figD_adv_abs_vs_excess_signal.png")

    # 6) 解释口径（console）
    print("\n" + "=" * 60)
    print("稳健 bridge 分析 — 关键解释口径")
    print("=" * 60)
    trim95_rho = None
    if "s_run_trim95" in robust_df.columns and "adv_abs" in robust_df.columns:
        sub = robust_df.dropna(subset=["s_run_trim95", "adv_abs"])
        if len(sub) >= 3:
            trim95_rho, _ = stats.spearmanr(sub["s_run_trim95"], sub["adv_abs"], nan_policy="omit")
    partial_trim = partial_df[(partial_df["x_metric"] == "trim95") & (partial_df["subset"] == "all")]
    partial_trim_rho = partial_trim["partial_spearman_rho"].iloc[0] if len(partial_trim) else None
    partial_sign = partial_df[(partial_df["x_metric"] == "sign_bias") & (partial_df["subset"] == "all")]
    partial_snr = partial_df[(partial_df["x_metric"] == "snr_delta") & (partial_df["subset"] == "all")]

    if trim95_rho is not None and partial_trim_rho is not None:
        if trim95_rho < 0 and partial_trim_rho < 0:
            print(
                "[解读] adv_abs 与 s_run_trim95 仍负相关，且 partial correlation 仍为负 → "
                "replace-one distribution shift 主要表现为非定向噪声，不转化为 membership advantage"
            )
        elif (partial_sign["partial_spearman_rho"].abs().max() if len(partial_sign) else 0) > 0.3 or (partial_snr["partial_spearman_rho"].abs().max() if len(partial_snr) else 0) > 0.3:
            print(
                "[解读] 负相关消失但 adv_abs 与 sign_bias / snr_delta 相关 → "
                "攻击优势来自方向一致性信号，而不是总体分布漂移（MMD）"
            )
        else:
            print("[解读] 仅 mean 相关、trim95/median 不相关 → 原相关由极端不稳定 outlier 驱动（训练失败/模式崩溃）")
    print("=" * 60)
    print("必出文件清单:")
    print(f"  {robust_path.name}")
    if control_path:
        print(f"  {control_path.name}")
    print(f"  {partial_path.name}")
    print(f"  {sens_path.name}")
    print("  图: figA_adv_abs_vs_s_run_trim95.png, figA2_adv_abs_vs_s_run_median.png, figB_adv_abs_vs_sign_bias.png, figC_adv_abs_vs_snr_delta.png" + (", figD_adv_abs_vs_excess_signal.png" if not control_df.empty else ""))


if __name__ == "__main__":
    main()
