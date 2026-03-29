#!/usr/bin/env python3
"""
最终四项验证分析：mediation / interaction / per-model / bound。

输入：bridge_robust_stability_table.csv、bridge_with_control_excess.csv（或已有汇总表），
  需含 run_dir, model, train_size, adv_abs, s_run_trim95, sign_bias, snr_delta, excess_signal（可选）。

0) 预处理 → run_level_final.csv（N_z、删缺失）
1) Mediation → mediation_models_summary.csv + fig_med1/med2/med3
2) Interaction → interaction_binned_spearman.csv + fig_interaction_signbias_by_Nbin.png
3) Per-model → per_model_mechanism_summary.csv + fig_pm_*
4) Bound → bound_proxy_summary.csv + fig_bound_adv_vs_sqrt_excess.png

用法:
  python scripts/run_final_four_validations.py --robust outputs/figures/bridge_robust_stability_table.csv --control outputs/figures/bridge_with_control_excess.csv --outdir outputs/figures
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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _ols_summary(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """OLS 系数、标准误、p-value、R²、n。X 含 intercept 列。"""
    n, k = X.shape
    if n <= k:
        return np.full(k, np.nan), np.full(k, np.nan), np.full(k, np.nan), np.nan, n
    reg = LinearRegression(fit_intercept=False).fit(X, y)
    beta = reg.coef_
    yhat = X @ beta
    resid = y - yhat
    rss = (resid ** 2).sum()
    df = n - k
    if df <= 0:
        return beta, np.full(k, np.nan), np.full(k, np.nan), np.nan, n
    try:
        inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return beta, np.full(k, np.nan), np.full(k, np.nan), np.nan, n
    se2 = (rss / df) * np.diag(inv)
    se = np.sqrt(np.maximum(se2, 0))
    t = np.where(se > 0, beta / se, 0)
    p = 2 * (1 - stats.t.cdf(np.abs(t), df))
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (rss / ss_tot) if ss_tot > 0 else np.nan
    return beta, se, p, float(r2), n


def build_run_level_final(
    robust_path: Path,
    control_path: Optional[Path],
    outdir: Path,
) -> pd.DataFrame:
    """
    合并 robust + control（on run_dir），删缺失，生成 N_z，写出 run_level_final.csv。
    """
    robust = pd.read_csv(robust_path)
    if control_path and control_path.exists():
        control = pd.read_csv(control_path)
        df = robust.merge(
            control[["run_dir", "mean_delta_target", "mean_delta_control", "excess_signal"]],
            on="run_dir",
            how="left",
        )
    else:
        df = robust.copy()
        df["excess_signal"] = np.nan
    required = ["run_dir", "model", "train_size", "adv_abs", "s_run_trim95", "sign_bias", "snr_delta"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"缺少列: {c}")
    df = df.dropna(subset=required).copy()
    df["N_z"] = (df["train_size"] - df["train_size"].mean()) / (df["train_size"].std() + 1e-12)
    outpath = outdir / "run_level_final.csv"
    df.to_csv(outpath, index=False, float_format="%.6f")
    print(f"已写: {outpath} (n={len(df)})")
    return df


def run_mediation(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    """M1–M4 及可选 M1b/M3b/M4b，输出 mediation_models_summary.csv 和三张图。"""
    df = df.copy()
    df["model_ddpm"] = (df["model"] == "ddpm").astype(int)
    y = df["adv_abs"].to_numpy(dtype=float)
    N_z = df["N_z"].to_numpy(dtype=float)
    model_ddpm = df["model_ddpm"].to_numpy(dtype=float)
    s_run = df["s_run_trim95"].to_numpy(dtype=float)
    sign_b = df["sign_bias"].to_numpy(dtype=float)
    excess = df["excess_signal"].to_numpy(dtype=float)
    has_excess = np.isfinite(excess).sum() >= 4 and (np.isfinite(excess) & (excess != 0)).any()

    def _fit_adv(X_cols: List[str], names: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
        X = np.column_stack([df[c].to_numpy(dtype=float) for c in X_cols])
        X = np.column_stack([np.ones(len(X)), X])
        return _ols_summary(X, y, ["intercept"] + names)

    def _row_base(model_id: str, formula: str, n: int, r2: float) -> Dict[str, Any]:
        return {"model_id": model_id, "formula": formula, "n": n, "R2": r2}

    rows = []
    # M1: adv_abs ~ s_run_trim95 + N_z + I(model)
    b1, se1, p1, r2_1, n1 = _fit_adv(["s_run_trim95", "N_z", "model_ddpm"], ["s_run_trim95", "N_z", "model_ddpm"])
    rows.append({
        **_row_base("M1", "adv_abs ~ s_run_trim95 + N_z + I(model)", n1, r2_1),
        "beta_s_run_trim95": b1[1], "p_s_run_trim95": p1[1],
        "beta_sign_bias": np.nan, "p_sign_bias": np.nan,
        "beta_excess_signal": np.nan, "p_excess_signal": np.nan,
    })
    # M2: adv_abs ~ sign_bias + N_z + I(model)
    b2, se2, p2, r2_2, n2 = _fit_adv(["sign_bias", "N_z", "model_ddpm"], ["sign_bias", "N_z", "model_ddpm"])
    rows.append({
        **_row_base("M2", "adv_abs ~ sign_bias + N_z + I(model)", n2, r2_2),
        "beta_s_run_trim95": np.nan, "p_s_run_trim95": np.nan,
        "beta_sign_bias": b2[1], "p_sign_bias": p2[1],
        "beta_excess_signal": np.nan, "p_excess_signal": np.nan,
    })
    # M3: adv_abs ~ s_run_trim95 + sign_bias + N_z + I(model)
    b3, se3, p3, r2_3, n3 = _fit_adv(["s_run_trim95", "sign_bias", "N_z", "model_ddpm"], ["s_run_trim95", "sign_bias", "N_z", "model_ddpm"])
    rows.append({
        **_row_base("M3", "adv_abs ~ s_run_trim95 + sign_bias + N_z + I(model)", n3, r2_3),
        "beta_s_run_trim95": b3[1], "p_s_run_trim95": p3[1],
        "beta_sign_bias": b3[2], "p_sign_bias": p3[2],
        "beta_excess_signal": np.nan, "p_excess_signal": np.nan,
    })
    # M4: sign_bias ~ s_run_trim95 + N_z + I(model)
    X4 = np.column_stack([np.ones(len(df)), s_run, N_z, model_ddpm])
    b4, se4, p4, r2_4, n4 = _ols_summary(X4, sign_b, ["intercept", "s_run_trim95", "N_z", "model_ddpm"])
    rows.append({
        **_row_base("M4", "sign_bias ~ s_run_trim95 + N_z + I(model)", n4, r2_4),
        "beta_s_run_trim95": b4[1], "p_s_run_trim95": p4[1],
        "beta_sign_bias": np.nan, "p_sign_bias": np.nan,
        "beta_excess_signal": np.nan, "p_excess_signal": np.nan,
    })

    if has_excess:
        df_ex = df.dropna(subset=["excess_signal"]).copy()
        if len(df_ex) >= 4:
            y_ex = df_ex["adv_abs"].to_numpy(dtype=float)
            sb_ex = df_ex["sign_bias"].to_numpy(dtype=float)
            X1b = np.column_stack([np.ones(len(df_ex)), df_ex["excess_signal"].to_numpy(dtype=float), df_ex["N_z"].to_numpy(dtype=float), df_ex["model_ddpm"].to_numpy(dtype=float)])
            b1b, _, p1b, r2_1b, n1b = _ols_summary(X1b, y_ex, ["intercept", "excess_signal", "N_z", "model_ddpm"])
            rows.append({
                **_row_base("M1b", "adv_abs ~ excess_signal + N_z + I(model)", n1b, r2_1b),
                "beta_s_run_trim95": np.nan, "p_s_run_trim95": np.nan,
                "beta_sign_bias": np.nan, "p_sign_bias": np.nan,
                "beta_excess_signal": b1b[1], "p_excess_signal": p1b[1],
            })
            X3b = np.column_stack([np.ones(len(df_ex)), df_ex["excess_signal"].to_numpy(dtype=float), df_ex["sign_bias"].to_numpy(dtype=float), df_ex["N_z"].to_numpy(dtype=float), df_ex["model_ddpm"].to_numpy(dtype=float)])
            b3b, _, p3b, r2_3b, n3b = _ols_summary(X3b, y_ex, ["intercept", "excess_signal", "sign_bias", "N_z", "model_ddpm"])
            rows.append({
                **_row_base("M3b", "adv_abs ~ excess_signal + sign_bias + N_z + I(model)", n3b, r2_3b),
                "beta_s_run_trim95": np.nan, "p_s_run_trim95": np.nan,
                "beta_sign_bias": b3b[2], "p_sign_bias": p3b[2],
                "beta_excess_signal": b3b[1], "p_excess_signal": p3b[1],
            })
            X4b = np.column_stack([np.ones(len(df_ex)), df_ex["excess_signal"].to_numpy(dtype=float), df_ex["N_z"].to_numpy(dtype=float), df_ex["model_ddpm"].to_numpy(dtype=float)])
            b4b, _, p4b, r2_4b, n4b = _ols_summary(X4b, sb_ex, ["intercept", "excess_signal", "N_z", "model_ddpm"])
            rows.append({
                **_row_base("M4b", "sign_bias ~ excess_signal + N_z + I(model)", n4b, r2_4b),
                "beta_s_run_trim95": np.nan, "p_s_run_trim95": np.nan,
                "beta_sign_bias": np.nan, "p_sign_bias": np.nan,
                "beta_excess_signal": b4b[1], "p_excess_signal": p4b[1],
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "mediation_models_summary.csv", index=False, float_format="%.6f")
    print(f"已写: {outdir / 'mediation_models_summary.csv'}")

    # 图 Med-1: adv_abs vs s_run_trim95
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in df["model"].unique():
        sub = df[df["model"] == m]
        ax.scatter(sub["s_run_trim95"], sub["adv_abs"], label=m, alpha=0.8, s=50)
    x = df["s_run_trim95"].to_numpy(dtype=float)
    y = df["adv_abs"].to_numpy(dtype=float)
    if len(x) > 1 and np.isfinite(x).all() and np.isfinite(y).all():
        slope, intercept, r, p, _ = stats.linregress(x, y)
        xx = np.linspace(x.min(), x.max(), 100)
        ax.plot(xx, slope * xx + intercept, "k-", lw=1.5, label="fit all")
    ax.set_xlabel("s_run_trim95")
    ax.set_ylabel("adv_abs")
    ax.set_title("Med-1: adv_abs vs s_run_trim95")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(outdir / "fig_med1_adv_vs_srun_trim95.png", dpi=200)
    plt.close(fig)

    # 图 Med-2: adv_abs vs sign_bias
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in df["model"].unique():
        sub = df[df["model"] == m]
        ax.scatter(sub["sign_bias"], sub["adv_abs"], label=m, alpha=0.8, s=50)
    x = df["sign_bias"].to_numpy(dtype=float)
    y = df["adv_abs"].to_numpy(dtype=float)
    if len(x) > 1 and np.isfinite(x).all() and np.isfinite(y).all():
        slope, intercept, r, p, _ = stats.linregress(x, y)
        xx = np.linspace(x.min(), x.max(), 100)
        ax.plot(xx, slope * xx + intercept, "k-", lw=1.5, label="fit all")
    ax.set_xlabel("sign_bias")
    ax.set_ylabel("adv_abs")
    ax.set_title("Med-2: adv_abs vs sign_bias")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(outdir / "fig_med2_adv_vs_sign_bias.png", dpi=200)
    plt.close(fig)

    # 图 Med-3: partial plot — residual(adv_abs ~ N_z + model) vs residual(sign_bias ~ N_z + model)
    X_partial = np.column_stack([np.ones(len(df)), N_z, model_ddpm])
    reg_adv = LinearRegression(fit_intercept=False).fit(X_partial, df["adv_abs"].to_numpy(dtype=float))
    reg_sb = LinearRegression(fit_intercept=False).fit(X_partial, sign_b)
    adv_resid = df["adv_abs"].to_numpy(dtype=float) - reg_adv.predict(X_partial)
    sb_resid = sign_b - reg_sb.predict(X_partial)
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in df["model"].unique():
        mask = df["model"] == m
        ax.scatter(sb_resid[mask], adv_resid[mask], label=m, alpha=0.8, s=50)
    if np.isfinite(sb_resid).all() and np.isfinite(adv_resid).all() and len(sb_resid) > 2:
        slope, intercept, r, p, _ = stats.linregress(sb_resid, adv_resid)
        xx = np.linspace(sb_resid.min(), sb_resid.max(), 100)
        ax.plot(xx, slope * xx + intercept, "k-", lw=1.5, label=f"fit (r={r:.3f})")
    ax.set_xlabel("sign_bias residual (| N_z, model)")
    ax.set_ylabel("adv_abs residual (| N_z, model)")
    ax.set_title("Med-3: Partial plot adv_abs vs sign_bias")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(outdir / "fig_med3_partial_adv_vs_sign_bias.png", dpi=200)
    plt.close(fig)
    print("已写: fig_med1, fig_med2, fig_med3")
    return summary


def run_interaction(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    """I1: adv_abs ~ sign_bias + N_z + sign_bias*N_z + model. 分档 Spearman。"""
    df = df.copy()
    df["model_ddpm"] = (df["model"] == "ddpm").astype(int)
    df["sign_bias_N_z"] = df["sign_bias"] * df["N_z"]
    y = df["adv_abs"].to_numpy(dtype=float)
    X = np.column_stack([
        np.ones(len(df)),
        df["sign_bias"].to_numpy(dtype=float),
        df["N_z"].to_numpy(dtype=float),
        df["sign_bias_N_z"].to_numpy(dtype=float),
        df["model_ddpm"].to_numpy(dtype=float),
    ])
    beta, se, p, r2, n = _ols_summary(X, y, ["intercept", "sign_bias", "N_z", "sign_bias_N_z", "model_ddpm"])
    pd.DataFrame([{
        "model": "I1",
        "beta_sign_bias_N_z": beta[3],
        "p_sign_bias_N_z": p[3],
        "R2": r2,
        "n": n,
    }]).to_csv(outdir / "interaction_I1_summary.csv", index=False, float_format="%.6f")

    # 分档：small <=500, mid=1000, large>=1500
    def _bin(n_val: float) -> str:
        if n_val <= 500:
            return "small"
        if n_val >= 1500:
            return "large"
        return "mid"

    df["N_bin"] = df["train_size"].map(_bin)
    binned = []
    for bin_name in ["small", "mid", "large"]:
        sub = df[df["N_bin"] == bin_name]
        if len(sub) < 3:
            continue
        rho, p_s = stats.spearmanr(sub["adv_abs"], sub["sign_bias"], nan_policy="omit")
        binned.append({"bin_name": bin_name, "n": len(sub), "spearman_rho": rho, "p": p_s})
    binned_df = pd.DataFrame(binned)
    binned_df.to_csv(outdir / "interaction_binned_spearman.csv", index=False, float_format="%.6f")
    print(f"已写: {outdir / 'interaction_binned_spearman.csv'}")

    # 图 Int-1: x=sign_bias, y=adv_abs, 两条线 small vs large
    fig, ax = plt.subplots(figsize=(7, 5))
    for bin_name, label in [("small", "N small (≤500)"), ("large", "N large (≥1500)")]:
        sub = df[df["N_bin"] == bin_name]
        if len(sub) < 2:
            continue
        ax.scatter(sub["sign_bias"], sub["adv_abs"], label=label, alpha=0.8, s=50)
        x = sub["sign_bias"].to_numpy(dtype=float)
        y = sub["adv_abs"].to_numpy(dtype=float)
        if len(x) > 1 and np.isfinite(x).all() and np.isfinite(y).all():
            slope, intercept, _, _, _ = stats.linregress(x, y)
            xx = np.linspace(x.min(), x.max(), 50)
            ax.plot(xx, slope * xx + intercept, lw=2, label=f"{label} fit")
    ax.set_xlabel("sign_bias")
    ax.set_ylabel("adv_abs")
    ax.set_title("Interaction: adv_abs vs sign_bias by N bin")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(outdir / "fig_interaction_signbias_by_Nbin.png", dpi=200)
    plt.close(fig)
    print("已写: fig_interaction_signbias_by_Nbin.png")
    return binned_df


def run_per_model(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    """CTGAN / DDPM 分别：Spearman(adv_abs, sign_bias/excess_signal/snr_delta) + 回归。"""
    rows = []
    for model_name in ["ctgan", "ddpm"]:
        sub = df[df["model"] == model_name]
        if len(sub) < 3:
            continue
        y = sub["adv_abs"].to_numpy(dtype=float)
        N_z = sub["N_z"].to_numpy(dtype=float)
        for metric, col in [("sign_bias", "sign_bias"), ("excess_signal", "excess_signal"), ("snr_delta", "snr_delta")]:
            if col not in sub.columns or sub[col].isna().all():
                continue
            x = sub[col].to_numpy(dtype=float)
            rho, p_s = stats.spearmanr(y, x, nan_policy="omit")
            X = np.column_stack([np.ones(len(sub)), x, N_z])
            beta, se, p, r2, n = _ols_summary(X, y, ["intercept", col, "N_z"])
            rows.append({
                "model": model_name,
                "metric": metric,
                "spearman_rho": rho,
                "spearman_p": p_s,
                "beta": beta[1] if len(beta) > 1 else np.nan,
                "p_beta": p[1] if len(p) > 1 else np.nan,
                "R2": r2,
                "n": n,
            })
    pm_df = pd.DataFrame(rows)
    pm_df.to_csv(outdir / "per_model_mechanism_summary.csv", index=False, float_format="%.6f")
    print(f"已写: {outdir / 'per_model_mechanism_summary.csv'}")

    for model_name in ["ctgan", "ddpm"]:
        sub = df[df["model"] == model_name]
        if len(sub) < 2:
            continue
        for x_col, fname in [("sign_bias", f"fig_pm_{model_name}_adv_vs_sign_bias.png"), ("excess_signal", f"fig_pm_{model_name}_adv_vs_excess_signal.png")]:
            if x_col not in sub.columns or sub[x_col].isna().all():
                continue
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(sub[x_col], sub["adv_abs"], alpha=0.8, s=50)
            x = sub[x_col].to_numpy(dtype=float)
            y = sub["adv_abs"].to_numpy(dtype=float)
            if len(x) > 1 and np.isfinite(x).all() and np.isfinite(y).all():
                slope, intercept, r, p, _ = stats.linregress(x, y)
                xx = np.linspace(x.min(), x.max(), 100)
                ax.plot(xx, slope * xx + intercept, "k-", lw=1.5, label=f"r={r:.3f}")
            ax.set_xlabel(x_col)
            ax.set_ylabel("adv_abs")
            ax.set_title(f"Per-model ({model_name}): adv_abs vs {x_col}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig.savefig(outdir / fname, dpi=200)
            plt.close(fig)
    print("已写: fig_pm_ctgan/ddpm_adv_vs_sign_bias.png, fig_pm_*_adv_vs_excess_signal.png")
    return pm_df


def run_bound(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    """excess_pos = max(excess_signal, 0), sqrt_excess = sqrt(excess_pos). Spearman + 回归。"""
    if "excess_signal" not in df.columns or df["excess_signal"].isna().all():
        bound_df = pd.DataFrame()
        bound_df.to_csv(outdir / "bound_proxy_summary.csv", index=False)
        print("无 excess_signal，跳过 bound 分析")
        return bound_df
    df = df.dropna(subset=["excess_signal"]).copy()
    df["excess_pos"] = df["excess_signal"].clip(lower=0)
    df["sqrt_excess"] = np.sqrt(df["excess_pos"])
    rows = []
    for subset_name, sub in [("all", df), ("ctgan", df[df["model"] == "ctgan"]), ("ddpm", df[df["model"] == "ddpm"])]:
        if len(sub) < 3:
            continue
        rho, p_s = stats.spearmanr(sub["adv_abs"], sub["sqrt_excess"], nan_policy="omit")
        sub = sub.copy()
        sub["model_ddpm"] = (sub["model"] == "ddpm").astype(int)
        X = np.column_stack([np.ones(len(sub)), sub["sqrt_excess"].to_numpy(dtype=float), sub["N_z"].to_numpy(dtype=float), sub["model_ddpm"].to_numpy(dtype=float)])
        y = sub["adv_abs"].to_numpy(dtype=float)
        beta, se, p, r2, n = _ols_summary(X, y, ["intercept", "sqrt_excess", "N_z", "model_ddpm"])
        rows.append({
            "subset": subset_name,
            "spearman_rho": rho,
            "spearman_p": p_s,
            "beta_sqrt_excess": beta[1] if len(beta) > 1 else np.nan,
            "p_sqrt_excess": p[1] if len(p) > 1 else np.nan,
            "R2": r2,
            "n": n,
        })
    bound_df = pd.DataFrame(rows)
    bound_df.to_csv(outdir / "bound_proxy_summary.csv", index=False, float_format="%.6f")
    print(f"已写: {outdir / 'bound_proxy_summary.csv'}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for m in df["model"].unique():
        sub = df[df["model"] == m]
        ax.scatter(sub["sqrt_excess"], sub["adv_abs"], label=m, alpha=0.8, s=50)
    x = df["sqrt_excess"].to_numpy(dtype=float)
    y = df["adv_abs"].to_numpy(dtype=float)
    if len(x) > 1 and np.isfinite(x).all() and np.isfinite(y).all():
        slope, intercept, r, p, _ = stats.linregress(x, y)
        xx = np.linspace(x.min(), x.max(), 100)
        ax.plot(xx, slope * xx + intercept, "k-", lw=1.5, label="fit all")
    ax.set_xlabel("sqrt(excess_pos)")
    ax.set_ylabel("adv_abs")
    ax.set_title("Bound: adv_abs vs sqrt(excess_signal)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(outdir / "fig_bound_adv_vs_sqrt_excess.png", dpi=200)
    plt.close(fig)
    print("已写: fig_bound_adv_vs_sqrt_excess.png")
    return bound_df


def main() -> None:
    parser = argparse.ArgumentParser(description="最终四项验证：mediation / interaction / per-model / bound")
    parser.add_argument("--robust", type=Path, default=PROJECT_ROOT / "outputs" / "figures" / "bridge_robust_stability_table.csv", help="bridge_robust_stability_table.csv")
    parser.add_argument("--control", type=Path, default=PROJECT_ROOT / "outputs" / "figures" / "bridge_with_control_excess.csv", help="bridge_with_control_excess.csv（可选）")
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "outputs" / "figures", help="输出目录")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    df = build_run_level_final(args.robust.resolve(), args.control.resolve() if args.control else None, args.outdir)
    if df.empty or len(df) < 4:
        print("run_level_final 行数不足，退出")
        return

    run_mediation(df, args.outdir)
    run_interaction(df, args.outdir)
    run_per_model(df, args.outdir)
    run_bound(df, args.outdir)

    print("\n最终交付清单:")
    print("  表: run_level_final.csv, mediation_models_summary.csv, interaction_binned_spearman.csv, per_model_mechanism_summary.csv, bound_proxy_summary.csv")
    print("  图: fig_med1_adv_vs_srun_trim95.png, fig_med2_adv_vs_sign_bias.png, fig_med3_partial_adv_vs_sign_bias.png, fig_interaction_signbias_by_Nbin.png, fig_pm_ctgan/ddpm_adv_vs_sign_bias.png, fig_pm_*_adv_vs_excess_signal.png, fig_bound_adv_vs_sqrt_excess.png")


if __name__ == "__main__":
    main()
