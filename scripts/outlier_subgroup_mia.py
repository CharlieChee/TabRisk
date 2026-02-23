#!/usr/bin/env python3
"""
Attack-agnostic outlier 子集上的 single-release MIA 评估。

原则：outlier 定义只用训练/候选数据分布或单份 synthetic 信息，不用 in/out、control、attack score。
- Outlier 1: Train-density (kNN in candidate 特征空间)
- Outlier 2: Rare-category (categorical -log freq)
- 子集: top p% (p ∈ {5,10,20,50,100}，适配 candidate≈100)
- 评估: single-release naive (target-level -median(min_dist_in)) 在子集上的 AUC
- 输出: 表 + 图 + 反后验自检 (outlier vs membership leak, 跨 seed 稳定性)
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IS_MEMBER_COL = "is_member"
OUTLIER_DEFS = ["density", "rare"]
P_PERCENTS = [5, 10, 20, 50, 100]
KNN_K = 5


def parse_run_dir(dirname: str) -> Dict[str, Any]:
    out = {"train_size": None, "rounds": None, "seed": None, "model": None}
    m = re.search(r"_train(\d+)_synth", dirname)
    if m:
        out["train_size"] = int(m.group(1))
    m = re.search(r"_rounds(\d+)_", f"_{dirname}_")
    if m:
        out["rounds"] = int(m.group(1))
    m = re.search(r"_seed(\d+)(?:_|$)", dirname)
    if m:
        out["seed"] = int(m.group(1))
    if "ddpm" in dirname.lower() or "tabddpm" in dirname.lower():
        out["model"] = "ddpm"
    elif "ctgan" in dirname.lower():
        out["model"] = "ctgan"
    return out


def scan_and_filter_runs(outputs_dir: Path) -> List[Path]:
    run_dirs = []
    for d in outputs_dir.iterdir():
        if not d.is_dir() or "control" not in d.name:
            continue
        if not (d / "candidate.csv").exists():
            continue
        m = d / "shadow_pair_metrics"
        if not m.is_dir() or not (m / "pair_metrics.csv").exists() or not (m / "pair_metrics_control.csv").exists():
            continue
        run_dirs.append(d)
    for sub in outputs_dir.iterdir():
        if not sub.is_dir() or sub.name.startswith("."):
            continue
        for d in sub.iterdir():
            if not d.is_dir() or "control" not in d.name:
                continue
            if not (d / "candidate.csv").exists():
                continue
            m = d / "shadow_pair_metrics"
            if not m.is_dir() or not (m / "pair_metrics.csv").exists() or not (m / "pair_metrics_control.csv").exists():
                continue
            run_dirs.append(d)
    run_dirs = sorted(set(run_dirs), key=lambda p: (p.name, str(p)))
    rows = []
    for d in run_dirs:
        p = parse_run_dir(d.name)
        if p.get("rounds") != 20 or not p.get("model"):
            continue
        rows.append({"run_dir": str(d.resolve()), "model": p["model"], "train_size": p.get("train_size"), "seed": p.get("seed")})
    if not rows:
        return []
    df = pd.DataFrame(rows)
    cnt = df.groupby(["model", "train_size"])["seed"].nunique().reset_index()
    cnt.columns = ["model", "train_size", "n_seeds"]
    valid = cnt[cnt["n_seeds"] >= 5][["model", "train_size"]]
    if valid.empty:
        return []
    df = df.merge(valid, on=["model", "train_size"], how="inner")
    df = df.drop_duplicates(subset=["model", "train_size", "seed"], keep="last")
    return [Path(p) for p in df["run_dir"].tolist()]


# ---------- Part 1: Outlier definitions (attack-agnostic) ----------
def _get_numeric_cat_cols(candidate: pd.DataFrame) -> Tuple[List[str], List[str]]:
    exclude = {IS_MEMBER_COL}
    num_cols = [c for c in candidate.select_dtypes(include=[np.number]).columns if c not in exclude]
    cat_cols = [c for c in candidate.columns if c not in exclude and c not in num_cols and candidate[c].dtype in ("object", "category", "string")]
    return num_cols, cat_cols


def outlier_score_density(candidate: pd.DataFrame, k: int = KNN_K) -> np.ndarray:
    """kNN 平均距离（越大越 outlier）。只用 candidate 特征，不含 is_member。"""
    num_cols, cat_cols = _get_numeric_cat_cols(candidate)
    if not num_cols and not cat_cols:
        return np.zeros(len(candidate)) + np.nan
    parts = []
    if num_cols:
        Xn = candidate[num_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()
        scaler = StandardScaler()
        Xn = scaler.fit_transform(Xn)
        parts.append(Xn)
    if cat_cols:
        Xc = np.zeros((len(candidate), len(cat_cols)))
        for j, col in enumerate(cat_cols):
            le = LabelEncoder()
            Xc[:, j] = le.fit_transform(candidate[col].astype(str).fillna("__NA__"))
        parts.append(Xc)
    X = np.hstack(parts)
    n = X.shape[0]
    k_use = min(k, n - 1)
    if k_use < 1:
        return np.zeros(n) + np.nan
    nn = NearestNeighbors(n_neighbors=k_use + 1, metric="euclidean").fit(X)
    dist, _ = nn.kneighbors(X)
    mean_dist = np.mean(dist[:, 1:], axis=1)
    return mean_dist


def outlier_score_rare(candidate: pd.DataFrame) -> np.ndarray:
    """Rarity: 对每条记录，categorical 列取值频率的 -log(freq) 之和（越大越稀有）。"""
    _, cat_cols = _get_numeric_cat_cols(candidate)
    if not cat_cols:
        return np.full(len(candidate), np.nan)
    scores = np.zeros(len(candidate))
    for col in cat_cols:
        counts = candidate[col].astype(str).fillna("__NA__").value_counts()
        n = len(candidate)
        freq = counts / n
        log_freq = np.log(np.clip(freq.values, 1e-10, 1))
        rarity = -log_freq
        mapping = dict(zip(counts.index, rarity))
        scores += candidate[col].astype(str).fillna("__NA__").map(lambda x: mapping.get(x, 0)).to_numpy()
    return scores


def compute_outlier_scores(candidate: pd.DataFrame) -> Dict[str, np.ndarray]:
    return {
        "density": outlier_score_density(candidate),
        "rare": outlier_score_rare(candidate),
    }


# ---------- Part 2: Target-level single-release score & subgroup AUC ----------
def load_manifest(run_dir: Path) -> Optional[pd.DataFrame]:
    p = run_dir / "shadow" / "target_manifest.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def target_level_scores(pair_path: Path, pair_control_path: Path) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    pair_metrics: target_idx -> score_naive = -median(min_dist_in);
    pair_metrics_control: target_idx -> score_naive_control = -median(min_dist_in).
    """
    if not pair_path.exists() or "min_dist_in" not in pd.read_csv(pair_path, nrows=1).columns:
        return {}, {}
    df_m = pd.read_csv(pair_path)
    df_c = pd.read_csv(pair_control_path)
    if "target_idx" not in df_m.columns or "min_dist_in" not in df_m.columns:
        return {}, {}
    member_scores = df_m.groupby("target_idx")["min_dist_in"].apply(lambda x: -np.median(x.dropna())).to_dict()
    if "target_idx" in df_c.columns and "min_dist_in" in df_c.columns:
        control_scores = df_c.groupby("target_idx")["min_dist_in"].apply(lambda x: -np.median(x.dropna())).to_dict()
    else:
        control_scores = {}
    return member_scores, control_scores


def top_p_indices(scores: np.ndarray, p_percent: float) -> np.ndarray:
    """Top p% 的 candidate 行索引（分数越大越 outlier，取最大的 p%）。"""
    n = len(scores)
    valid = np.isfinite(scores)
    if not np.any(valid):
        return np.array([], dtype=int)
    n_valid = np.sum(valid)
    k = max(1, int(np.ceil(n_valid * p_percent / 100)))
    order = np.argsort(-np.where(valid, scores, -np.inf))
    return order[:k]


def subgroup_auc(
    manifest: pd.DataFrame,
    member_scores: Dict[int, float],
    control_scores: Dict[int, float],
    candidate_row_indices: Set[int],
) -> Tuple[float, int, int]:
    """只对 candidate_row_idx 在 candidate_row_indices 中的 target 算 AUC。返回 (auc, n_member, n_control)。"""
    if "candidate_row_idx" not in manifest.columns:
        return float("nan"), 0, 0
    in_subset = manifest["candidate_row_idx"].isin(candidate_row_indices)
    sub = manifest[in_subset]
    scores_list = []
    labels_list = []
    for _, row in sub.iterrows():
        tid = int(row["target_idx"])
        is_mem = int(row.get("is_member", 0))
        if is_mem == 1 and tid in member_scores:
            scores_list.append(member_scores[tid])
            labels_list.append(1)
        elif is_mem == 0 and tid in control_scores:
            scores_list.append(control_scores[tid])
            labels_list.append(0)
    scores = np.array(scores_list)
    labels = np.array(labels_list)
    n_m, n_c = int(np.sum(labels == 1)), int(np.sum(labels == 0))
    if len(scores) < 2 or n_m == 0 or n_c == 0:
        return float("nan"), n_m, n_c
    auc = roc_auc_score(labels, scores)
    auc = max(float(auc), 1.0 - float(auc))
    return auc, n_m, n_c


def process_one_run(run_dir: Path) -> Optional[Dict[str, Any]]:
    candidate_path = run_dir / "candidate.csv"
    if not candidate_path.exists():
        return None
    candidate = pd.read_csv(candidate_path)
    if IS_MEMBER_COL not in candidate.columns:
        return None
    manifest = load_manifest(run_dir)
    if manifest is None:
        manifest = pd.DataFrame({"target_idx": range(len(candidate)), "candidate_row_idx": range(len(candidate)), "is_member": candidate[IS_MEMBER_COL].tolist()})
    metrics_dir = run_dir / "shadow_pair_metrics"
    pair_path = metrics_dir / "pair_metrics.csv"
    pair_control_path = metrics_dir / "pair_metrics_control.csv"
    if not pair_path.exists() or not pair_control_path.exists():
        return None
    member_scores, control_scores = target_level_scores(pair_path, pair_control_path)
    if not member_scores or not control_scores:
        return None
    outlier_scores = compute_outlier_scores(candidate)
    parsed = parse_run_dir(run_dir.name)
    n_cand = len(candidate)

    rows_curves = []
    for def_name, o_scores in outlier_scores.items():
        for p in P_PERCENTS:
            top_idx = top_p_indices(o_scores, p)
            cand_set = set(int(i) for i in top_idx)
            auc_s, n_m, n_c = subgroup_auc(manifest, member_scores, control_scores, cand_set)
            n_t = n_m + n_c
            rows_curves.append({
                "run_dir": run_dir.name,
                "model": parsed.get("model"),
                "train_size": parsed.get("train_size"),
                "seed": parsed.get("seed"),
                "outlier_def": def_name,
                "p": p,
                "n_targets": n_t,
                "n_member": n_m,
                "n_control": n_c,
                "auc_naive_subgroup": auc_s,
            })

    full_auc, _, _ = subgroup_auc(manifest, member_scores, control_scores, set(range(n_cand)))
    score_stats = {}
    for def_name, o_scores in outlier_scores.items():
        v = o_scores[np.isfinite(o_scores)]
        score_stats[def_name] = {"mean": np.mean(v) if len(v) else np.nan, "std": np.std(v) if len(v) > 1 else np.nan,
                                 "p90": np.percentile(v, 90) if len(v) else np.nan,
                                 "p95": np.percentile(v, 95) if len(v) else np.nan,
                                 "p99": np.percentile(v, 99) if len(v) else np.nan}

    return {
        "run_dir": run_dir.name,
        "run_dir_path": str(run_dir.resolve()),
        "model": parsed.get("model"),
        "train_size": parsed.get("train_size"),
        "seed": parsed.get("seed"),
        "auc_full": full_auc,
        "curves": rows_curves,
        "outlier_scores": outlier_scores,
        "score_stats": score_stats,
        "candidate": candidate,
        "manifest": manifest,
    }


def run_all(outputs_dir: Path, n_jobs: int) -> List[Dict[str, Any]]:
    run_dirs = scan_and_filter_runs(outputs_dir)
    results = []
    for d in run_dirs:
        r = process_one_run(d)
        if r is not None:
            results.append(r)
    return results


# ---------- Part 5: Sanity checks ----------
def sanity_membership_leak(outlier_scores: np.ndarray, is_member: np.ndarray) -> float:
    """AUC(outlier_score 预测 is_member)。期望 ~0.5。"""
    if np.unique(is_member).size < 2 or not np.any(np.isfinite(outlier_scores)):
        return float("nan")
    valid = np.isfinite(outlier_scores)
    return float(roc_auc_score(is_member[valid], outlier_scores[valid]))


def leak_check_debug_row(
    run_dir: str,
    run_dir_path: str,
    model: Any,
    train_size: Any,
    seed: Any,
    outlier_def: str,
    o_scores: np.ndarray,
    is_member: np.ndarray,
) -> Dict[str, Any]:
    """单 run × outlier_def 的 debug 行：n_candidate, mean_is_member, score 统计, n_unique, spearman 等。"""
    n_candidate = len(is_member)
    mean_is_member = float(np.mean(is_member))
    valid = np.isfinite(o_scores)
    s = o_scores[valid]
    if len(s) == 0:
        return {
            "run_dir": run_dir, "model": model, "train_size": train_size, "seed": seed, "outlier_def": outlier_def,
            "n_candidate": n_candidate, "mean_is_member": mean_is_member,
            "score_mean": np.nan, "score_std": np.nan, "score_min": np.nan, "score_max": np.nan,
            "score_p01": np.nan, "score_p50": np.nan, "score_p99": np.nan,
            "n_unique_scores": 0, "unique_ratio": np.nan,
            "spearman_r": np.nan, "spearman_p": np.nan, "auc_membership_leak": np.nan,
        }
    score_mean = float(np.mean(s))
    score_std = float(np.std(s)) if len(s) > 1 else 0.0
    score_min = float(np.min(s))
    score_max = float(np.max(s))
    score_p01 = float(np.percentile(s, 1))
    score_p50 = float(np.percentile(s, 50))
    score_p99 = float(np.percentile(s, 99))
    n_unique = int(len(np.unique(s)))
    unique_ratio = n_unique / n_candidate if n_candidate else np.nan
    auc_leak = sanity_membership_leak(o_scores, is_member)
    y = is_member[valid]
    if len(np.unique(y)) >= 2 and len(s) >= 2:
        sp = scipy_stats.spearmanr(s, y)
        spearman_r = float(sp.statistic)
        spearman_p = float(sp.pvalue)
    else:
        spearman_r = np.nan
        spearman_p = np.nan
    return {
        "run_dir": run_dir, "model": model, "train_size": train_size, "seed": seed, "outlier_def": outlier_def,
        "n_candidate": n_candidate, "mean_is_member": mean_is_member,
        "score_mean": score_mean, "score_std": score_std, "score_min": score_min, "score_max": score_max,
        "score_p01": score_p01, "score_p50": score_p50, "score_p99": score_p99,
        "n_unique_scores": n_unique, "unique_ratio": unique_ratio,
        "spearman_r": spearman_r, "spearman_p": spearman_p, "auc_membership_leak": auc_leak,
    }


def candidate_file_fingerprint(run_dir_path: str) -> Tuple[int, str]:
    """返回 (文件大小, 前 4096 字节的 SHA256 前 16 位指纹)。"""
    p = Path(run_dir_path) / "candidate.csv"
    if not p.exists():
        return -1, ""
    size = p.stat().st_size
    with open(p, "rb") as f:
        head = f.read(4096)
    h = hashlib.sha256(head).hexdigest()[:16]
    return size, h


def main() -> int:
    parser = argparse.ArgumentParser(description="Attack-agnostic outlier 子集上 single-release MIA 评估")
    parser.add_argument("--outputs", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "outputs" / "outlier_mia")
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    results = run_all(args.outputs, args.n_jobs)
    if not results:
        print("No runs processed.")
        return 1

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Flatten curves
    all_curves = []
    for r in results:
        all_curves.extend(r["curves"])
    curves_df = pd.DataFrame(all_curves)
    curves_df.to_csv(args.outdir / "outlier_auc_curves.csv", index=False)
    print(f"Wrote {args.outdir / 'outlier_auc_curves.csv'}")

    # Summary by (model, train_size, outlier_def, p)
    summary = curves_df.groupby(["model", "train_size", "outlier_def", "p"], as_index=False).agg(
        auc_mean=("auc_naive_subgroup", "mean"),
        auc_std=("auc_naive_subgroup", "std"),
        n_targets_mean=("n_targets", "mean"),
    )
    summary.to_csv(args.outdir / "outlier_auc_summary_byN.csv", index=False)
    print(f"Wrote {args.outdir / 'outlier_auc_summary_byN.csv'}")

    # Score stats
    score_rows = []
    for r in results:
        for def_name, stats in r["score_stats"].items():
            score_rows.append({
                "run_dir": r["run_dir"], "model": r["model"], "train_size": r["train_size"], "seed": r["seed"],
                "outlier_def": def_name, **stats,
            })
    pd.DataFrame(score_rows).to_csv(args.outdir / "outlier_score_stats.csv", index=False)

    # Sanity: membership leak + debug 表 + candidate 文件指纹 + 同 train_size 一致性检查
    leak_rows = []
    leak_debug_rows = []
    seen_run_paths: Set[str] = set()
    for r in results:
        cand = r["candidate"]
        if IS_MEMBER_COL not in cand.columns:
            continue
        y = cand[IS_MEMBER_COL].to_numpy()
        run_dir_path = r.get("run_dir_path", "")
        if run_dir_path and run_dir_path not in seen_run_paths:
            seen_run_paths.add(run_dir_path)
            size, fp = candidate_file_fingerprint(run_dir_path)
            print(f"[candidate.csv] run_dir={r['run_dir']} size={size} fingerprint={fp}")
        for def_name, o_scores in r["outlier_scores"].items():
            auc_leak = sanity_membership_leak(o_scores, y)
            leak_rows.append({"run_dir": r["run_dir"], "model": r["model"], "train_size": r["train_size"], "outlier_def": def_name, "auc_membership_leak": auc_leak})
            leak_debug_rows.append(leak_check_debug_row(
                r["run_dir"], run_dir_path, r["model"], r["train_size"], r["seed"], def_name, o_scores, y
            ))
    leak_df = pd.DataFrame(leak_rows)
    leak_df.to_csv(args.outdir / "outlier_score_membership_leak_check.csv", index=False)
    print(f"Membership leak check: {leak_df['auc_membership_leak'].mean():.4f} (expect ~0.5)")
    debug_df = pd.DataFrame(leak_debug_rows)
    debug_df.to_csv(args.outdir / "outlier_score_membership_leak_check_debug.csv", index=False)
    print(f"Wrote {args.outdir / 'outlier_score_membership_leak_check_debug.csv'}")

    # 同一 train_size 下多 seed 的 score_mean/score_std/n_unique_scores 是否完全一致
    for key, grp in debug_df.groupby(["model", "train_size", "outlier_def"], dropna=False):
        model, train_size, outlier_def = key
        sm = grp["score_mean"].dropna()
        ss = grp["score_std"].dropna()
        nu = grp["n_unique_scores"]
        if len(grp) >= 2:
            same_mean = sm.nunique() <= 1
            same_std = ss.nunique() <= 1
            same_nu = nu.nunique() <= 1
            if same_mean and same_std and same_nu:
                print(f"[WARNING] 同一 train_size 下各 seed 的 outlier 统计完全一致: model={model} train_size={train_size} outlier_def={outlier_def} "
                      f"score_mean={sm.iloc[0]} score_std={ss.iloc[0]} n_unique_scores={int(nu.iloc[0])} (疑似 candidate 未变或实现有误)")
                for _, row in grp.iterrows():
                    print(f"  seed={row['seed']} run_dir={row['run_dir']} score_mean={row['score_mean']} score_std={row['score_std']} n_unique_scores={row['n_unique_scores']}")

    # Fig 1: Subset size vs AUC (facet by train_size, line by outlier_def)
    for model in curves_df["model"].dropna().unique():
        sub = curves_df[curves_df["model"] == model]
        if sub.empty:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharey=True)
        axes = axes.flatten()
        train_sizes = sorted(sub["train_size"].dropna().unique())
        for i, ts in enumerate(train_sizes):
            if i >= len(axes):
                break
            ax = axes[i]
            b = sub[sub["train_size"] == ts]
            for def_name in OUTLIER_DEFS:
                bd = b[b["outlier_def"] == def_name]
                agg = bd.groupby("p", as_index=False)["auc_naive_subgroup"].agg(["mean", "std"])
                agg = agg.reset_index()
                ax.errorbar(agg["p"], agg["mean"], yerr=agg["std"].fillna(0), marker="o", label=def_name, capsize=3)
            ax.axhline(0.5, color="gray", linestyle="--")
            ax.set_xlabel("Top p%")
            ax.set_ylabel("AUC (naive subgroup)")
            ax.set_title(f"N={ts}")
            ax.legend()
            ax.grid(True, alpha=0.3)
        for j in range(len(train_sizes), len(axes)):
            axes[j].set_visible(False)
        fig.suptitle(f"Subset size vs AUC — {model.upper()}")
        fig.tight_layout()
        fig.savefig(args.outdir / f"fig_outlier_auc_vs_subsetsize_{model}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved fig_outlier_auc_vs_subsetsize_*.png")

    # Fig 2: Fixed p, AUC vs N
    fig, ax = plt.subplots(figsize=(8, 5))
    for def_name in OUTLIER_DEFS:
        for p in [5, 10]:
            s = summary[(summary["outlier_def"] == def_name) & (summary["p"] == p)]
            if s.empty:
                continue
            by_n = s.groupby(["model", "train_size"], as_index=False)["auc_mean"].mean()
            for model in by_n["model"].unique():
                m = by_n[by_n["model"] == model].sort_values("train_size")
                ax.plot(m["train_size"], m["auc_mean"], marker="o", label=f"{def_name} p={p}% ({model})")
    full_line = curves_df[curves_df["p"] == 100].groupby(["model", "train_size"])["auc_naive_subgroup"].mean().reset_index()
    for model in full_line["model"].unique():
        m = full_line[full_line["model"] == model].sort_values("train_size")
        ax.plot(m["train_size"], m["auc_naive_subgroup"], "k--", alpha=0.7, label=f"Full (p=100%) {model}")
    ax.axhline(0.5, color="gray", linestyle=":")
    ax.set_xlabel("Train size (N)")
    ax.set_ylabel("AUC (naive subgroup)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.outdir / "fig_outlier_auc_vs_N_fixedp.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_outlier_auc_vs_N_fixedp.png")

    # Fig 3: Outlier score distribution by N
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for idx, def_name in enumerate(OUTLIER_DEFS):
        ax = axes[idx]
        for r in results[:20]:
            s = r["outlier_scores"][def_name]
            s = s[np.isfinite(s)]
            if len(s):
                ax.hist(s, bins=20, alpha=0.3, density=True, label=f"N={r['train_size']}")
        ax.set_title(f"Outlier score: {def_name}")
        ax.set_xlabel("Score")
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(args.outdir / "fig_outlier_score_distribution_byN.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_outlier_score_distribution_byN.png")

    # Fig: membership leak (sanity)
    if not leak_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        for def_name in leak_df["outlier_def"].unique():
            sub = leak_df[leak_df["outlier_def"] == def_name]
            ax.scatter(sub["train_size"], sub["auc_membership_leak"], label=def_name, alpha=0.7)
        ax.axhline(0.5, color="gray", linestyle="--", label="Expected ~0.5")
        ax.set_xlabel("Train size (N)")
        ax.set_ylabel("AUC(outlier_score → is_member)")
        ax.set_title("Sanity: outlier def should not leak membership")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(args.outdir / "fig_outlier_score_membership_auc.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Fig 4: Min p such that AUC(p) >= 0.65
    curves_df["p_float"] = curves_df["p"].astype(float)
    min_p_rows = []
    for (run_dir, model, train_size, seed, outlier_def), g in curves_df.groupby(["run_dir", "model", "train_size", "seed", "outlier_def"]):
        g = g.sort_values("p")
        above = g[g["auc_naive_subgroup"] >= 0.65]
        p_min = above["p"].min() if len(above) else np.nan
        min_p_rows.append({"run_dir": run_dir, "model": model, "train_size": train_size, "seed": seed, "outlier_def": outlier_def, "p_min_65": p_min})
    min_p_df = pd.DataFrame(min_p_rows)
    if not min_p_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        for model in min_p_df["model"].unique():
            for def_name in min_p_df["outlier_def"].unique():
                sub = min_p_df[(min_p_df["model"] == model) & (min_p_df["outlier_def"] == def_name)]
                agg = sub.groupby("train_size")["p_min_65"].mean().reset_index()
                ax.plot(agg["train_size"], agg["p_min_65"], marker="o", label=f"{model} {def_name}")
        ax.set_xlabel("Train size (N)")
        ax.set_ylabel("Min p% s.t. AUC(subgroup) ≥ 0.65")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(args.outdir / "fig_min_subset_for_auc_threshold.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        min_p_df.to_csv(args.outdir / "outlier_min_p_for_auc65.csv", index=False)

    # Stability across seeds: per (model, train_size, outlier_def) score stats over seeds
    stab = pd.DataFrame(score_rows)
    if not stab.empty:
        stab_agg = stab.groupby(["model", "train_size", "outlier_def"], as_index=False).agg(
            mean_mean=("mean", "mean"), std_mean=("mean", "std"),
            mean_p95=("p95", "mean"), std_p95=("p95", "std"),
        )
        stab_agg.to_csv(args.outdir / "outlier_stability_across_seeds.csv", index=False)
        print("Wrote outlier_stability_across_seeds.csv")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
