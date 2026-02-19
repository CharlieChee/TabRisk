#!/usr/bin/env python3
"""
对 shadow/target_{i}/ 下的 in/out synthetic 对 (synthetic_round_{k}_in/out.csv)
计算分布级别与列级别差异，以及相对于原始 target 的最近邻距离。

输出：
- pair_metrics.csv: 每行对应一个 (target_idx, round_k) 的 in/out pair；
- summary_by_target.csv: 按 target 聚合各指标在所有 rounds 上的均值/方差。

假设目录结构：
  run_dir/
    candidate.csv
    schema.json (可选，仅用于和其它脚本对齐，不直接使用)
    shadow/
      target_0/
        synthetic_round_0_in.csv
        synthetic_round_0_out.csv
        ...
      target_1/
        ...

target 记录来源优先级：
1) 若存在 shadow/target_i/target.csv 且仅一行，则直接用该行；
2) 否则，若存在 shadow/target_manifest.csv，则用其中的 candidate_row_idx 从 candidate.csv 取；
3) 否则，退化为 candidate.csv.iloc[target_idx]。
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


NumericArray = np.ndarray


@dataclass
class PairMetrics:
    target_idx: int
    round: int
    mmd: float
    mmd_numeric_only: bool
    min_dist_in: float
    min_dist_out: float
    delta_min_dist: float
    num_mean_diff_mean: float
    num_mean_diff_max: float
    num_std_diff_mean: float
    num_std_diff_max: float
    cat_tv_mean: float
    cat_tv_max: float


def _find_shadow_targets(shadow_dir: Path, max_targets: Optional[int]) -> List[int]:
    """在 shadow 目录下自动发现 target_i 子目录，并按编号排序。"""
    if not shadow_dir.exists():
        raise FileNotFoundError(f"shadow 目录不存在: {shadow_dir}")
    indices: List[int] = []
    for d in shadow_dir.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if not name.startswith("target_"):
            continue
        suffix = name[7:]
        if not suffix.isdigit():
            continue
        indices.append(int(suffix))
    indices.sort()
    if max_targets is not None:
        indices = indices[: max_targets]
    if not indices:
        raise RuntimeError(f"shadow 下未找到任何 target_i 目录: {shadow_dir}")
    return indices


def _get_round_files(target_dir: Path) -> List[Tuple[int, Path, Path]]:
    """枚举某个 target_i 下的所有 (round_k, in_path, out_path)。"""
    in_files = [p for p in target_dir.iterdir() if p.is_file() and p.name.endswith("_in.csv")]
    out_files = [p for p in target_dir.iterdir() if p.is_file() and p.name.endswith("_out.csv")]

    def _parse_round(p: Path) -> Optional[int]:
        name = p.name
        # synthetic_round_{k}_in.csv / synthetic_round_{k}_out.csv
        try:
            core = name.split("synthetic_round_", 1)[1]
            num = core.split("_", 1)[0]
            return int(num)
        except Exception:
            return None

    in_map: Dict[int, Path] = {}
    for p in in_files:
        k = _parse_round(p)
        if k is not None:
            in_map[k] = p

    pairs: List[Tuple[int, Path, Path]] = []
    for p in out_files:
        k = _parse_round(p)
        if k is None or k not in in_map:
            continue
        pairs.append((k, in_map[k], p))

    pairs.sort(key=lambda x: x[0])
    return pairs


def _load_target_row(
    run_dir: Path,
    target_idx: int,
    candidate_df: pd.DataFrame,
) -> pd.Series:
    """按约定顺序获取 target_i 对应的原始记录。"""
    # 1) shadow/target_i/target.csv
    t_dir = run_dir / "shadow" / f"target_{target_idx}"
    t_csv = t_dir / "target.csv"
    if t_csv.exists():
        df = pd.read_csv(t_csv)
        if len(df) != 1:
            raise ValueError(f"{t_csv} 期望只有一行，实际 {len(df)} 行")
        return df.iloc[0]

    # 2) shadow/target_manifest.csv
    manifest_path = run_dir / "shadow" / "target_manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        row = manifest[manifest["target_idx"] == target_idx]
        if len(row) == 0:
            raise ValueError(f"target_manifest 中找不到 target_idx={target_idx}")
        cand_idx = int(row.iloc[0]["candidate_row_idx"])
        if cand_idx < 0 or cand_idx >= len(candidate_df):
            raise ValueError(f"candidate_row_idx={cand_idx} 超出 candidate.csv 行数 {len(candidate_df)}")
        return candidate_df.iloc[cand_idx]

    # 3) 回退：candidate_df.iloc[target_idx]
    if target_idx < 0 or target_idx >= len(candidate_df):
        raise ValueError(f"target_idx={target_idx} 超出 candidate.csv 行数 {len(candidate_df)}")
    return candidate_df.iloc[target_idx]


def _split_numeric_categorical(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """根据 dtype 简单区分数值列与类别列。"""
    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    cat_cols = [c for c in df.columns if c not in num_cols]
    return num_cols, cat_cols


def _compute_mmd_rbf(
    x: NumericArray,
    y: NumericArray,
    sigma: float = 1.0,
) -> float:
    """在标准化后的特征上计算 RBF kernel MMD（无偏估计）。"""
    if x.size == 0 or y.size == 0:
        return float("nan")
    n, m = x.shape[0], y.shape[0]
    if n < 2 or m < 2:
        return float("nan")

    gamma = 1.0 / (2.0 * sigma * sigma)
    k_xx = rbf_kernel(x, x, gamma=gamma)
    k_yy = rbf_kernel(y, y, gamma=gamma)
    k_xy = rbf_kernel(x, y, gamma=gamma)

    # 去掉对角线，使用无偏估计
    np.fill_diagonal(k_xx, 0.0)
    np.fill_diagonal(k_yy, 0.0)

    mmd2 = k_xx.sum() / (n * (n - 1)) + k_yy.sum() / (m * (m - 1)) - 2.0 * k_xy.mean()
    return float(max(mmd2, 0.0) ** 0.5)


def _sample_rows(df: pd.DataFrame, max_rows: int, random_state: int = 0) -> pd.DataFrame:
    """若行数过多，随机下采样到 max_rows 行，以控制 MMD 计算复杂度。"""
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)


def _column_stats_numeric(in_df: pd.DataFrame, out_df: pd.DataFrame, num_cols: Sequence[str]) -> Tuple[float, float, float, float]:
    """数值列：各列 |mean_in-mean_out| 与 |std_in-std_out| 的均值与最大值。"""
    if not num_cols:
        return (float("nan"), float("nan"), float("nan"), float("nan"))

    mean_diffs: List[float] = []
    std_diffs: List[float] = []
    for c in num_cols:
        a = in_df[c].astype(float)
        b = out_df[c].astype(float)
        mean_diffs.append(abs(a.mean() - b.mean()))
        std_diffs.append(abs(a.std(ddof=0) - b.std(ddof=0)))

    return (
        float(np.mean(mean_diffs)),
        float(np.max(mean_diffs)),
        float(np.mean(std_diffs)),
        float(np.max(std_diffs)),
    )


def _column_stats_categorical(in_df: pd.DataFrame, out_df: pd.DataFrame, cat_cols: Sequence[str]) -> Tuple[float, float]:
    """类别列：每列经验分布的 TV 距离的均值 / 最大值。"""
    if not cat_cols:
        return (float("nan"), float("nan"))

    tvs: List[float] = []
    for c in cat_cols:
        vc_in = in_df[c].value_counts(normalize=True)
        vc_out = out_df[c].value_counts(normalize=True)
        all_vals = set(vc_in.index).union(set(vc_out.index))
        diff_sum = 0.0
        for v in all_vals:
            p_in = float(vc_in.get(v, 0.0))
            p_out = float(vc_out.get(v, 0.0))
            diff_sum += abs(p_in - p_out)
        tv = 0.5 * diff_sum
        tvs.append(tv)

    return (float(np.mean(tvs)), float(np.max(tvs)))


def _prepare_numeric_for_mmd(
    in_df: pd.DataFrame,
    out_df: pd.DataFrame,
    num_cols: Sequence[str],
    max_rows: int,
) -> Tuple[NumericArray, NumericArray]:
    """为 MMD 准备数值特征：联合标准化后抽样。"""
    if not num_cols:
        return np.empty((0, 0)), np.empty((0, 0))

    in_s = _sample_rows(in_df[num_cols], max_rows=max_rows, random_state=0)
    out_s = _sample_rows(out_df[num_cols], max_rows=max_rows, random_state=1)

    all_data = pd.concat([in_s, out_s], axis=0, ignore_index=True)
    mu = all_data.mean()
    sigma = all_data.std(ddof=0).replace(0.0, 1.0)

    x = ((in_s - mu) / sigma).to_numpy(dtype=float)
    y = ((out_s - mu) / sigma).to_numpy(dtype=float)
    return x, y


def _compute_mmd_mixed(
    in_df: pd.DataFrame,
    out_df: pd.DataFrame,
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
    max_rows: int,
) -> float:
    """
    混合型 MMD：
    - 数值部分：标准化后用 RBF kernel；
    - 类别部分：用 Hamming 相似度 kernel（1 - mismatch_ratio）；
    - 最终 kernel = 0.5 * K_num + 0.5 * K_cat（若某一类不存在则退化为另一类）。
    """
    has_num = bool(num_cols)
    has_cat = bool(cat_cols)
    if not has_num and not has_cat:
        return float("nan")

    # 下采样，避免 O(n^2) 过大
    in_s = _sample_rows(in_df, max_rows=max_rows, random_state=0)
    out_s = _sample_rows(out_df, max_rows=max_rows, random_state=1)

    n, m = len(in_s), len(out_s)
    if n < 2 or m < 2:
        return float("nan")

    # 数值部分 kernel
    K_xx_num = K_yy_num = K_xy_num = None
    if has_num:
        x_num, y_num = _prepare_numeric_for_mmd(in_s, out_s, num_cols, max_rows=max_rows)
        if x_num.size > 0 and y_num.size > 0:
            K_xx_num = rbf_kernel(x_num, x_num, gamma=1.0 / 2.0)
            K_yy_num = rbf_kernel(y_num, y_num, gamma=1.0 / 2.0)
            K_xy_num = rbf_kernel(x_num, y_num, gamma=1.0 / 2.0)

    # 类别部分 kernel：Hamming 相似度
    K_xx_cat = K_yy_cat = K_xy_cat = None
    if has_cat:
        x_cat = in_s[cat_cols].astype(str).to_numpy(dtype=object)
        y_cat = out_s[cat_cols].astype(str).to_numpy(dtype=object)
        d_cat = x_cat.shape[1]
        if d_cat > 0:
            # xx
            A = x_cat[:, None, :]  # n x 1 x d
            B = x_cat[None, :, :]  # 1 x n x d
            mismatches_xx = (A != B).mean(axis=2)  # n x n
            K_xx_cat = 1.0 - mismatches_xx
            # yy
            A = y_cat[:, None, :]
            B = y_cat[None, :, :]
            mismatches_yy = (A != B).mean(axis=2)  # m x m
            K_yy_cat = 1.0 - mismatches_yy
            # xy
            A = x_cat[:, None, :]  # n x 1 x d
            B = y_cat[None, :, :]  # 1 x m x d
            mismatches_xy = (A != B).mean(axis=2)  # n x m
            K_xy_cat = 1.0 - mismatches_xy

    # 组合 kernel，按是否存在数值/类别来加权
    if has_num and K_xx_num is not None and has_cat and K_xx_cat is not None:
        w_num = 0.5
        w_cat = 0.5
        K_xx = w_num * K_xx_num + w_cat * K_xx_cat
        K_yy = w_num * K_yy_num + w_cat * K_yy_cat
        K_xy = w_num * K_xy_num + w_cat * K_xy_cat
    elif has_num and K_xx_num is not None:
        K_xx, K_yy, K_xy = K_xx_num, K_yy_num, K_xy_num
    elif has_cat and K_xx_cat is not None:
        K_xx, K_yy, K_xy = K_xx_cat, K_yy_cat, K_xy_cat
    else:
        return float("nan")

    np.fill_diagonal(K_xx, 0.0)
    np.fill_diagonal(K_yy, 0.0)

    mmd2 = K_xx.sum() / (n * (n - 1)) + K_yy.sum() / (m * (m - 1)) - 2.0 * K_xy.mean()
    return float(max(mmd2, 0.0) ** 0.5)


def _mixed_type_min_dist(
    target_row: pd.Series,
    df: pd.DataFrame,
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
) -> float:
    """计算 target_row 到 df 中所有行的 mixed-type 最小距离。"""
    n_rows = len(df)
    if n_rows == 0:
        return float("nan")

    # 数值部分：使用 in/out 联合数据的均值/方差做标准化，这里简单用 df 内部统计
    if num_cols:
        data_num = df[num_cols].astype(float)
        mu = data_num.mean()
        sigma = data_num.std(ddof=0).replace(0.0, 1.0)
        data_num_z = ((data_num - mu) / sigma).to_numpy(dtype=float)
        t_num = target_row[num_cols].astype(float)
        t_num_z = ((t_num - mu) / sigma).to_numpy(dtype=float)
        # L2 / sqrt(d_num)
        diff = data_num_z - t_num_z.reshape(1, -1)
        d_num = diff ** 2
        d_num = np.sqrt(d_num.sum(axis=1) / float(len(num_cols)))
    else:
        d_num = np.zeros(n_rows, dtype=float)

    # 类别部分：0/1 mismatch 比例
    if cat_cols:
        data_cat = df[cat_cols].astype(str).to_numpy(dtype=object)
        t_cat = target_row[cat_cols].astype(str).to_numpy(dtype=object)
        mismatches = (data_cat != t_cat.reshape(1, -1))
        d_cat = mismatches.mean(axis=1).astype(float)
    else:
        d_cat = np.zeros(n_rows, dtype=float)

    dist = d_num + d_cat
    return float(dist.min())


def _compute_pair_metrics_for_target(
    run_dir: Path,
    target_idx: int,
    candidate_df: pd.DataFrame,
    mmd_numeric_only: bool,
    mmd_max_rows: int,
) -> List[PairMetrics]:
    """对单个 target 计算所有 rounds 的 pair metrics。"""
    target_dir = run_dir / "shadow" / f"target_{target_idx}"
    if not target_dir.exists():
        raise FileNotFoundError(f"target 目录不存在: {target_dir}")

    pairs = _get_round_files(target_dir)
    if not pairs:
        raise RuntimeError(f"{target_dir} 下未找到任何 synthetic_round_k_in/out.csv 对")

    target_row = _load_target_row(run_dir, target_idx, candidate_df)

    results: List[PairMetrics] = []
    for k, in_path, out_path in pairs:
        in_df = pd.read_csv(in_path)
        out_df = pd.read_csv(out_path)

        # 对齐列顺序（防御性）
        out_df = out_df[in_df.columns]

        num_cols, cat_cols = _split_numeric_categorical(in_df)

        # 1) MMD：根据 mmd_numeric_only 决定是否加入类别列
        if mmd_numeric_only or not cat_cols:
            if num_cols:
                x_num, y_num = _prepare_numeric_for_mmd(in_df, out_df, num_cols, max_rows=mmd_max_rows)
                mmd_val = _compute_mmd_rbf(x_num, y_num, sigma=1.0)
            else:
                mmd_val = float("nan")
            mmd_numeric_only_flag = True
        else:
            mmd_val = _compute_mmd_mixed(in_df, out_df, num_cols, cat_cols, max_rows=mmd_max_rows)
            mmd_numeric_only_flag = False

        # 2) 列级差异统计
        num_mean_diff_mean, num_mean_diff_max, num_std_diff_mean, num_std_diff_max = _column_stats_numeric(
            in_df, out_df, num_cols
        )
        cat_tv_mean, cat_tv_max = _column_stats_categorical(in_df, out_df, cat_cols)

        # 3) target 最近邻距离
        min_dist_in = _mixed_type_min_dist(target_row, in_df, num_cols, cat_cols)
        min_dist_out = _mixed_type_min_dist(target_row, out_df, num_cols, cat_cols)
        delta_min = min_dist_out - min_dist_in if not (math.isnan(min_dist_in) or math.isnan(min_dist_out)) else float(
            "nan"
        )

        results.append(
            PairMetrics(
                target_idx=target_idx,
                round=k,
                mmd=mmd_val,
                mmd_numeric_only=bool(mmd_numeric_only_flag),
                min_dist_in=min_dist_in,
                min_dist_out=min_dist_out,
                delta_min_dist=delta_min,
                num_mean_diff_mean=num_mean_diff_mean,
                num_mean_diff_max=num_mean_diff_max,
                num_std_diff_mean=num_std_diff_mean,
                num_std_diff_max=num_std_diff_max,
                cat_tv_mean=cat_tv_mean,
                cat_tv_max=cat_tv_max,
            )
        )

    return results


def _worker_compute_pair_metrics(
    args: Tuple[Path, int, pd.DataFrame, bool, int],
) -> List[PairMetrics]:
    """multiprocessing.Pool 用的顶层 worker，避免本地函数不可 pickle 问题。"""
    run_dir_i, ti_i, cand_df_i, mmd_num_only_i, mmd_max_rows_i = args
    return _compute_pair_metrics_for_target(
        run_dir=run_dir_i,
        target_idx=ti_i,
        candidate_df=cand_df_i,
        mmd_numeric_only=mmd_num_only_i,
        mmd_max_rows=mmd_max_rows_i,
    )


def _pair_metrics_to_dataframe(rows: Sequence[PairMetrics]) -> pd.DataFrame:
    """将 PairMetrics 列表转换为 DataFrame。"""
    data = [
        {
            "target_idx": r.target_idx,
            "round": r.round,
            "mmd": r.mmd,
            "mmd_numeric_only": r.mmd_numeric_only,
            "min_dist_in": r.min_dist_in,
            "min_dist_out": r.min_dist_out,
            "delta_min_dist": r.delta_min_dist,
            "num_mean_diff_mean": r.num_mean_diff_mean,
            "num_mean_diff_max": r.num_mean_diff_max,
            "num_std_diff_mean": r.num_std_diff_mean,
            "num_std_diff_max": r.num_std_diff_max,
            "cat_tv_mean": r.cat_tv_mean,
            "cat_tv_max": r.cat_tv_max,
        }
        for r in rows
    ]
    return pd.DataFrame(data)


def _summarize_by_target(pair_df: pd.DataFrame) -> pd.DataFrame:
    """按 target 聚合所有 rounds 的均值与方差。"""
    if pair_df.empty:
        return pair_df

    numeric_cols = pair_df.select_dtypes(include=[np.number, bool]).columns.tolist()
    # 不对 round 本身求 summary
    numeric_cols = [c for c in numeric_cols if c not in ("target_idx", "round")]

    grouped = pair_df.groupby("target_idx", as_index=False)
    agg_dict = {c: ["mean", "var"] for c in numeric_cols}
    summary = grouped.agg(agg_dict)

    # 展平成单层列名，如 mmd_mean, mmd_var
    summary.columns = [
        col if isinstance(col, str) else f"{col[0]}_{col[1]}"
        for col in summary.columns.to_flat_index()
    ]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="对 shadow target_i 的 in/out synthetic 计算 MMD、列级差异与 target 最近邻距离。",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="单次训练的 run 目录（含 shadow/target_i/、candidate.csv 等）",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=None,
        help="仅统计前 N 个 target（按 target_i 编号排序），默认全量",
    )
    parser.add_argument(
        "--mmd-max-rows",
        type=int,
        default=500,
        help="计算 MMD 时每侧最多使用的样本数，用于控制复杂度（默认 500）",
    )
    parser.add_argument(
        "--mmd-numeric-only",
        action="store_true",
        help="仅基于数值列计算 MMD，并在输出列 mmd_numeric_only 中标记（默认行为）",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="并行进程数（按 target 粒度并行，默认 1=不并行）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="结果保存目录（默认写入 run_dir/shadow_pair_metrics）",
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    if not run_dir.exists():
        parser.error(f"run_dir 不存在: {run_dir}")

    candidate_path = run_dir / "candidate.csv"
    if not candidate_path.exists():
        parser.error(f"candidate.csv 不存在: {candidate_path}")
    candidate_df = pd.read_csv(candidate_path)

    shadow_dir = run_dir / "shadow"
    target_indices = _find_shadow_targets(shadow_dir, args.max_targets)

    # 按 target 维度并行：每个进程处理若干 target_i
    all_rows: List[PairMetrics] = []
    n_jobs = max(1, int(args.n_jobs))
    if n_jobs == 1 or len(target_indices) == 1:
        for ti in target_indices:
            res = _compute_pair_metrics_for_target(
                run_dir=run_dir,
                target_idx=ti,
                candidate_df=candidate_df,
                mmd_numeric_only=bool(args.mmd_numeric_only),
                mmd_max_rows=int(args.mmd_max_rows),
            )
            all_rows.extend(res)
    else:
        from multiprocessing import Pool

        worker_args = [
            (run_dir, ti, candidate_df, bool(args.mmd_numeric_only), int(args.mmd_max_rows))
            for ti in target_indices
        ]

        with Pool(processes=n_jobs) as pool:
            for res in pool.map(_worker_compute_pair_metrics, worker_args):
                all_rows.extend(res)

    pair_df = _pair_metrics_to_dataframe(all_rows)

    if args.output_dir:
        out_dir = Path(args.output_dir)
        if not out_dir.is_absolute():
            out_dir = PROJECT_ROOT / out_dir
    else:
        out_dir = run_dir / "shadow_pair_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_path = out_dir / "pair_metrics.csv"
    pair_df.to_csv(pair_path, index=False)

    summary_df = _summarize_by_target(pair_df)
    summary_path = out_dir / "summary_by_target.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"pair_metrics.csv 已写入: {pair_path}")
    print(f"summary_by_target.csv 已写入: {summary_path}")


if __name__ == "__main__":
    main()

