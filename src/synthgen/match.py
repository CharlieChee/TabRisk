"""
在合成数据中查找与给定 target 行「近似相等」的条数。

支持多种近似匹配方法，可配置精度（容差、比例等）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from synthgen.data.schema import Schema, infer_schema


# 与 shadow 一致：比较时排除的列
SHADOW_MEMBER_COL = "is_member"


def _feature_columns(df: pd.DataFrame, schema: Optional[Schema] = None) -> Tuple[List[str], List[str], List[str]]:
    """返回用于匹配的特征列名（排除 is_member 等），以及分类型/连续型列表。"""
    cols = [c for c in df.columns if c != SHADOW_MEMBER_COL]
    if schema is None:
        schema = infer_schema(df[cols])
    cat = [c for c in schema.categorical_columns if c in cols]
    cont = [c for c in schema.continuous_columns if c in cols]
    # 未在 schema 中的列按数据类型推断
    for c in cols:
        if c not in cat and c not in cont:
            if pd.api.types.is_numeric_dtype(df[c].dtype):
                cont.append(c)
            else:
                cat.append(c)
    return cols, cat, cont


def _align_and_dtype(synthetic: pd.DataFrame, target: pd.Series, cols: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    """取共有的 cols，并统一 dtypes（便于比较）。"""
    common = [c for c in cols if c in synthetic.columns and c in target.index]
    S = synthetic[common].copy()
    t = target[common].copy()
    for c in common:
        if S[c].dtype != t[c].dtype:
            if pd.api.types.is_numeric_dtype(S[c]) and pd.api.types.is_numeric_dtype(t[c]):
                S[c] = pd.to_numeric(S[c], errors="coerce")
                t[c] = pd.to_numeric(t[c], errors="coerce")
            else:
                S[c] = S[c].astype(str)
                t[c] = str(t[c])
    return S, t


def method_epsilon(
    synthetic: pd.DataFrame,
    target: pd.Series,
    categorical_columns: List[str],
    continuous_columns: List[str],
    epsilon: float = 1e-6,
    rel_epsilon: Optional[float] = None,
) -> np.ndarray:
    """
    方法一：分类型精确匹配；连续型在绝对值 epsilon 内（或相对误差 rel_epsilon）。
    返回 synthetic 中每行是否近似等于 target 的布尔掩码。
    """
    cols = categorical_columns + continuous_columns
    S, t = _align_and_dtype(synthetic, target, cols)
    n = len(S)
    mask = np.ones(n, dtype=bool)

    for c in categorical_columns:
        if c not in S.columns:
            continue
        # 统一成字符串比较，NaN 视为不等
        s_vals = S[c].astype(str).values
        t_val = str(t[c]) if pd.notna(t[c]) else "nan"
        mask &= (s_vals == t_val)

    for c in continuous_columns:
        if c not in S.columns:
            continue
        s_vals = pd.to_numeric(S[c], errors="coerce").values
        t_val = pd.to_numeric(t[c], errors="coerce")
        if pd.isna(t_val):
            mask &= np.isnan(s_vals)
            continue
        if rel_epsilon is not None and rel_epsilon > 0 and abs(t_val) > 1e-12:
            tol = abs(t_val) * rel_epsilon
        else:
            tol = epsilon
        mask &= (np.abs(s_vals - t_val) <= tol)

    return mask


def method_l2_ball(
    synthetic: pd.DataFrame,
    target: pd.Series,
    categorical_columns: List[str],
    continuous_columns: List[str],
    radius: float = 0.5,
    normalize: bool = True,
) -> np.ndarray:
    """
    方法二：分类型必须精确一致；连续列视为向量，标准化后 L2 距离 <= radius。
    """
    cols = categorical_columns + continuous_columns
    S, t = _align_and_dtype(synthetic, target, cols)
    mask = np.ones(len(S), dtype=bool)

    for c in categorical_columns:
        if c not in S.columns:
            continue
        s_vals = S[c].astype(str).values
        t_val = str(t[c]) if pd.notna(t[c]) else "nan"
        mask &= (s_vals == t_val)

    if not continuous_columns:
        return mask

    cont_common = [c for c in continuous_columns if c in S.columns]
    if not cont_common:
        return mask

    S_num = S[cont_common].apply(pd.to_numeric, errors="coerce")
    t_num = t[cont_common].apply(pd.to_numeric, errors="coerce")
    # 含 NaN 的行在该方法下视为不匹配（或可改为跳过该维度）
    t_vec = t_num.values.astype(float)
    if np.any(np.isnan(t_vec)):
        return np.zeros(len(S), dtype=bool)

    X = S_num.values.astype(float)
    if normalize:
        col_min = np.nanmin(X, axis=0)
        col_max = np.nanmax(X, axis=0)
        range_ = col_max - col_min
        range_[range_ < 1e-12] = 1.0
        X = (X - col_min) / range_
        t_vec = (t_vec - col_min) / range_
    # NaN 在 synthetic 中该维度用 target 值填充再算距离（或排除）
    np.putmask(X, np.isnan(X), np.tile(t_vec, (len(X), 1)))
    dist = np.linalg.norm(X - t_vec, axis=1)
    mask &= (dist <= radius)
    return mask


def method_fraction_match(
    synthetic: pd.DataFrame,
    target: pd.Series,
    categorical_columns: List[str],
    continuous_columns: List[str],
    fraction: float = 0.9,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """
    方法三：每列在容差内算匹配；要求匹配的列数 >= fraction * 总列数。
    """
    cols = categorical_columns + continuous_columns
    S, t = _align_and_dtype(synthetic, target, cols)
    n_cols = len(cols)
    required = max(1, int(np.ceil(fraction * n_cols)))
    match_count = np.zeros(len(S), dtype=int)

    for c in categorical_columns:
        if c not in S.columns:
            continue
        s_vals = S[c].astype(str).values
        t_val = str(t[c]) if pd.notna(t[c]) else "nan"
        match_count += (s_vals == t_val).astype(int)

    for c in continuous_columns:
        if c not in S.columns:
            continue
        s_vals = pd.to_numeric(S[c], errors="coerce").values
        t_val = pd.to_numeric(t[c], errors="coerce")
        if pd.isna(t_val):
            match_count += np.isnan(s_vals).astype(int)
        else:
            match_count += (np.abs(s_vals - t_val) <= epsilon).astype(int)

    return match_count >= required


def method_binned_exact(
    synthetic: pd.DataFrame,
    target: pd.Series,
    categorical_columns: List[str],
    continuous_columns: List[str],
    n_bins: int = 10,
) -> np.ndarray:
    """
    方法四：连续列按分箱离散化后，整行与 target 的离散化结果精确匹配。
    """
    cols = categorical_columns + continuous_columns
    S, t = _align_and_dtype(synthetic, target, cols)
    S_bin = S.copy()
    t_bin = t.copy()

    for c in continuous_columns:
        if c not in S.columns:
            continue
        s_vals = pd.to_numeric(S[c], errors="coerce")
        t_val = pd.to_numeric(t[c], errors="coerce")
        combined = pd.concat([s_vals, pd.Series([t_val])], ignore_index=True).dropna()
        if len(combined) < 2:
            try:
                S_bin[c] = 0
                t_bin[c] = 0
            except Exception:
                S_bin[c] = "nan"
                t_bin[c] = "nan"
            continue
        try:
            _, bin_edges = np.histogram(combined, bins=n_bins)
            bin_edges = np.asarray(bin_edges)
            bin_edges[-1] += 1e-9
            # digitize: 右开区间，返回 1..n_bins
            s_filled = s_vals.fillna(np.nanmin(combined))
            S_bin[c] = np.digitize(s_filled.values, bin_edges[1:], right=False)
            t_bin[c] = np.digitize(np.atleast_1d(t_val), bin_edges[1:], right=False)[0]
        except Exception:
            S_bin[c] = s_vals.astype(str)
            t_bin[c] = str(t_val)

    mask = np.ones(len(S), dtype=bool)
    for c in cols:
        if c not in S_bin.columns:
            continue
        s_vals = S_bin[c].astype(str).values
        t_val = str(t_bin[c]) if pd.notna(t_bin[c]) else "nan"
        mask &= (s_vals == t_val)
    return mask


def run_all_methods(
    synthetic: pd.DataFrame,
    target: pd.Series,
    schema: Optional[Schema] = None,
    *,
    epsilon: float = 1e-6,
    rel_epsilon: Optional[float] = None,
    l2_radius: float = 0.5,
    fraction: float = 0.9,
    n_bins: int = 10,
) -> Dict[str, Tuple[np.ndarray, pd.DataFrame]]:
    """
    对 synthetic 与 target 运行多种近似匹配方法，返回每种方法的 (mask, synthetic_matched_df)。

    Returns:
        {
            "epsilon": (mask, matched_df),
            "l2_ball": (mask, matched_df),
            "fraction_match": (mask, matched_df),
            "binned_exact": (mask, matched_df),
        }
    """
    cols, cat, cont = _feature_columns(synthetic, schema)
    # target 可能是 DataFrame 的一行
    if isinstance(target, pd.DataFrame):
        target = target.iloc[0]

    results = {}

    m1 = method_epsilon(synthetic, target, cat, cont, epsilon=epsilon, rel_epsilon=rel_epsilon)
    results["epsilon"] = (m1, synthetic.loc[m1].reset_index(drop=True))

    m2 = method_l2_ball(synthetic, target, cat, cont, radius=l2_radius, normalize=True)
    results["l2_ball"] = (m2, synthetic.loc[m2].reset_index(drop=True))

    m3 = method_fraction_match(synthetic, target, cat, cont, fraction=fraction, epsilon=epsilon)
    results["fraction_match"] = (m3, synthetic.loc[m3].reset_index(drop=True))

    m4 = method_binned_exact(synthetic, target, cat, cont, n_bins=n_bins)
    results["binned_exact"] = (m4, synthetic.loc[m4].reset_index(drop=True))

    return results
