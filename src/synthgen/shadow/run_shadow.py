"""
Shadow 数据生成流程（严格 leave-one-out 语义）。

对每个 target，每轮 shadow：
- base_aux_k：从 aux 一次性抽取 (N-1) 条，in/out 共用
- in：train_in_k = base_aux_k ∪ {target}
- out：train_out_k = base_aux_k ∪ {x}，x 为 non-member 且 x≠target
- 唯一差异为是否包含 target
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from synthgen.data.schema import Schema
from synthgen.models.base import BaseModel
from synthgen.preprocess import get_preprocessor
from synthgen.utils.shadow_data import SHADOW_MEMBER_COL


def _instantiate_model_from_cfg(model_cfg: DictConfig, seed: int) -> BaseModel:
    """从配置实例化模型。"""
    if "_target_" not in model_cfg:
        raise ValueError("模型配置中缺少 _target_ 字段")

    target = model_cfg._target_
    params = dict(OmegaConf.to_container(model_cfg.get("params", {}), resolve=True) or {})
    params.setdefault("random_state", seed)

    module_path, class_name = target.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    model_class = getattr(module, class_name)
    return model_class(**params)


def _load_preprocessor(run_dir: Path, fit_on: pd.DataFrame) -> Any:
    """从 preprocess.json 加载预处理器并在 fit_on 上 fit（恢复内部状态）。"""
    preprocess_path = run_dir / "preprocess.json"
    if not preprocess_path.exists():
        return get_preprocessor("none", {})

    with open(preprocess_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    name = state.get("name", "none")
    params = state.get("params", {})
    preproc = get_preprocessor(name, params)
    # fit 以恢复 _log1p_applied 等内部状态
    preproc.fit(fit_on)
    return preproc


def _prepare_train_df_for_model(df: pd.DataFrame, member_col: str = SHADOW_MEMBER_COL) -> pd.DataFrame:
    """移除 is_member 等 shadow 专用列，并统一 dtype。"""
    out = df.copy()
    if member_col in out.columns:
        out = out.drop(columns=[member_col])
    # 与主流程一致：string/object -> category
    for col in out.select_dtypes(include=["string"]).columns:
        out[col] = out[col].astype("category")
    for col in out.select_dtypes(include=["object"]).columns:
        out[col] = out[col].astype("category")
    return out


def run_shadow_generation(
    run_dir: Path,
    cfg: DictConfig,
    project_root: Path,
) -> None:
    """
    在主训练与评估完成后执行 Shadow 数据生成（仅当 shadow_model=true 时调用）。

    严格 leave-one-out 语义：
    - 每轮 k：base_aux_k 从 aux 一次性抽取 (N-1) 条
    - in：train_in_k = base_aux_k ∪ {target}
    - out：train_out_k = base_aux_k ∪ {x}，x 为 non-member
    - base_aux_k 在 in/out 中完全相同

    Shadow 阶段不做 utility 评估、不保存 train/test，只生成 synthetic_round_*.csv。
    """
    run_dir = Path(run_dir)
    candidate_path = run_dir / "candidate.csv"
    aux_path = run_dir / "aux.csv"
    schema_path = run_dir / "schema.json"

    if not candidate_path.exists() or not aux_path.exists():
        logger.warning("Shadow 跳过：candidate.csv 或 aux.csv 不存在，请确保已生成 shadow 候选数据")
        return
    if not schema_path.exists():
        logger.warning("Shadow 跳过：schema.json 不存在")
        return

    shadow_cfg = cfg.get("shadow") or OmegaConf.create({})
    run_cfg = OmegaConf.load(run_dir / "train_config.yaml")

    # 解析 shadow 参数，null 时回退到主流程配置
    train_rows = OmegaConf.select(run_cfg, "train_rows", default=None)
    synthetic_rows = int(OmegaConf.select(run_cfg, "synthetic_rows", default=0) or 0)
    main_seed = int(OmegaConf.select(run_cfg, "seed", default=42) or 42)

    N = shadow_cfg.get("shadow_train_rows")
    if N is None:
        N = train_rows
    if N is None or (hasattr(N, "__iter__") and not isinstance(N, int)):
        N = None
    else:
        N = int(N)

    synth_rows = shadow_cfg.get("shadow_synthetic_rows")
    if synth_rows is None:
        synth_rows = synthetic_rows
    if synth_rows is None:
        synth_rows = 1000
    synth_rows = int(synth_rows)

    seed = shadow_cfg.get("random_seed")
    if seed is None:
        seed = main_seed
    seed = int(seed)

    num_rounds = int(shadow_cfg.get("num_shadow_rounds", 1))
    strategy = str(shadow_cfg.get("target_selection_strategy", "random_k"))
    max_targets = int(shadow_cfg.get("max_targets", 10))

    # 加载数据
    candidate_df = pd.read_csv(candidate_path)
    aux_df = pd.read_csv(aux_path)
    schema = Schema.load(str(schema_path))

    if SHADOW_MEMBER_COL not in candidate_df.columns:
        logger.warning("Shadow 跳过：candidate.csv 缺少 is_member 列")
        return

    train_members = candidate_df[candidate_df[SHADOW_MEMBER_COL] == 1].reset_index(drop=True)
    non_members = candidate_df[candidate_df[SHADOW_MEMBER_COL] == 0].reset_index(drop=True)

    if len(train_members) == 0:
        logger.warning("Shadow 跳过：无 train member 可作为 target")
        return
    if len(non_members) == 0:
        logger.warning("Shadow 跳过：无 non-member，无法构造 out 训练集")
        return

    # 选择 targets
    if strategy == "all":
        targets = list(range(len(train_members)))
    else:
        n_t = min(max_targets, len(train_members))
        rng = np.random.default_rng(seed)
        targets = rng.choice(len(train_members), size=n_t, replace=False).tolist()

    if N is None:
        N = len(train_members)
    n_base = N - 1

    if len(aux_df) < n_base:
        logger.warning(
            f"Shadow 跳过：aux 仅 {len(aux_df)} 行，需要至少 {n_base} 行作为 base_aux"
        )
        return

    # 预处理器：与主流程一致，fit 于主训练集（train_members），仅做 transform 时与主 run 一致
    fit_df = _prepare_train_df_for_model(train_members)
    preproc = _load_preprocessor(run_dir, fit_df)

    model_cfg = run_cfg.model
    shadow_dir = run_dir / "shadow"
    shadow_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Shadow 开始：targets={len(targets)}, rounds={num_rounds}, N={N}, "
        f"synthetic_rows={synth_rows}, seed={seed}"
    )

    for t_idx, target_idx in enumerate(targets):
        target_row = train_members.iloc[target_idx : target_idx + 1]
        target_dir = shadow_dir / f"target_{t_idx}"
        target_dir.mkdir(parents=True, exist_ok=True)

        for k in range(num_rounds):
            rng = np.random.default_rng(seed + t_idx * 10000 + k)

            # 1. 从 aux 一次性抽取 base_aux_k（in/out 共用）
            aux_indices = rng.choice(len(aux_df), size=n_base, replace=False)
            base_aux_k = aux_df.iloc[aux_indices].reset_index(drop=True)

            # 2. 选择 x（non-member，x != target）
            x_idx = rng.integers(0, len(non_members))
            x_row = non_members.iloc[x_idx : x_idx + 1].reset_index(drop=True)

            # 3. 构造 train_in_k = base_aux_k ∪ {target}
            train_in_k = pd.concat([base_aux_k, target_row], axis=0, ignore_index=True)
            train_in_k = _prepare_train_df_for_model(train_in_k)

            # 4. 构造 train_out_k = base_aux_k ∪ {x}
            train_out_k = pd.concat([base_aux_k, x_row], axis=0, ignore_index=True)
            train_out_k = _prepare_train_df_for_model(train_out_k)

            # 5. 预变换
            train_in_transformed = preproc.transform(train_in_k)
            train_out_transformed = preproc.transform(train_out_k)

            # 6. 训练并生成 in
            model_in = _instantiate_model_from_cfg(model_cfg, seed + k * 2)
            model_in.fit(train_in_transformed, schema)
            synth_in = model_in.sample(synth_rows)
            synth_in = preproc.inverse_transform(synth_in)
            out_in_path = target_dir / f"synthetic_round_{k}_in.csv"
            synth_in.to_csv(out_in_path, index=False)
            logger.info(f"Shadow target_{t_idx} round_{k} in 已保存: {out_in_path}")

            # 7. 训练并生成 out
            model_out = _instantiate_model_from_cfg(model_cfg, seed + k * 2 + 1)
            model_out.fit(train_out_transformed, schema)
            synth_out = model_out.sample(synth_rows)
            synth_out = preproc.inverse_transform(synth_out)
            out_out_path = target_dir / f"synthetic_round_{k}_out.csv"
            synth_out.to_csv(out_out_path, index=False)
            logger.info(f"Shadow target_{t_idx} round_{k} out 已保存: {out_out_path}")

    logger.info(f"Shadow 完成: {shadow_dir}")
