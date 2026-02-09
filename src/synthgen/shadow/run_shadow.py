"""
Shadow 数据生成流程（严格 leave-one-out 语义）。

对每个 target，每轮 shadow：
- base_aux_k：从 aux 一次性抽取 (N-1) 条，in/out 共用
- in：train_in_k = base_aux_k ∪ {target}
- out：train_out_k = base_aux_k ∪ {x}，x 为 non-member 且 x≠target
- 唯一差异为是否包含 target

shadow_model=true 时支持多进程并行（num_workers × gpu_ids），每进程绑定一个 GPU。
"""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def _run_one_shadow_job(args: Tuple) -> Optional[str]:
    """
    单任务：在指定 GPU 上完成一个 (target_idx, round_k) 的 in/out 训练与合成。
    供多进程调用，必须为模块级函数且参数可 pickle。
    args: (run_dir_str, t_idx, target_idx, k, gpu_id, N, n_base, synth_rows, seed)
    """
    (
        run_dir_str,
        t_idx,
        target_idx,
        k,
        gpu_id,
        N,
        n_base,
        synth_rows,
        seed,
    ) = args
    run_dir = Path(run_dir_str)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    run_cfg = OmegaConf.load(run_dir / "train_config.yaml")
    candidate_df = pd.read_csv(run_dir / "candidate.csv")
    aux_df = pd.read_csv(run_dir / "aux.csv")
    schema = Schema.load(str(run_dir / "schema.json"))
    train_members = candidate_df[candidate_df[SHADOW_MEMBER_COL] == 1].reset_index(drop=True)
    non_members = candidate_df[candidate_df[SHADOW_MEMBER_COL] == 0].reset_index(drop=True)
    fit_df = _prepare_train_df_for_model(train_members)
    preproc = _load_preprocessor(run_dir, fit_df)
    model_cfg = run_cfg.model

    target_row = train_members.iloc[target_idx : target_idx + 1]
    target_dir = run_dir / "shadow" / f"target_{t_idx}"
    target_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed + t_idx * 10000 + k)
    aux_indices = rng.choice(len(aux_df), size=n_base, replace=False)
    base_aux_k = aux_df.iloc[aux_indices].reset_index(drop=True)
    x_idx = rng.integers(0, len(non_members))
    x_row = non_members.iloc[x_idx : x_idx + 1].reset_index(drop=True)

    train_in_k = pd.concat([base_aux_k, target_row], axis=0, ignore_index=True)
    train_in_k = _prepare_train_df_for_model(train_in_k)
    train_out_k = pd.concat([base_aux_k, x_row], axis=0, ignore_index=True)
    train_out_k = _prepare_train_df_for_model(train_out_k)

    train_in_transformed = preproc.transform(train_in_k)
    train_out_transformed = preproc.transform(train_out_k)

    model_in = _instantiate_model_from_cfg(model_cfg, seed + k * 2)
    model_in.fit(train_in_transformed, schema)
    synth_in = model_in.sample(synth_rows)
    synth_in = preproc.inverse_transform(synth_in)
    out_in_path = target_dir / f"synthetic_round_{k}_in.csv"
    synth_in.to_csv(out_in_path, index=False)

    model_out = _instantiate_model_from_cfg(model_cfg, seed + k * 2 + 1)
    model_out.fit(train_out_transformed, schema)
    synth_out = model_out.sample(synth_rows)
    synth_out = preproc.inverse_transform(synth_out)
    out_out_path = target_dir / f"synthetic_round_{k}_out.csv"
    synth_out.to_csv(out_out_path, index=False)

    return f"target_{t_idx} round_{k}"


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

    # 并行配置：确保为整数/列表，避免 OmegaConf 返回 None 或错误类型导致误走顺序分支
    _nw = shadow_cfg.get("num_workers", 32)
    try:
        num_workers = int(_nw) if _nw is not None else 32
    except (TypeError, ValueError):
        num_workers = 32
    num_workers = max(1, num_workers)

    gpu_ids_raw = shadow_cfg.get("gpu_ids")
    if gpu_ids_raw is not None:
        _list = OmegaConf.to_container(gpu_ids_raw, resolve=True) or []
        gpu_ids = [int(x) for x in _list]
    else:
        _ng = shadow_cfg.get("num_gpus", 8)
        num_gpus = int(_ng) if _ng is not None else 8
        gpu_ids = list(range(max(1, num_gpus)))
    if not gpu_ids:
        gpu_ids = list(range(8))

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

    # 预处理器在子进程内按需加载，此处仅做目录与任务列表准备
    shadow_dir = run_dir / "shadow"
    shadow_dir.mkdir(parents=True, exist_ok=True)

    # 构建所有 (t_idx, target_idx, k) 任务，并按 job_index 分配 gpu_id
    run_dir_str = str(run_dir.resolve())
    job_tuples = [
        (t_idx, target_idx, k)
        for t_idx, target_idx in enumerate(targets)
        for k in range(num_rounds)
    ]
    job_args = [
        (
            run_dir_str,
            t_idx,
            target_idx,
            k,
            gpu_ids[j % len(gpu_ids)] if gpu_ids else 0,
            N,
            n_base,
            synth_rows,
            seed,
        )
        for j, (t_idx, target_idx, k) in enumerate(job_tuples)
    ]

    logger.info(
        f"Shadow 开始：targets={len(targets)}, rounds={num_rounds}, jobs={len(job_args)}, "
        f"num_workers={num_workers}, gpu_ids={gpu_ids}, N={N}, synthetic_rows={synth_rows}, seed={seed}"
    )

    use_parallel = num_workers > 1 and len(gpu_ids) > 0
    if not use_parallel:
        logger.info("Shadow 使用顺序执行（num_workers=%s, len(gpu_ids)=%s）", num_workers, len(gpu_ids))
        for args in job_args:
            _run_one_shadow_job(args)
    else:
        # 使用 spawn 启动子进程，避免 fork 继承主进程 CUDA 上下文导致所有任务挤在同一张卡
        workers = min(num_workers, len(job_args))
        ctx = multiprocessing.get_context("spawn")
        logger.info("Shadow 使用多进程并行: workers=%s, spawn 启动（避免 CUDA fork 继承）", workers)
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
            futures = {executor.submit(_run_one_shadow_job, args): args for args in job_args}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        logger.debug(f"Shadow 完成: {result}")
                except Exception as e:
                    logger.exception(f"Shadow 任务失败: {e}")

    logger.info(f"Shadow 完成: {shadow_dir}")
