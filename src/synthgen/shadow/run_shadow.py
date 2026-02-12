"""
Shadow 数据生成流程（严格 leave-one-out 语义）。

对每个 target，每轮 shadow：
- base_aux_k：从 aux 一次性抽取 (N-1) 条，in/out 共用
- in：train_in_k = base_aux_k ∪ {target}
- out：train_out_k = base_aux_k ∪ {x}，x 为 non-member 且 x≠target
- 唯一差异为是否包含 target

shadow_model=true 时支持多进程并行（num_workers × gpu_ids），每进程绑定一个 GPU。
"""

import hashlib
import multiprocessing
import os
import shutil
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


def _row_content_hash(row: pd.Series) -> str:
    """对一行数据做稳定哈希，便于在 manifest 中标识该记录。"""
    key = "_".join(str(v) for v in row.astype(str).tolist())
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


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

    注意：
    - 此函数用于 legacy 引擎的 ProcessPoolExecutor 路径，保持原有行为不变；
    - 高性能 worker 引擎不会调用该函数，而是复用内部逻辑以减少重复 I/O。
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
    n_member = int((candidate_df[SHADOW_MEMBER_COL] == 1).sum())
    non_member_candidate_indices = list(range(n_member, len(candidate_df)))
    fit_df = _prepare_train_df_for_model(candidate_df[candidate_df[SHADOW_MEMBER_COL] == 1])
    preproc = _load_preprocessor(run_dir, fit_df)
    model_cfg = run_cfg.model

    # target_idx 为 candidate 中的行号（candidate_row_idx）
    target_row = candidate_df.iloc[target_idx : target_idx + 1].reset_index(drop=True)
    target_dir = run_dir / "shadow" / f"target_{t_idx}"
    target_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed + t_idx * 10000 + k)
    aux_indices = rng.choice(len(aux_df), size=n_base, replace=False)
    base_aux_k = aux_df.iloc[aux_indices].reset_index(drop=True)
    # x 为 non-member 且 x != target，保证 in/out 仅差 target 这一条
    x_candidates = [i for i in non_member_candidate_indices if i != target_idx]
    if not x_candidates:
        x_candidates = [i for i in range(len(candidate_df)) if i != target_idx]
    x_candidate_idx = int(rng.choice(x_candidates))
    x_row = candidate_df.iloc[x_candidate_idx : x_candidate_idx + 1].reset_index(drop=True)

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


def _run_shadow_generation_legacy(
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
    target_mix = bool(shadow_cfg.get("target_mix", True))

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
    n_member = len(train_members)
    n_non_member = len(non_members)
    # candidate 中 member 行为 0..n_member-1，non-member 行为 n_member..n_member+n_non_member-1
    member_candidate_indices = list(range(0, n_member))
    non_member_candidate_indices = list(range(n_member, n_member + n_non_member))

    if n_member == 0:
        logger.warning("Shadow 跳过：无 train member 可作为 target")
        return
    if n_non_member == 0:
        logger.warning("Shadow 跳过：无 non-member，无法构造 out 训练集")
        return

    rng = np.random.default_rng(seed)
    # targets = 候选人在 candidate.csv 中的行号列表（candidate_row_idx）
    if target_mix:
        # 混合：约一半来自 train，一半来自 non-member，便于 MIA 正负样本平衡
        if strategy == "all":
            targets = member_candidate_indices + non_member_candidate_indices
            rng.shuffle(targets)
        else:
            n_from_member = min(max_targets // 2, n_member)
            n_from_non = min(max_targets - n_from_member, n_non_member)
            if n_from_non < max_targets - n_from_member:
                n_from_member = min(max_targets - n_from_non, n_member)
            chosen_m = rng.choice(member_candidate_indices, size=min(n_from_member, len(member_candidate_indices)), replace=False)
            chosen_n = rng.choice(non_member_candidate_indices, size=min(n_from_non, len(non_member_candidate_indices)), replace=False)
            targets = np.concatenate([np.atleast_1d(chosen_m), np.atleast_1d(chosen_n)]).astype(int).tolist()
            rng.shuffle(targets)
    else:
        # 仅从 train 选取（旧逻辑）
        if strategy == "all":
            targets = member_candidate_indices.copy()
        else:
            n_t = min(max_targets, n_member)
            targets = rng.choice(member_candidate_indices, size=n_t, replace=False).tolist()

    if N is None:
        N = n_member
    n_base = N - 1

    if len(aux_df) < n_base:
        logger.warning(
            f"Shadow 跳过：aux 仅 {len(aux_df)} 行，需要至少 {n_base} 行作为 base_aux"
        )
        return

    # 预处理器在子进程内按需加载，此处仅做目录与任务列表准备
    shadow_dir = run_dir / "shadow"
    shadow_dir.mkdir(parents=True, exist_ok=True)

    # 保存 target 清单：每个 target_0, target_1, ... 对应 candidate 中哪一行、是否 member、来源等，便于 MIA 评估与复现
    manifest_rows = []
    for t_idx, candidate_row_idx in enumerate(targets):
        candidate_row_idx = int(candidate_row_idx)
        row = candidate_df.iloc[candidate_row_idx]
        is_member = int(row[SHADOW_MEMBER_COL]) if SHADOW_MEMBER_COL in row else 0
        manifest_rows.append({
            "target_idx": t_idx,
            "target_dir": f"target_{t_idx}",
            "candidate_row_idx": candidate_row_idx,
            "train_row_idx": candidate_row_idx if is_member else -1,
            "is_member": is_member,
            "source": "train" if is_member else "out_candidate",
            "row_hash": _row_content_hash(row),
        })
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = shadow_dir / "target_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    logger.info(f"Shadow target 清单已保存: {manifest_path} ({len(manifest_df)} 条)")

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


def _shadow_worker_process(
    run_dir_str: str,
    gpu_id: int,
    jobs: List[Tuple[int, int, int]],
    N: int,
    n_base: int,
    synth_rows: int,
    seed: int,
) -> None:
    """
    高性能 Shadow worker 进程。

    - 每个进程绑定到单个 GPU（通过 CUDA_VISIBLE_DEVICES）；
    - 在进程启动后一次性加载 candidate/aux/schema/preprocessor；
    - 顺序执行分配给该 GPU 的所有 (t_idx, target_idx, k) 任务；
    - 内部逻辑与 _run_one_shadow_job 完全对齐，以保持 MIA 语义。
    """
    run_dir = Path(run_dir_str)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    logger.info(
        f"[worker] GPU {gpu_id}: start, jobs={len(jobs)}, N={N}, n_base={n_base}, synth_rows={synth_rows}, seed={seed}"
    )

    # 加载配置与数据：每个 worker 进程只执行一次
    run_cfg = OmegaConf.load(run_dir / "train_config.yaml")
    candidate_df = pd.read_csv(run_dir / "candidate.csv")
    aux_df = pd.read_csv(run_dir / "aux.csv")
    schema = Schema.load(str(run_dir / "schema.json"))
    n_member = int((candidate_df[SHADOW_MEMBER_COL] == 1).sum())
    non_member_candidate_indices = list(range(n_member, len(candidate_df)))

    # 预处理器：仅在 member 行上 fit 一次，复用到所有任务
    fit_df = _prepare_train_df_for_model(candidate_df[candidate_df[SHADOW_MEMBER_COL] == 1])
    preproc = _load_preprocessor(run_dir, fit_df)
    model_cfg = run_cfg.model

    for (t_idx, target_idx, k) in jobs:
        # target_idx 为 candidate 中的行号（candidate_row_idx）
        target_row = candidate_df.iloc[target_idx : target_idx + 1].reset_index(drop=True)
        target_dir = run_dir / "shadow" / f"target_{t_idx}"
        target_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(seed + t_idx * 10000 + k)
        aux_indices = rng.choice(len(aux_df), size=n_base, replace=False)
        base_aux_k = aux_df.iloc[aux_indices].reset_index(drop=True)

        # x 为 non-member 且 x != target，保证 in/out 仅差 target 这一条
        x_candidates = [i for i in non_member_candidate_indices if i != target_idx]
        if not x_candidates:
            x_candidates = [i for i in range(len(candidate_df)) if i != target_idx]
        x_candidate_idx = int(rng.choice(x_candidates))
        x_row = candidate_df.iloc[x_candidate_idx : x_candidate_idx + 1].reset_index(drop=True)

        train_in_k = pd.concat([base_aux_k, target_row], axis=0, ignore_index=True)
        train_in_k = _prepare_train_df_for_model(train_in_k)
        train_out_k = pd.concat([base_aux_k, x_row], axis=0, ignore_index=True)
        train_out_k = _prepare_train_df_for_model(train_out_k)

        train_in_transformed = preproc.transform(train_in_k)
        train_out_transformed = preproc.transform(train_out_k)

        # 注意：种子公式与 _run_one_shadow_job 完全一致
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

        logger.debug(f"[worker] GPU {gpu_id}: done target_{t_idx} round_{k}")

    logger.info(f"[worker] GPU {gpu_id}: all jobs done ({len(jobs)} jobs)")


def _shadow_reuse_worker_process(
    run_dir_str: str,
    gpu_id: int,
    jobs: List[Tuple[int, int, int]],
    base_aux_indices_list: List[np.ndarray],
    synth_rows: int,
    seed: int,
) -> None:
    """
    Shadow 复用模式的 worker 进程。

    与 _shadow_worker_process 的主要区别：
    - 任务单位从 (t_idx, target_idx, k) 换成了 (dataset_id, base_id, candidate_idx)
    - 每个任务只训练一个“base_aux ∪ {candidate_row}”对应的唯一模型
    - 输出统一写到 shadow/_reuse/dataset_{dataset_id}.csv

    上层会在所有唯一数据集训练完成后，将这些 CSV 复制到
    target_k/synthetic_round_*.csv 位置，保证对外接口与原逻辑完全一致。
    """
    run_dir = Path(run_dir_str)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    logger.info(
        f"[reuse-worker] GPU {gpu_id}: start, jobs={len(jobs)}, "
        f"bases={len(base_aux_indices_list)}, synth_rows={synth_rows}, seed={seed}"
    )

    # 每个 worker 进程仅加载一次配置与数据
    run_cfg = OmegaConf.load(run_dir / "train_config.yaml")
    candidate_df = pd.read_csv(run_dir / "candidate.csv")
    aux_df = pd.read_csv(run_dir / "aux.csv")
    schema = Schema.load(str(run_dir / "schema.json"))

    # 预处理器：在 member 行上 fit 一次，复用于所有数据集
    if SHADOW_MEMBER_COL in candidate_df.columns:
        fit_df = _prepare_train_df_for_model(candidate_df[candidate_df[SHADOW_MEMBER_COL] == 1])
    else:
        fit_df = _prepare_train_df_for_model(candidate_df)
    preproc = _load_preprocessor(run_dir, fit_df)
    model_cfg = run_cfg.model

    reuse_dir = run_dir / "shadow" / "_reuse"
    reuse_dir.mkdir(parents=True, exist_ok=True)

    for (dataset_id, base_id, candidate_idx) in jobs:
        if base_id < 0 or base_id >= len(base_aux_indices_list):
            logger.warning(f"[reuse-worker] GPU {gpu_id}: invalid base_id={base_id}, skip dataset_id={dataset_id}")
            continue

        aux_indices = base_aux_indices_list[base_id]
        base_aux = aux_df.iloc[aux_indices].reset_index(drop=True)
        candidate_row = candidate_df.iloc[candidate_idx : candidate_idx + 1].reset_index(drop=True)

        train_df = pd.concat([base_aux, candidate_row], axis=0, ignore_index=True)
        train_df = _prepare_train_df_for_model(train_df)
        train_transformed = preproc.transform(train_df)

        # 复用模式下不需要与 legacy 完全对齐的随机种子，仅保证可复现即可
        model = _instantiate_model_from_cfg(model_cfg, seed + dataset_id * 2)
        model.fit(train_transformed, schema)
        synth = model.sample(synth_rows)
        synth = preproc.inverse_transform(synth)

        out_path = reuse_dir / f"dataset_{dataset_id}.csv"
        synth.to_csv(out_path, index=False)
        logger.debug(f"[reuse-worker] GPU {gpu_id}: done dataset_{dataset_id} (base_id={base_id}, cand_idx={candidate_idx})")

    logger.info(f"[reuse-worker] GPU {gpu_id}: all jobs done ({len(jobs)} jobs)")


def _run_shadow_generation_worker(
    run_dir: Path,
    cfg: DictConfig,
    project_root: Path,
) -> None:
    """
    高性能 Shadow 引擎实现：每 GPU 一个长寿命 worker 进程。

    语义保证：
    - 与 legacy 引擎使用相同的 target 选择、N/n_base 计算、随机种子公式；
    - 仅改变调度与数据加载方式（减少重复 I/O、进程创建和预处理开销）。
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
    target_mix = bool(shadow_cfg.get("target_mix", True))

    # 并行配置：worker 引擎支持每 GPU 多个并发 worker 进程
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

    _wc = shadow_cfg.get("worker_concurrency_per_gpu", 1)
    try:
        worker_concurrency = int(_wc) if _wc is not None else 1
    except (TypeError, ValueError):
        worker_concurrency = 1
    worker_concurrency = max(1, worker_concurrency)

    # 加载数据（仅在主进程执行一次）
    candidate_df = pd.read_csv(candidate_path)
    aux_df = pd.read_csv(aux_path)
    schema = Schema.load(str(schema_path))

    if SHADOW_MEMBER_COL not in candidate_df.columns:
        logger.warning("Shadow 跳过：candidate.csv 缺少 is_member 列")
        return

    train_members = candidate_df[candidate_df[SHADOW_MEMBER_COL] == 1].reset_index(drop=True)
    non_members = candidate_df[candidate_df[SHADOW_MEMBER_COL] == 0].reset_index(drop=True)
    n_member = len(train_members)
    n_non_member = len(non_members)
    # candidate 中 member 行为 0..n_member-1，non-member 行为 n_member..n_member+n_non_member-1
    member_candidate_indices = list(range(0, n_member))
    non_member_candidate_indices = list(range(n_member, n_member + n_non_member))

    if n_member == 0:
        logger.warning("Shadow 跳过：无 train member 可作为 target")
        return
    if n_non_member == 0:
        logger.warning("Shadow 跳过：无 non-member，无法构造 out 训练集")
        return

    rng = np.random.default_rng(seed)
    # targets = 候选人在 candidate.csv 中的行号列表（candidate_row_idx）
    if target_mix:
        # 混合：约一半来自 train，一半来自 non-member，便于 MIA 正负样本平衡
        if strategy == "all":
            targets = member_candidate_indices + non_member_candidate_indices
            rng.shuffle(targets)
        else:
            n_from_member = min(max_targets // 2, n_member)
            n_from_non = min(max_targets - n_from_member, n_non_member)
            if n_from_non < max_targets - n_from_member:
                n_from_member = min(max_targets - n_from_non, n_member)
            chosen_m = rng.choice(member_candidate_indices, size=min(n_from_member, len(member_candidate_indices)), replace=False)
            chosen_n = rng.choice(non_member_candidate_indices, size=min(n_from_non, len(non_member_candidate_indices)), replace=False)
            targets = np.concatenate([np.atleast_1d(chosen_m), np.atleast_1d(chosen_n)]).astype(int).tolist()
            rng.shuffle(targets)
    else:
        # 仅从 train 选取（旧逻辑）
        if strategy == "all":
            targets = member_candidate_indices.copy()
        else:
            n_t = min(max_targets, n_member)
            targets = rng.choice(member_candidate_indices, size=n_t, replace=False).tolist()

    if N is None:
        N = n_member
    n_base = N - 1

    if len(aux_df) < n_base:
        logger.warning(
            f"Shadow 跳过：aux 仅 {len(aux_df)} 行，需要至少 {n_base} 行作为 base_aux"
        )
        return

    # 预处理器在 worker 进程内按需加载，此处仅做目录与任务列表准备
    shadow_dir = run_dir / "shadow"
    shadow_dir.mkdir(parents=True, exist_ok=True)

    # 保存 target 清单：与 legacy 引擎保持完全一致
    manifest_rows = []
    for t_idx, candidate_row_idx in enumerate(targets):
        candidate_row_idx = int(candidate_row_idx)
        row = candidate_df.iloc[candidate_row_idx]
        is_member = int(row[SHADOW_MEMBER_COL]) if SHADOW_MEMBER_COL in row else 0
        manifest_rows.append({
            "target_idx": t_idx,
            "target_dir": f"target_{t_idx}",
            "candidate_row_idx": candidate_row_idx,
            "train_row_idx": candidate_row_idx if is_member else -1,
            "is_member": is_member,
            "source": "train" if is_member else "out_candidate",
            "row_hash": _row_content_hash(row),
        })
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = shadow_dir / "target_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    logger.info(f"Shadow target 清单已保存: {manifest_path} ({len(manifest_df)} 条)")

    # 构建所有 (t_idx, target_idx, k) 任务
    run_dir_str = str(run_dir.resolve())
    job_tuples = [
        (t_idx, target_idx, k)
        for t_idx, target_idx in enumerate(targets)
        for k in range(num_rounds)
    ]

    # 构建 worker 槽位列表：[(gpu_id, slot_idx), ...]，用于轮询分配任务
    worker_slots: List[Tuple[int, int]] = []
    for gpu in gpu_ids:
        for slot_idx in range(worker_concurrency):
            worker_slots.append((gpu, slot_idx))

    jobs_per_slot: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {
        slot: [] for slot in worker_slots
    }
    for j, (t_idx, target_idx, k) in enumerate(job_tuples):
        slot = worker_slots[j % len(worker_slots)] if worker_slots else (gpu_ids[0], 0)
        jobs_per_slot[slot].append((t_idx, target_idx, k))

    total_jobs = sum(len(v) for v in jobs_per_slot.values())
    logger.info(
        f"Shadow(worker) 开始：targets={len(targets)}, rounds={num_rounds}, jobs={total_jobs}, "
        f"gpu_ids={gpu_ids}, worker_concurrency_per_gpu={worker_concurrency}, "
        f"N={N}, synthetic_rows={synth_rows}, seed={seed}"
    )

    if total_jobs == 0:
        logger.info("Shadow(worker)：无待执行的任务，直接返回")
        return

    ctx = multiprocessing.get_context("spawn")
    processes: List[multiprocessing.Process] = []
    for (gpu, slot_idx), jobs in jobs_per_slot.items():
        if not jobs:
            continue
        p = ctx.Process(
            target=_shadow_worker_process,
            args=(run_dir_str, int(gpu), jobs, N, n_base, synth_rows, seed),
        )
        p.start()
        processes.append(p)
        logger.info(
            f"Shadow(worker)：已启动 worker 进程 pid={p.pid} 绑定 GPU {gpu} 槽位 {slot_idx}，jobs={len(jobs)}"
        )

    # 等待所有 worker 完成
    for p in processes:
        p.join()
        if p.exitcode != 0:
            logger.warning(f"Shadow(worker)：worker 进程 pid={p.pid} 非零退出码 {p.exitcode}")

    logger.info(f"Shadow(worker) 完成: {shadow_dir}")


def _run_shadow_generation_reuse(
    run_dir: Path,
    cfg: DictConfig,
    project_root: Path,
) -> None:
    """
    Shadow 复用模式：

    目标：
    - 对每个 target 仍然生成 num_shadow_rounds 轮严格 leave-one-out 的 (in, out) pair；
    - 但在内部对训练数据集进行全局去重复用，大幅减少实际训练的模型数量；
    - 外部可见行为完全保持一致：输出目录结构、文件命名、输入配置均不变。
    """
    run_dir = Path(run_dir)
    candidate_path = run_dir / "candidate.csv"
    aux_path = run_dir / "aux.csv"
    schema_path = run_dir / "schema.json"

    if not candidate_path.exists() or not aux_path.exists():
        logger.warning("Shadow(reuse) 跳过：candidate.csv 或 aux.csv 不存在，请确保已生成 shadow 候选数据")
        return
    if not schema_path.exists():
        logger.warning("Shadow(reuse) 跳过：schema.json 不存在")
        return

    shadow_cfg = cfg.get("shadow") or OmegaConf.create({})
    run_cfg = OmegaConf.load(run_dir / "train_config.yaml")

    # 解析 shadow 参数，与 worker 引擎保持一致
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
    target_mix = bool(shadow_cfg.get("target_mix", True))

    # 复用相关参数：可选，提供合理默认值
    _rb = shadow_cfg.get("reuse_num_bases", 10)
    try:
        reuse_num_bases = int(_rb) if _rb is not None else 10
    except (TypeError, ValueError):
        reuse_num_bases = 10
    reuse_num_bases = max(1, reuse_num_bases)

    _rc = shadow_cfg.get("reuse_num_out_candidates", 10)
    try:
        reuse_num_out_candidates = int(_rc) if _rc is not None else 10
    except (TypeError, ValueError):
        reuse_num_out_candidates = 10
    reuse_num_out_candidates = max(1, reuse_num_out_candidates)

    # 并行配置：与 worker 引擎一致
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

    _wc = shadow_cfg.get("worker_concurrency_per_gpu", 1)
    try:
        worker_concurrency = int(_wc) if _wc is not None else 1
    except (TypeError, ValueError):
        worker_concurrency = 1
    worker_concurrency = max(1, worker_concurrency)

    # 加载数据（仅在主进程执行一次）
    candidate_df = pd.read_csv(candidate_path)
    aux_df = pd.read_csv(aux_path)
    schema = Schema.load(str(schema_path))

    if SHADOW_MEMBER_COL not in candidate_df.columns:
        logger.warning("Shadow(reuse) 跳过：candidate.csv 缺少 is_member 列")
        return

    train_members = candidate_df[candidate_df[SHADOW_MEMBER_COL] == 1].reset_index(drop=True)
    non_members = candidate_df[candidate_df[SHADOW_MEMBER_COL] == 0].reset_index(drop=True)
    n_member = len(train_members)
    n_non_member = len(non_members)
    member_candidate_indices = list(range(0, n_member))
    non_member_candidate_indices = list(range(n_member, n_member + n_non_member))

    if n_member == 0:
        logger.warning("Shadow(reuse) 跳过：无 train member 可作为 target")
        return
    if n_non_member == 0:
        logger.warning("Shadow(reuse) 跳过：无 non-member，无法构造 out 训练集")
        return

    rng = np.random.default_rng(seed)
    # 目标样本选择逻辑与 worker 引擎完全一致
    if target_mix:
        if strategy == "all":
            targets = member_candidate_indices + non_member_candidate_indices
            rng.shuffle(targets)
        else:
            n_from_member = min(max_targets // 2, n_member)
            n_from_non = min(max_targets - n_from_member, n_non_member)
            if n_from_non < max_targets - n_from_member:
                n_from_member = min(max_targets - n_from_non, n_member)
            chosen_m = rng.choice(
                member_candidate_indices,
                size=min(n_from_member, len(member_candidate_indices)),
                replace=False,
            )
            chosen_n = rng.choice(
                non_member_candidate_indices,
                size=min(n_from_non, len(non_member_candidate_indices)),
                replace=False,
            )
            targets = np.concatenate([np.atleast_1d(chosen_m), np.atleast_1d(chosen_n)]).astype(int).tolist()
            rng.shuffle(targets)
    else:
        if strategy == "all":
            targets = member_candidate_indices.copy()
        else:
            n_t = min(max_targets, n_member)
            targets = rng.choice(member_candidate_indices, size=n_t, replace=False).tolist()

    if N is None:
        N = n_member
    n_base = N - 1

    if len(aux_df) < n_base:
        logger.warning(
            f"Shadow(reuse) 跳过：aux 仅 {len(aux_df)} 行，需要至少 {n_base} 行作为 base_aux"
        )
        return

    shadow_dir = run_dir / "shadow"
    shadow_dir.mkdir(parents=True, exist_ok=True)

    # 保存 target_manifest，与其他引擎一致
    manifest_rows = []
    for t_idx, candidate_row_idx in enumerate(targets):
        candidate_row_idx = int(candidate_row_idx)
        row = candidate_df.iloc[candidate_row_idx]
        is_member = int(row[SHADOW_MEMBER_COL]) if SHADOW_MEMBER_COL in row else 0
        manifest_rows.append(
            {
                "target_idx": t_idx,
                "target_dir": f"target_{t_idx}",
                "candidate_row_idx": candidate_row_idx,
                "train_row_idx": candidate_row_idx if is_member else -1,
                "is_member": is_member,
                "source": "train" if is_member else "out_candidate",
                "row_hash": _row_content_hash(row),
            }
        )
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = shadow_dir / "target_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    logger.info(f"Shadow(reuse) target 清单已保存: {manifest_path} ({len(manifest_df)} 条)")

    # 1) 预先生成有限数量的 base_aux（全局共享）
    num_bases = min(reuse_num_bases, max(1, num_rounds))
    base_aux_indices_list: List[np.ndarray] = []
    for b in range(num_bases):
        aux_indices = rng.choice(len(aux_df), size=n_base, replace=False)
        base_aux_indices_list.append(aux_indices)

    # 2) 预先选择少量 non-member 作为 out 的 x 候选，提升复用率
    num_out_pool = min(reuse_num_out_candidates, n_non_member)
    if num_out_pool <= 0:
        logger.warning("Shadow(reuse)：无可用 non-member 作为 out 候选，终止")
        return
    out_candidate_pool = rng.choice(
        non_member_candidate_indices,
        size=num_out_pool,
        replace=False,
    ).tolist()

    # 3) 为每个 (t_idx, k, role) 分配一个唯一数据集键 (base_id, candidate_idx)
    #    并构建全局唯一数据集 -> dataset_id 的映射
    dataset_key_to_id: Dict[Tuple[int, int], int] = {}
    dataset_specs: List[Tuple[int, int]] = []
    usage_records: List[Tuple[int, int, str, int]] = []

    def _get_or_create_dataset_id(base_id: int, cand_idx: int) -> int:
        key = (base_id, cand_idx)
        if key in dataset_key_to_id:
            return dataset_key_to_id[key]
        dataset_id = len(dataset_specs)
        dataset_key_to_id[key] = dataset_id
        dataset_specs.append((base_id, cand_idx))
        return dataset_id

    for t_idx, target_idx in enumerate(targets):
        target_idx = int(target_idx)
        for k in range(num_rounds):
            # 将 base_id 均匀分配到所有 (t_idx, k) 上，提升利用率
            base_id = (t_idx * num_rounds + k) % num_bases

            # in 端：base_aux ∪ {target}
            in_dataset_id = _get_or_create_dataset_id(base_id, target_idx)
            usage_records.append((t_idx, k, "in", in_dataset_id))

            # out 端：base_aux ∪ {x}，优先从 non-member 中选取 x，保证与原逻辑一致：
            # 1) 先在受限的 out_candidate_pool 内选非 target；
            # 2) 若为空，退回到所有 non-member 中选非 target；
            # 3) 若仍为空（极端情况：唯一 non-member 即为 target），
            #    则退回到整个 candidate 中选非 target（与 legacy/worker 行为对齐）。
            valid_out_pool = [i for i in out_candidate_pool if i != target_idx]
            if not valid_out_pool:
                valid_out_pool = [i for i in non_member_candidate_indices if i != target_idx]
            if not valid_out_pool:
                valid_out_pool = [i for i in range(len(candidate_df)) if i != target_idx]
            if not valid_out_pool:
                logger.warning(
                    f"Shadow(reuse)：无法为 target_idx={target_idx} 找到替换样本，跳过该轮 k={k}"
                )
                continue
            x_candidate_idx = int(rng.choice(valid_out_pool))
            out_dataset_id = _get_or_create_dataset_id(base_id, x_candidate_idx)
            usage_records.append((t_idx, k, "out", out_dataset_id))

    total_unique_datasets = len(dataset_specs)
    logger.info(
        f"Shadow(reuse)：规划完成，targets={len(targets)}, rounds={num_rounds}, "
        f"unique_datasets={total_unique_datasets}, "
        f"num_bases={num_bases}, out_pool_size={num_out_pool}"
    )

    # 保存复用模式下的数据集与使用关系的元数据，便于后续分析：
    # - reuse_datasets.csv：每个 dataset_id 对应的 (base_id, candidate_idx) 以及该行是否 member 等信息；
    # - reuse_usage.csv：每个 (target_idx, round, role) 对应使用了哪个 dataset_id。
    reuse_datasets_rows = []
    for dataset_id, (base_id, cand_idx) in enumerate(dataset_specs):
        row = candidate_df.iloc[cand_idx]
        is_member = int(row[SHADOW_MEMBER_COL]) if SHADOW_MEMBER_COL in row else 0
        reuse_datasets_rows.append(
            {
                "dataset_id": dataset_id,
                "base_id": int(base_id),
                "candidate_row_idx": int(cand_idx),
                "is_member": is_member,
                "row_hash": _row_content_hash(row),
            }
        )
    if reuse_datasets_rows:
        reuse_datasets_df = pd.DataFrame(reuse_datasets_rows)
        reuse_datasets_path = shadow_dir / "reuse_datasets.csv"
        reuse_datasets_df.to_csv(reuse_datasets_path, index=False)
        logger.info(
            f"Shadow(reuse)：数据集清单已保存: {reuse_datasets_path} "
            f"(unique_datasets={total_unique_datasets})"
        )

    reuse_usage_rows = []
    for t_idx, k, role, dataset_id in usage_records:
        reuse_usage_rows.append(
            {
                "target_idx": int(t_idx),
                "round": int(k),
                "role": str(role),
                "dataset_id": int(dataset_id),
                "synthetic_path": f"shadow/target_{t_idx}/synthetic_round_{k}_{role}.csv",
            }
        )
    if reuse_usage_rows:
        reuse_usage_df = pd.DataFrame(reuse_usage_rows)
        reuse_usage_path = shadow_dir / "reuse_usage.csv"
        reuse_usage_df.to_csv(reuse_usage_path, index=False)
        logger.info(
            f"Shadow(reuse)：使用关系清单已保存: {reuse_usage_path} "
            f"(rows={len(reuse_usage_rows)})"
        )

    if total_unique_datasets == 0:
        logger.info("Shadow(reuse)：无唯一数据集需要训练，直接返回")
        return

    # 4) 将唯一数据集分配到各 GPU worker 上进行训练
    run_dir_str = str(run_dir.resolve())
    worker_slots: List[Tuple[int, int]] = []
    for gpu in gpu_ids:
        for slot_idx in range(worker_concurrency):
            worker_slots.append((gpu, slot_idx))

    jobs_per_slot: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {slot: [] for slot in worker_slots}
    for dataset_id, (base_id, cand_idx) in enumerate(dataset_specs):
        slot = worker_slots[dataset_id % len(worker_slots)] if worker_slots else (gpu_ids[0], 0)
        jobs_per_slot[slot].append((dataset_id, base_id, cand_idx))

    total_jobs = sum(len(v) for v in jobs_per_slot.values())
    logger.info(
        f"Shadow(reuse) 开始训练唯一数据集：unique_datasets={total_unique_datasets}, "
        f"jobs={total_jobs}, gpu_ids={gpu_ids}, worker_concurrency_per_gpu={worker_concurrency}, "
        f"N={N}, synthetic_rows={synth_rows}, seed={seed}"
    )

    if total_jobs == 0:
        logger.info("Shadow(reuse)：无待执行的训练任务，直接返回")
        return

    ctx = multiprocessing.get_context("spawn")
    processes: List[multiprocessing.Process] = []
    for (gpu, slot_idx), jobs in jobs_per_slot.items():
        if not jobs:
            continue
        p = ctx.Process(
            target=_shadow_reuse_worker_process,
            args=(run_dir_str, int(gpu), jobs, base_aux_indices_list, synth_rows, seed),
        )
        p.start()
        processes.append(p)
        logger.info(
            f"Shadow(reuse)：已启动 worker 进程 pid={p.pid} 绑定 GPU {gpu} 槽位 {slot_idx}，jobs={len(jobs)}"
        )

    for p in processes:
        p.join()
        if p.exitcode != 0:
            logger.warning(f"Shadow(reuse)：worker 进程 pid={p.pid} 非零退出码 {p.exitcode}")

    # 5) 将唯一数据集结果复制到各 target_k/synthetic_round_*.csv
    reuse_dir = shadow_dir / "_reuse"
    if not reuse_dir.exists():
        logger.warning("Shadow(reuse)：_reuse 目录不存在，可能所有 worker 均失败，终止复制阶段")
        return

    logger.info("Shadow(reuse)：开始复制合成数据到各 target_x/synthetic_round_k_in/out.csv")
    for t_idx, k, role, dataset_id in usage_records:
        src_path = reuse_dir / f"dataset_{dataset_id}.csv"
        if not src_path.exists():
            logger.warning(
                f"Shadow(reuse)：源文件不存在，跳过复制：dataset_id={dataset_id}, "
                f"target_{t_idx}, round={k}, role={role}"
            )
            continue
        target_dir = shadow_dir / f"target_{t_idx}"
        target_dir.mkdir(parents=True, exist_ok=True)
        dst_path = target_dir / f"synthetic_round_{k}_{role}.csv"
        # 允许重复保存（相同内容可在不同 target/round 中多次出现）
        shutil.copyfile(src_path, dst_path)

    logger.info(f"Shadow(reuse) 完成: {shadow_dir}")


def run_shadow_generation(
    run_dir: Path,
    cfg: DictConfig,
    project_root: Path,
) -> None:
    """
    Shadow 生成统一入口。

    根据 cfg.shadow.engine 选择具体实现：
    - legacy（默认）：使用现有的 ProcessPoolExecutor 实现，行为完全保持不变；
    - worker：使用每 GPU 一个长寿命 worker 进程的高性能实现。
    """
    shadow_cfg = cfg.get("shadow") or OmegaConf.create({})
    # 新增开关：reuse_mode，仅在显式开启时走复用模式，保证默认行为完全不变
    reuse_mode = bool(shadow_cfg.get("reuse_mode", False))
    if reuse_mode:
        logger.info("Shadow 模式：开启复用模式 (reuse_mode=true)")
        return _run_shadow_generation_reuse(run_dir, cfg, project_root)

    engine = str(shadow_cfg.get("engine", "legacy")).strip().lower()
    if engine not in {"legacy", "worker"}:
        logger.warning(f"未知的 shadow.engine={engine!r}，回退为 'legacy'")
        engine = "legacy"

    logger.info(f"Shadow 引擎选择: {engine}")
    if engine == "worker":
        return _run_shadow_generation_worker(run_dir, cfg, project_root)
    return _run_shadow_generation_legacy(run_dir, cfg, project_root)
