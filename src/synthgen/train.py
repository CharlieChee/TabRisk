"""Training entry point."""

import json
import os
import sys
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from rich.console import Console
from rich.table import Table

from synthgen.data.load import load_csv, load_dataset
from synthgen.data.schema import infer_schema
from synthgen.models.base import BaseModel
from synthgen.preprocess import get_preprocessor


# 限制 CPU 线程数，避免多用户环境下过度占用
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_NO_TORCH_WARNING", "1")


def _maybe_set_cuda_visible_devices_from_cfg(cfg: DictConfig) -> None:
    """从 cfg.model.params.cuda_visible_devices 读取并设置 CUDA_VISIBLE_DEVICES（仅使用 model.params）。"""
    cuda_vis = None
    try:
        if hasattr(cfg.model, "params") and cfg.model.params is not None:
            cuda_vis = cfg.model.params.get("cuda_visible_devices", None)
    except Exception:
        cuda_vis = None
    if cuda_vis is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_vis)

# 获取项目根目录（configs/ 在项目根目录）
# 当使用 python -m synthgen.train 时，工作目录是项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
CONFIG_DIR = PROJECT_ROOT / "configs"


def _safe_str(val) -> str:
    """将配置值转为目录名安全的字符串（去除特殊字符）。"""
    s = str(val).strip()
    for c in "/\\:*?\"<>|":
        s = s.replace(c, "_")
    return s if s else ""


def build_output_dir(cfg: DictConfig, project_root: Path) -> Path:
    """构建包含详细信息的输出目录名，便于从目录名识别实验配置。

    示例：train_adult_hf_preprocess_none_train5000_synth2000_ctgan_100iter_20260205_132935
    """
    parts = ["train"]

    # 从 Hydra runtime.choices 获取 config group 选择（数据名、模型名等）
    choices = None
    try:
        from hydra.core.hydra_config import HydraConfig
        hydra_cfg = HydraConfig.get()
        choices = getattr(getattr(hydra_cfg, "runtime", None), "choices", None)
    except Exception:
        pass

    # 数据名：如 standard/adult_hf -> adult_hf
    if choices:
        data_choice = OmegaConf.select(choices, "data", default=None)
        if data_choice:
            data_name = str(data_choice).split("/")[-1] if "/" in str(data_choice) else str(data_choice)
            if data_name:
                parts.append(_safe_str(data_name))

    # preprocess：monotonic 或 none
    preprocess_choice = None
    if choices:
        preprocess_choice = OmegaConf.select(choices, "preprocess", default=None)
    if not preprocess_choice:
        preprocess_choice = OmegaConf.select(cfg, "preprocess.name", default=None)
    if preprocess_choice:
        parts.append(f"preprocess_{_safe_str(str(preprocess_choice))}")

    # train 条数
    train_rows = getattr(cfg, "train_rows", None)
    if train_rows is not None and int(train_rows) > 0:
        parts.append(f"train{int(train_rows)}")

    # synth 条数
    synthetic_rows = getattr(cfg, "synthetic_rows", None)
    if synthetic_rows is not None:
        try:
            n = int(synthetic_rows)
            if n >= 0:
                parts.append(f"synth{n}")
        except (TypeError, ValueError):
            pass

    # model 名：优先从 choices 获取，否则用 model.name
    model_str = None
    if choices:
        model_choice = OmegaConf.select(choices, "model", default=None)
        model_str = str(model_choice) if model_choice else None
    if not model_str:
        model_str = OmegaConf.select(cfg, "model.name", default=None)
    if model_str and str(model_str) not in ("None", "null"):
        parts.append(_safe_str(str(model_str).split(".")[-1]))

    # n_iter
    n_iter = OmegaConf.select(cfg, "model.params.n_iter", default=None)
    if n_iter is not None:
        try:
            parts.append(f"{int(n_iter)}iter")
        except (TypeError, ValueError):
            pass

    # batch_size（优先使用 model.params.batch_size，其次顶层 batch_size）
    batch_size = OmegaConf.select(cfg, "model.params.batch_size", default=None)
    if batch_size is None:
        batch_size = getattr(cfg, "batch_size", None)
    if batch_size is not None:
        try:
            parts.append(f"bs{int(batch_size)}")
        except (TypeError, ValueError):
            parts.append(f"bs{_safe_str(batch_size)}")

    # 影子模型模式：将 max_targets、num_shadow_rounds 写入路径；control_branch 时加 control
    shadow_model_enabled = getattr(cfg, "shadow_model", False) or False
    if shadow_model_enabled:
        num_shadow_rounds = OmegaConf.select(cfg, "shadow.num_shadow_rounds", default=None)
        if num_shadow_rounds is not None:
            try:
                parts.append(f"rounds{int(num_shadow_rounds)}")
            except (TypeError, ValueError):
                parts.append(f"rounds{_safe_str(num_shadow_rounds)}")
        max_targets = OmegaConf.select(cfg, "shadow.max_targets", default=None)
        if max_targets is not None:
            try:
                parts.append(f"candidate{int(max_targets)}")
            except (TypeError, ValueError):
                parts.append(f"candidate{_safe_str(max_targets)}")
        if OmegaConf.select(cfg, "shadow.control_branch", default=False):
            parts.append("control")
        shadow_seed = OmegaConf.select(cfg, "shadow.random_seed", default=None)
        if shadow_seed is not None:
            try:
                parts.append(f"seed{int(shadow_seed)}")
            except (TypeError, ValueError):
                parts.append(f"seed{_safe_str(shadow_seed)}")

    # 时间戳
    parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))

    dir_name = "_".join(str(p) for p in parts)
    return project_root / "outputs" / dir_name


def setup_logging(output_dir: Path) -> None:
    """设置日志。"""
    log_file = output_dir / "train.log"
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG",
    )


def set_seed(seed: int) -> None:
    """设置随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    # Python 的随机种子
    import os
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"随机种子设置为: {seed}")


def instantiate_model(cfg: DictConfig) -> BaseModel:
    """实例化模型。"""
    model_cfg = cfg.model
    if "_target_" not in model_cfg:
        raise ValueError("模型配置中缺少 _target_ 字段")

    target = model_cfg._target_
    params = dict(model_cfg.get("params", {}))
    # 统一注入随机种子，保证 SynthCity 等后端可复现
    if "seed" in cfg:
        params.setdefault("random_state", cfg.seed)

    # 动态导入并实例化
    module_path, class_name = target.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    model_class = getattr(module, class_name)

    logger.info(f"实例化模型: {target}")
    return model_class(**params)


def instantiate_data_loader(cfg: DictConfig, project_root: Path) -> pd.DataFrame:
    """实例化数据加载器并加载数据（支持 load_csv 与 load_dataset）。"""
    data_cfg = cfg.data
    if "_target_" not in data_cfg:
        raise ValueError("数据配置中缺少 _target_ 字段")

    target = data_cfg._target_
    params = dict(OmegaConf.to_container(data_cfg.get("params", {}), resolve=True) or {})

    # 处理文件路径（如果是相对路径，则相对于项目根目录）
    if "file_path" in params:
        file_path = Path(params["file_path"])
        if not file_path.is_absolute():
            params["file_path"] = str(project_root / file_path)

    # 标准数据集 load_dataset：将 cache_dir 解析为绝对路径并注入 project_root
    if "source" in params:
        if "cache_dir" in params and params["cache_dir"]:
            cache = Path(params["cache_dir"])
            if not cache.is_absolute():
                params["cache_dir"] = str(project_root / cache)
        params.setdefault("project_root", str(project_root))
        logger.info(f"加载数据: {target} (source={params.get('source')}, name={params.get('name')})")
        return load_dataset(**params)

    # 动态导入并调用（如 load_csv）
    module_path, func_name = target.rsplit(".", 1)
    module = __import__(module_path, fromlist=[func_name])
    load_func = getattr(module, func_name)
    logger.info(f"加载数据: {target}")
    return load_func(**params)


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="train")
def main(cfg: DictConfig) -> None:
    """训练主函数。"""

    # 使用 HuggingFace 数据源时，必须在首次 import datasets/huggingface_hub 前设置 HF_ENDPOINT，
    # 否则会出现 "Cannot send a request, as the client has been closed" 等错误
    if OmegaConf.is_config(cfg.get("data")):
        params = getattr(cfg.data, "params", None) or {}
        if OmegaConf.select(params, "source") == "hf" and "HF_ENDPOINT" not in os.environ:
            from synthgen.data.load import HF_MIRROR_DEFAULT
            os.environ["HF_ENDPOINT"] = HF_MIRROR_DEFAULT.rstrip("/")

    # 在导入 torch / 创建 SynthCity 插件前，根据 cfg.model.params.cuda_visible_devices 绑定 CUDA_VISIBLE_DEVICES
    _maybe_set_cuda_visible_devices_from_cfg(cfg)

    import torch  # 延迟导入，确保上面的环境变量已生效

    # 限制 torch CPU 线程数，避免 CPU 线程爆炸
    torch.set_num_threads(8)

    logger.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES", "not set"))
    logger.info(
        "torch=%s, cuda_available=%s, device_count=%s",
        torch.__version__,
        torch.cuda.is_available(),
        torch.cuda.device_count(),
    )

    console = Console()

    # 支持简写参数：data.path -> data.params.file_path（只读 path，不修改 cfg.data 结构）
    if OmegaConf.is_config(cfg.get("data")) and "path" in cfg.data:
        path_value = cfg.data.path  # 只读，避免 struct mode 下 pop 触发 ConfigTypeError
        if "params" not in cfg.data:
            OmegaConf.update(cfg.data, {"params": OmegaConf.create({})}, force_add=True)
        cfg.data.params["file_path"] = path_value
        logger.info(f"检测到简写参数 data.path，已映射到 data.params.file_path")

    # 获取项目根目录（Hydra 会改变工作目录，所以需要绝对路径）
    project_root = PROJECT_ROOT.resolve()

    # 创建输出目录：默认使用详细名称（含 data/train_rows/model/n_iter）；显式指定 output_dir 时使用之
    use_detailed = getattr(cfg, "use_detailed_output_dir", True)
    if use_detailed:
        output_dir = build_output_dir(cfg, project_root)
    else:
        output_dir = Path(cfg.output_dir) if Path(cfg.output_dir).is_absolute() else project_root / cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置日志
    setup_logging(output_dir)

    logger.info(
        "torch=%s, cuda_available=%s, device_count=%s",
        torch.__version__,
        torch.cuda.is_available(),
        torch.cuda.device_count(),
    )

    console.print(f"[bold green]开始训练[/bold green]")
    console.print(f"输出目录: {output_dir}")

    # 设置随机种子
    set_seed(cfg.seed)

    # 加载数据
    logger.info("=" * 60)
    logger.info("步骤 1: 加载数据")
    logger.info("=" * 60)
    df = instantiate_data_loader(cfg, project_root)
    logger.info(f"数据形状: {df.shape}")

    # 保留完整数据引用（抽样前），用于计算与 original 互斥的 holdout
    df_all = df.copy()

    # 训练数据抽样：若配置了 train_rows > 0，则随机抽取 n 条参与训练
    train_rows = getattr(cfg, "train_rows", None)
    if train_rows is not None and int(train_rows) > 0:
        n_sample = min(int(train_rows), len(df))
        sample_result = df.sample(n=n_sample, random_state=cfg.seed)
        df = sample_result.reset_index(drop=True)
        holdout_df = df_all.drop(sample_result.index).reset_index(drop=True)
        logger.info(f"已随机抽样 {n_sample} 条数据用于训练（train_rows={train_rows}）")
    else:
        holdout_df = pd.DataFrame(columns=df.columns)

    # 预处理（fit_transform 在 schema 推断 / 模型 fit 之前）
    preproc_cfg = getattr(cfg, "preprocess", None) or OmegaConf.create({"name": "none", "params": {}})
    # 兼容 params 为 DictConfig / dict / None 三种情况
    params_obj = preproc_cfg.get("params", {}) if hasattr(preproc_cfg, "get") else {}
    if params_obj is None:
        params_obj = {}
    if OmegaConf.is_config(params_obj):
        preproc_params = OmegaConf.to_container(params_obj, resolve=True)
    else:
        # 已经是普通 dict 或其他可用容器，直接使用
        preproc_params = params_obj

    preproc = get_preprocessor(
        OmegaConf.select(preproc_cfg, "name", default="none"),
        preproc_params,
    )
    train_df = preproc.fit_transform(df)
    logger.info(f"预处理: {preproc.state_dict()['name']}, 训练数据形状: {train_df.shape}")

    # 推断 schema（使用 transform 后的 df，__is_zero 等新列会进入 schema）
    logger.info("=" * 60)
    logger.info("步骤 2: 推断 schema")
    logger.info("=" * 60)
    schema = infer_schema(train_df)
    schema.save(str(output_dir / "schema.json"))

    # 显示 schema 信息
    table = Table(title="Schema 信息")
    table.add_column("列名", style="cyan")
    table.add_column("类型", style="magenta")
    table.add_column("唯一值", style="green")
    table.add_column("缺失值", style="yellow")

    for col_schema in schema.columns:
        col_type = "分类" if col_schema.is_categorical else "连续"
        table.add_row(
            col_schema.name,
            col_type,
            str(col_schema.unique_count) if col_schema.unique_count else "N/A",
            str(col_schema.null_count),
        )

    console.print(table)

    # 实例化模型
    logger.info("=" * 60)
    logger.info("步骤 3: 实例化模型")
    logger.info("=" * 60)
    model = instantiate_model(cfg)

    # 训练模型
    logger.info("=" * 60)
    logger.info("步骤 4: 训练模型")
    logger.info("=" * 60)
    # 在训练前，将 pandas 的 string/object 列统一转换为 category，
    # 以避免下游 Schema 验证在遇到 pandas StringDtype 时出错
    string_cols = train_df.select_dtypes(include=["string"]).columns
    if len(string_cols) > 0:
        logger.info(f"检测到 {len(string_cols)} 个 string dtype 列，转换为 category: {list(string_cols)}")
        train_df[string_cols] = train_df[string_cols].astype("category")

    object_cols = train_df.select_dtypes(include=["object"]).columns
    if len(object_cols) > 0:
        logger.info(f"检测到 {len(object_cols)} 个 object dtype 列，转换为 category: {list(object_cols)}")
        train_df[object_cols] = train_df[object_cols].astype("category")

    model.fit(train_df, schema)

    # 步骤 5: 保存可序列化产物（禁止 pickle plugin，SynthCity 不保证 pickle-safe）
    logger.info("=" * 60)
    logger.info("步骤 5: 保存产物")
    logger.info("=" * 60)

    # 保存运行配置
    run_config_path = output_dir / "train_config.yaml"
    with open(run_config_path, "w", encoding="utf-8") as f:
        OmegaConf.save(config=cfg, f=f)
    logger.info(f"训练配置已保存: {run_config_path}")

    # 保存随机种子
    (output_dir / "random_seed.txt").write_text(str(cfg.seed), encoding="utf-8")
    logger.info(f"随机种子已保存: {output_dir / 'random_seed.txt'}")

    # 保存原始训练数据集（抽样后的 df，未 transform）
    original_path = output_dir / "original.csv"
    df.to_csv(original_path, index=False)
    logger.info(f"原始训练数据已保存: {original_path} ({len(df)} 行)")
    # 同时保存一份 train.csv（与 original.csv 内容相同）
    train_path = output_dir / "train.csv"
    df.to_csv(train_path, index=False)
    logger.info(f"训练数据副本已保存: {train_path} ({len(df)} 行)")

    # 保存 holdout 数据（与 original.csv 行级互斥，用于 MIA 等评估）
    test_path = output_dir / "test.csv"
    holdout_df.to_csv(test_path, index=False)
    logger.info(f"Holdout 评估数据已保存: {test_path} ({len(holdout_df)} 行)")

    # Shadow Modelling：仅在 shadow_model=true 时生成 candidate/aux
    shadow_model_enabled = getattr(cfg, "shadow_model", False) or False
    if shadow_model_enabled:
        from synthgen.utils.shadow_data import generate_candidate_and_aux

        n_out = int(train_rows) if train_rows is not None and int(train_rows) > 0 else len(df)
        generate_candidate_and_aux(
            train_df=df,
            test_df=holdout_df,
            output_dir=output_dir,
            n_out_candidate=n_out,
            random_state=cfg.seed,
        )

    # 保存预处理规则（便于复现）
    preprocess_path = output_dir / "preprocess.json"
    with open(preprocess_path, "w", encoding="utf-8") as f:
        json.dump(preproc.state_dict(), f, ensure_ascii=False, indent=2)
    logger.info(f"预处理规则已保存: {preprocess_path}")

    # 按 save_model 模式保存元数据（不 pickle plugin）
    save_mode = getattr(cfg, "save_model", False) or False
    if save_mode == "metadata_only":
        model.save_metadata(str(output_dir))
        logger.info(f"模型元数据已保存: {output_dir / 'model_metadata.yaml'}")

    # 训练完成后立即生成合成数据（使用内存中的 model，避免依赖 pickle 加载）
    synthetic_rows = int(getattr(cfg, "synthetic_rows", 0) or 0)
    if synthetic_rows > 0:
        synthetic_df = model.sample(synthetic_rows)
        synthetic_df = preproc.inverse_transform(synthetic_df)
        synthetic_path = output_dir / "synthetic.csv"
        synthetic_df.to_csv(synthetic_path, index=False)
        logger.info(f"合成数据已生成: {synthetic_path} ({synthetic_rows} 行)")

    # 创建 latest 符号链接（用于快速访问最新输出）
    latest_dir = project_root / "outputs" / "latest"
    latest_dir.parent.mkdir(parents=True, exist_ok=True)
    if latest_dir.exists() or latest_dir.is_symlink():
        latest_dir.unlink()
    latest_dir.symlink_to(output_dir.absolute())

    # 训练完成标记
    (output_dir / ".done").write_text("", encoding="utf-8")

    console.print(f"[bold green]训练完成！[/bold green]")
    console.print(f"Schema: {output_dir / 'schema.json'}")
    console.print(f"配置: {run_config_path}")
    console.print(f"原始训练数据: {output_dir / 'original.csv'} ({len(df)} 行)")
    console.print(f"Holdout 评估数据: {output_dir / 'test.csv'} ({len(holdout_df)} 行)")
    if synthetic_rows > 0:
        console.print(f"合成数据: {output_dir / 'synthetic.csv'} ({synthetic_rows} 行)")

    # 可选：训练保存完成后自动在该 run 目录下执行 sdv + synthcity 评估
    auto_evaluate = getattr(cfg, "auto_evaluate", True)
    if auto_evaluate and synthetic_rows > 0:
        from synthgen.evaluation.runner import run_evaluation
        run_dir_str = str(output_dir.resolve())
        eval_base = {"run_dir": run_dir_str, "output_subdir": "eval", "target_col": None, "task_type": None}
        for ev_name in ("sdv", "synthcity"):
            try:
                ev_cfg = OmegaConf.load(CONFIG_DIR / "evaluator" / f"{ev_name}.yaml")
                eval_cfg = OmegaConf.merge(OmegaConf.create(eval_base), OmegaConf.create({"evaluator": ev_cfg}))
                logger.info(f"自动评估: evaluator={ev_name} run_dir={run_dir_str}")
                run_evaluation(eval_cfg)
            except Exception as e:
                logger.warning(f"自动评估 evaluator={ev_name} 失败: {e}")

    # Shadow Modelling：仅在 shadow_model=true 时执行 leave-one-out 影子数据生成
    if shadow_model_enabled:
        try:
            from synthgen.shadow.run_shadow import run_shadow_generation

            logger.info("开始 Shadow 数据生成...")
            run_shadow_generation(
                run_dir=output_dir,
                cfg=cfg,
                project_root=project_root,
            )
        except Exception as e:
            logger.warning(f"Shadow 数据生成失败: {e}")
            raise


if __name__ == "__main__":
    main()
