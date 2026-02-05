"""Training entry point."""

import os
import sys
import random
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


# 限制 CPU 线程数，避免多用户环境下过度占用
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_NO_TORCH_WARNING", "1")


def _maybe_set_cuda_visible_devices_from_cfg(cfg: DictConfig) -> None:
    """从 cfg.model.params.gpu_id 读取并设置 CUDA_VISIBLE_DEVICES（仅使用 model.params）。"""
    gpu_id = None
    try:
        if hasattr(cfg.model, "params") and cfg.model.params is not None:
            gpu_id = cfg.model.params.get("gpu_id", None)
    except Exception:
        gpu_id = None
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

# 获取项目根目录（configs/ 在项目根目录）
# 当使用 python -m synthgen.train 时，工作目录是项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
CONFIG_DIR = PROJECT_ROOT / "configs"


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

    # 在导入 torch / 创建 SynthCity 插件前，根据 cfg.model.params.gpu_id 绑定 CUDA_VISIBLE_DEVICES
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
    
    # 创建输出目录（相对于项目根目录）
    output_dir = project_root / cfg.output_dir
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

    # 推断 schema
    logger.info("=" * 60)
    logger.info("步骤 2: 推断 schema")
    logger.info("=" * 60)
    schema = infer_schema(df)
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
    model.fit(df, schema)

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

    # 按 save_model 模式保存元数据（不 pickle plugin）
    save_mode = getattr(cfg, "save_model", False) or False
    if save_mode == "metadata_only":
        model.save_metadata(str(output_dir))
        logger.info(f"模型元数据已保存: {output_dir / 'model_metadata.yaml'}")

    # 训练完成后立即生成合成数据（使用内存中的 model，避免依赖 pickle 加载）
    synthetic_rows = int(getattr(cfg, "synthetic_rows", 0) or 0)
    if synthetic_rows > 0:
        synthetic_df = model.sample(synthetic_rows)
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
    if synthetic_rows > 0:
        console.print(f"合成数据: {output_dir / 'synthetic.csv'} ({synthetic_rows} 行)")


if __name__ == "__main__":
    main()
