"""Training entry point."""

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

from synthgen.data.load import load_csv
from synthgen.data.schema import infer_schema
from synthgen.models.base import BaseModel

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
    params = model_cfg.get("params", {})

    # 动态导入并实例化
    module_path, class_name = target.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    model_class = getattr(module, class_name)

    logger.info(f"实例化模型: {target}")
    return model_class(**params)


def instantiate_data_loader(cfg: DictConfig, project_root: Path) -> pd.DataFrame:
    """实例化数据加载器并加载数据。"""
    data_cfg = cfg.data
    if "_target_" not in data_cfg:
        raise ValueError("数据配置中缺少 _target_ 字段")

    target = data_cfg._target_
    params = data_cfg.get("params", {}).copy()

    # 处理文件路径（如果是相对路径，则相对于项目根目录）
    if "file_path" in params:
        file_path = Path(params["file_path"])
        if not file_path.is_absolute():
            params["file_path"] = str(project_root / file_path)

    # 动态导入并调用
    module_path, func_name = target.rsplit(".", 1)
    module = __import__(module_path, fromlist=[func_name])
    load_func = getattr(module, func_name)

    logger.info(f"加载数据: {target}")
    return load_func(**params)


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="train")
def main(cfg: DictConfig) -> None:
    """训练主函数。"""
    console = Console()

    # 支持简写参数：data.path -> data.params.file_path
    if OmegaConf.is_config(cfg.get("data")) and "path" in cfg.data:
        if "params" not in cfg.data:
            OmegaConf.set(cfg.data, "params", {})
        OmegaConf.set(cfg.data.params, "file_path", cfg.data.pop("path"))
        logger.info(f"检测到简写参数 data.path，已映射到 data.params.file_path")

    # 获取项目根目录（Hydra 会改变工作目录，所以需要绝对路径）
    project_root = PROJECT_ROOT.resolve()
    
    # 创建输出目录（相对于项目根目录）
    output_dir = project_root / cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置日志
    setup_logging(output_dir)

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

    # 保存模型
    logger.info("=" * 60)
    logger.info("步骤 5: 保存模型")
    logger.info("=" * 60)
    model_path = output_dir / "model.pkl"
    model.save(str(model_path))

    # 保存运行配置
    run_config_path = output_dir / "run_config.yaml"
    with open(run_config_path, "w", encoding="utf-8") as f:
        OmegaConf.save(config=cfg, f=f)
    logger.info(f"运行配置已保存到: {run_config_path}")

    # 创建 latest 符号链接（用于 sample.py 快速访问）
    latest_dir = project_root / "outputs" / "latest"
    latest_dir.parent.mkdir(parents=True, exist_ok=True)
    if latest_dir.exists() or latest_dir.is_symlink():
        latest_dir.unlink()
    latest_dir.symlink_to(output_dir.absolute())

    console.print(f"[bold green]训练完成！[/bold green]")
    console.print(f"模型保存在: {model_path}")
    console.print(f"Schema 保存在: {output_dir / 'schema.json'}")


if __name__ == "__main__":
    main()
