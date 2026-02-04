"""Sampling entry point."""

import sys
from pathlib import Path
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from rich.console import Console

from synthgen.models.base import BaseModel
from synthgen.data.schema import Schema

# 获取项目根目录（configs/ 在项目根目录）
# 当使用 python -m synthgen.sample 时，工作目录是项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
CONFIG_DIR = PROJECT_ROOT / "configs"


def setup_logging() -> None:
    """设置日志。"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="sample")
def main(cfg: DictConfig) -> None:
    """采样主函数。"""
    console = Console()

    setup_logging()

    # 支持简写参数：sample.n -> num_rows
    if OmegaConf.is_config(cfg.get("sample")) and "n" in cfg.sample:
        OmegaConf.set(cfg, "num_rows", cfg.sample.pop("n"))
        logger.info(f"检测到简写参数 sample.n，已映射到 num_rows")

    # 获取项目根目录（Hydra 会改变工作目录，所以需要绝对路径）
    project_root = PROJECT_ROOT.resolve()

    console.print(f"[bold green]开始生成合成数据[/bold green]")

    # 加载模型
    logger.info("=" * 60)
    logger.info("步骤 1: 加载模型")
    logger.info("=" * 60)

    # 处理相对路径（相对于项目根目录）
    model_path = Path(cfg.model_path)
    if not model_path.is_absolute():
        model_path = project_root / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 从模型路径推断模型类型（这里简化处理，假设是 SDVCTGANModel）
    # 实际应该从保存的配置中读取
    from synthgen.models.sdv_ctgan import SDVCTGANModel

    model: BaseModel = SDVCTGANModel.load(str(model_path))
    logger.info(f"模型加载完成: {model_path}")

    # 加载 schema（可选，用于验证）
    schema_path = Path(cfg.schema_path)
    if not schema_path.is_absolute():
        schema_path = project_root / schema_path
    if schema_path.exists():
        schema = Schema.load(str(schema_path))
        logger.info(f"Schema 加载完成: {schema_path}")
        logger.info(f"原始数据形状: {schema.total_rows} 行 x {schema.total_columns} 列")
    else:
        logger.warning(f"Schema 文件不存在: {schema_path}，跳过验证")

    # 生成合成数据
    logger.info("=" * 60)
    logger.info("步骤 2: 生成合成数据")
    logger.info("=" * 60)

    num_rows = cfg.num_rows
    logger.info(f"生成 {num_rows} 行数据...")

    synthetic_df = model.sample(num_rows)

    # 保存合成数据
    logger.info("=" * 60)
    logger.info("步骤 3: 保存合成数据")
    logger.info("=" * 60)

    # 处理输出路径（相对于项目根目录）
    output_path = Path(cfg.output_path)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    synthetic_df.to_csv(output_path, index=False)
    logger.info(f"合成数据已保存到: {output_path}")

    # 显示统计信息
    console.print(f"[bold green]生成完成！[/bold green]")
    console.print(f"生成数据形状: {synthetic_df.shape}")
    console.print(f"保存路径: {output_path}")

    # 显示前几行
    console.print("\n[bold]前 5 行数据预览:[/bold]")
    console.print(synthetic_df.head().to_string())


if __name__ == "__main__":
    main()
