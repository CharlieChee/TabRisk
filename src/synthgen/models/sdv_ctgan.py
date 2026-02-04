"""SDV CTGAN model implementation."""

import pickle
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from loguru import logger

try:
    from sdv.tabular import CTGAN
except ImportError:
    raise ImportError(
        "SDV 未安装。请运行: pip install sdv"
    )

from synthgen.models.base import BaseModel
from synthgen.data.schema import Schema


class SDVCTGANModel(BaseModel):
    """基于 SDV CTGAN 的生成模型。"""

    def __init__(
        self,
        epochs: int = 300,
        batch_size: int = 500,
        verbose: bool = True,
        generator_dim: Optional[list] = None,
        discriminator_dim: Optional[list] = None,
        generator_lr: float = 2e-4,
        discriminator_lr: float = 2e-4,
        generator_decay: float = 1e-6,
        discriminator_decay: float = 1e-6,
    ):
        """
        初始化 CTGAN 模型。

        Args:
            epochs: 训练轮数
            batch_size: 批次大小
            verbose: 是否显示训练进度
            generator_dim: 生成器维度
            discriminator_dim: 判别器维度
            generator_lr: 生成器学习率
            discriminator_lr: 判别器学习率
            generator_decay: 生成器权重衰减
            discriminator_decay: 判别器权重衰减
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.generator_dim = generator_dim or [256, 256]
        self.discriminator_dim = discriminator_dim or [256, 256]
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        self.generator_decay = generator_decay
        self.discriminator_decay = discriminator_decay

        self.model: Optional[CTGAN] = None
        self.schema: Optional[Schema] = None

    def fit(self, data: pd.DataFrame, schema: Schema) -> None:
        """
        训练 CTGAN 模型。

        Args:
            data: 训练数据
            schema: 数据 schema
        """
        logger.info("开始训练 CTGAN 模型...")
        logger.info(f"训练数据形状: {data.shape}")
        logger.info(f"训练轮数: {self.epochs}, 批次大小: {self.batch_size}")

        self.schema = schema

        # 创建 CTGAN 模型
        self.model = CTGAN(
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            generator_dim=tuple(self.generator_dim),
            discriminator_dim=tuple(self.discriminator_dim),
            generator_lr=self.generator_lr,
            discriminator_lr=self.discriminator_lr,
            generator_decay=self.generator_decay,
            discriminator_decay=self.discriminator_decay,
        )

        # 训练模型
        self.model.fit(data)

        logger.info("模型训练完成")

    def sample(self, num_rows: int) -> pd.DataFrame:
        """
        生成合成数据。

        Args:
            num_rows: 生成的行数

        Returns:
            合成的 DataFrame
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用 fit()")

        logger.info(f"生成 {num_rows} 行合成数据...")

        synthetic_data = self.model.sample(num_rows=num_rows)

        logger.info(f"生成完成，形状: {synthetic_data.shape}")

        return synthetic_data

    def save(self, file_path: str) -> None:
        """
        保存模型。

        Args:
            file_path: 保存路径
        """
        if self.model is None:
            raise ValueError("模型尚未训练，无法保存")

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # SDV 模型有自己的 save 方法，但我们用 pickle 保存整个对象
        # 这样可以同时保存 schema 和其他元数据
        save_dict = {
            "model": self.model,
            "schema": self.schema,
            "config": {
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "generator_dim": self.generator_dim,
                "discriminator_dim": self.discriminator_dim,
                "generator_lr": self.generator_lr,
                "discriminator_lr": self.discriminator_lr,
                "generator_decay": self.generator_decay,
                "discriminator_decay": self.discriminator_decay,
            },
        }

        with open(path, "wb") as f:
            pickle.dump(save_dict, f)

        logger.info(f"模型已保存到: {file_path}")

    @classmethod
    def load(cls, file_path: str) -> "SDVCTGANModel":
        """
        加载模型。

        Args:
            file_path: 模型文件路径

        Returns:
            加载的模型实例
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在: {file_path}")

        logger.info(f"加载模型: {file_path}")

        with open(path, "rb") as f:
            save_dict = pickle.load(f)

        instance = cls(**save_dict["config"])
        instance.model = save_dict["model"]
        instance.schema = save_dict["schema"]

        logger.info("模型加载完成")

        return instance
