"""Evaluation entry point. 完全解耦的 utility evaluation，不修改 train 逻辑。"""

from pathlib import Path

import hydra
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="evaluate")
def main(cfg: DictConfig) -> None:
    """加载 evaluate 配置并执行评估。"""
    from synthgen.evaluation.runner import run_evaluation
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
