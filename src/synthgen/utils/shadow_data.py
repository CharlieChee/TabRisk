"""
Shadow Modelling 所需 candidate / aux 数据生成。

用于 dataset-level targeted MIA（leave-one-out 语义）：
- candidate.csv = train + out_candidate，含 is_member 标签
- aux.csv = test 剩余样本，与 candidate 在记录层面不重叠
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


SHADOW_MEMBER_COL = "is_member"


def generate_candidate_and_aux(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    n_out_candidate: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[Path, Path]:
    """
    从 train/test 生成 candidate.csv 与 aux.csv，并保存到 output_dir。

    - out_candidate: 从 test 中随机抽取 n_out_candidate 条（默认与 train 行数相同）
    - candidate.csv = concat(train + out_candidate)，添加 is_member 列（train=1, out_candidate=0）
    - aux.csv = test 中剩余样本（与 candidate 在记录层面不重叠）

    Args:
        train_df: 训练集
        test_df: 测试/ holdout 集
        output_dir: 输出目录
        n_out_candidate: out_candidate 数量，默认等于 len(train_df)
        random_state: 随机种子

    Returns:
        (candidate_path, aux_path)
    """
    if n_out_candidate is None:
        n_out_candidate = len(train_df)

    n_train = len(train_df)
    n_test = len(test_df)

    # 若 test 不足，则尽可能抽取
    n_sample = min(n_out_candidate, n_test)
    if n_sample < n_out_candidate and n_out_candidate > 0:
        logger.warning(
            f"test 仅 {n_test} 行，不足 n_out_candidate={n_out_candidate}，"
            f"将抽取 {n_sample} 条作为 out_candidate"
        )

    rng = np.random.default_rng(random_state)

    # 从 test 中随机抽取 out_candidate
    if n_sample > 0 and n_test > 0:
        indices = rng.choice(test_df.index, size=n_sample, replace=False)
        out_candidate_df = test_df.loc[indices].copy()
        out_candidate_df[SHADOW_MEMBER_COL] = 0

        # aux = test 中剩余样本，与 candidate 在记录层面不重叠（candidate=train+out_candidate，aux=test\out_candidate）
        aux_df = test_df.drop(indices).reset_index(drop=True)
    else:
        out_candidate_df = pd.DataFrame(columns=list(train_df.columns) + [SHADOW_MEMBER_COL])
        aux_df = test_df.copy()

    # candidate = train + out_candidate
    train_with_label = train_df.copy()
    train_with_label[SHADOW_MEMBER_COL] = 1
    candidate_df = pd.concat(
        [train_with_label, out_candidate_df],
        axis=0,
        ignore_index=True,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_path = output_dir / "candidate.csv"
    aux_path = output_dir / "aux.csv"

    candidate_df.to_csv(candidate_path, index=False)
    aux_df.to_csv(aux_path, index=False)

    logger.info(
        f"Shadow candidate/aux 已生成: candidate={candidate_path} "
        f"({len(candidate_df)} 行, member={n_train}, non_member={len(out_candidate_df)}), "
        f"aux={aux_path} ({len(aux_df)} 行)"
    )
    return candidate_path, aux_path
