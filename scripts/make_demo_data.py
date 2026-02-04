#!/usr/bin/env python3
"""生成包含混合类型的演示数据脚本。"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def generate_demo_data(n_rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    生成包含混合类型的演示数据。

    Args:
        n_rows: 生成的行数
        seed: 随机种子

    Returns:
        生成的 DataFrame
    """
    np.random.seed(seed)

    # age: 连续数值（整数，18-80）
    age = np.random.randint(18, 81, n_rows)

    # income: 连续数值（浮点数，正态分布）
    income = np.random.normal(50000, 20000, n_rows)
    income = np.clip(income, 0, 200000)  # 限制在合理范围

    # education: 分类（有序分类）
    education_levels = ["小学", "初中", "高中", "本科", "硕士", "博士"]
    education = np.random.choice(education_levels, n_rows, p=[0.1, 0.15, 0.25, 0.3, 0.15, 0.05])

    # gender: 分类（二元分类）
    gender = np.random.choice(["男", "女"], n_rows, p=[0.52, 0.48])

    # country: 分类（多分类）
    countries = ["中国", "美国", "日本", "德国", "法国", "英国", "其他"]
    country = np.random.choice(countries, n_rows, p=[0.4, 0.2, 0.1, 0.1, 0.05, 0.05, 0.1])

    # target: 目标变量（可以是分类或连续，这里用二元分类）
    # 让 target 与 income 和 education 有一定相关性
    target_probs = 1 / (1 + np.exp(-(income / 10000 - 3 + (pd.Categorical(education).codes / 2))))
    target = np.random.binomial(1, target_probs, n_rows).astype(bool)

    # 创建 DataFrame
    df = pd.DataFrame({
        "age": age,
        "income": income.round(2),
        "education": education,
        "gender": gender,
        "country": country,
        "target": target,
    })

    return df


def main():
    """主函数。"""
    parser = argparse.ArgumentParser(description="生成包含混合类型的演示数据")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/demo.csv",
        help="输出文件路径（默认: data/demo.csv）",
    )
    parser.add_argument(
        "--rows",
        "-n",
        type=int,
        default=1000,
        help="生成的行数（默认: 1000）",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="随机种子（默认: 42）",
    )

    args = parser.parse_args()

    # 生成数据
    print(f"生成 {args.rows} 行演示数据...")
    df = generate_demo_data(n_rows=args.rows, seed=args.seed)

    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"✓ 演示数据已生成: {output_path}")
    print(f"  数据形状: {df.shape[0]} 行 x {df.shape[1]} 列")
    print(f"\n列信息:")
    print(f"  - age: 连续数值 (18-80)")
    print(f"  - income: 连续数值 (收入)")
    print(f"  - education: 分类 (小学/初中/高中/本科/硕士/博士)")
    print(f"  - gender: 分类 (男/女)")
    print(f"  - country: 分类 (国家)")
    print(f"  - target: 二元分类 (True/False)")
    print(f"\n前 5 行预览:")
    print(df.head().to_string())
    print(f"\n数据类型:")
    print(df.dtypes)
    print(f"\n基本统计:")
    print(df.describe(include="all"))


if __name__ == "__main__":
    main()
