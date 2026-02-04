#!/usr/bin/env python3
"""生成演示数据脚本。"""

import pandas as pd
import numpy as np
from pathlib import Path

if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)

    # 生成数据
    n_rows = 500

    data = {
        "age": np.random.randint(18, 80, n_rows),
        "income": np.random.normal(50000, 15000, n_rows),
        "city": np.random.choice(["北京", "上海", "广州", "深圳"], n_rows),
        "is_active": np.random.choice([True, False], n_rows),
        "score": np.random.uniform(0, 100, n_rows),
    }

    df = pd.DataFrame(data)

    # 保存
    output_path = Path("data/demo.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"演示数据已生成: {output_path}")
    print(f"数据形状: {df.shape[0]} 行 x {df.shape[1]} 列")
    print(f"\n前 5 行预览:")
    print(df.head())
