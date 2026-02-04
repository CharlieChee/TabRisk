#!/bin/bash

# Smoke test 脚本
# 快速验证整个流程是否正常工作

set -e  # 遇到错误立即退出

echo "=========================================="
echo "开始 Smoke Test"
echo "=========================================="

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# 创建必要的目录
mkdir -p data outputs

# 生成演示数据
echo ""
echo "步骤 1: 生成演示数据..."
python3 << 'EOF'
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)
n_rows = 500

data = {
    "age": np.random.randint(18, 80, n_rows),
    "income": np.random.normal(50000, 15000, n_rows),
    "city": np.random.choice(["北京", "上海", "广州", "深圳"], n_rows),
    "is_active": np.random.choice([True, False], n_rows),
    "score": np.random.uniform(0, 100, n_rows),
}

df = pd.DataFrame(data)
Path("data/demo.csv").parent.mkdir(parents=True, exist_ok=True)
df.to_csv("data/demo.csv", index=False)
print(f"演示数据已生成: data/demo.csv ({df.shape[0]} 行 x {df.shape[1]} 列)")
EOF

# 运行 pytest
echo ""
echo "步骤 2: 运行 pytest 测试..."
python3 -m pytest tests/test_smoke.py -v --tb=short

echo ""
echo "=========================================="
echo "Smoke Test 完成！"
echo "=========================================="
