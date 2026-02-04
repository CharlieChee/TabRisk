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
python3 scripts/make_demo_data.py --rows 500 --output data/demo.csv

# 运行 pytest
echo ""
echo "步骤 2: 运行 pytest 测试..."
python3 -m pytest tests/test_smoke.py -v --tb=short

echo ""
echo "=========================================="
echo "Smoke Test 完成！"
echo "=========================================="
