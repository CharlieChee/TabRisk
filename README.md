# TabRisk - Tabular Synthetic Data Generation

基于 SDV CTGAN 的表格合成数据生成工具。

## 功能特性

- 📊 自动推断 CSV 数据的 schema（分类/连续变量）
- 🤖 使用 SDV CTGAN 训练生成模型
- 💾 保存模型和元数据
- 🎲 生成合成数据并导出为 CSV
- ⚙️ 基于 Hydra 的配置管理
- 🧪 快速 smoke test 验证

## 快速开始

### 安装

```bash
# 安装依赖（如果尚未安装）
pip install sdv

# 安装项目（可编辑模式）
pip install -e .
```

### 生成演示数据

```bash
# 生成包含混合类型的演示数据（默认 1000 行，输出到 data/demo.csv）
python scripts/make_demo_data.py

# 自定义参数
python scripts/make_demo_data.py --rows 500 --output data/my_demo.csv
```

演示数据包含以下列：
- `age`: 连续数值（18-80）
- `income`: 连续数值（收入）
- `education`: 分类（小学/初中/高中/本科/硕士/博士）
- `gender`: 分类（男/女）
- `country`: 分类（国家）
- `target`: 二元分类（True/False）

### 训练模型

```bash
python -m synthgen.train
```

默认配置会使用 `configs/data/demo.yaml` 中指定的数据文件（`data/demo.csv`）。

### 生成合成数据

```bash
python -m synthgen.sample
```

### 运行 Smoke Test

Smoke test 会自动完成完整流程：生成演示数据 → 训练模型 → 生成合成数据

```bash
# 使用脚本（推荐）
bash scripts/smoke_test.sh

# 或使用 pytest
pytest tests/test_smoke.py -v
```

## 项目结构

```
TabRisk/
├── configs/              # Hydra 配置文件
│   ├── train.yaml       # 训练配置
│   ├── sample.yaml      # 采样配置
│   ├── model/
│   │   └── ctgan.yaml   # CTGAN 模型配置
│   └── data/
│       └── demo.yaml    # 数据配置
├── src/
│   └── synthgen/        # 主包
│       ├── __init__.py
│       ├── data/        # 数据加载和 schema 推断
│       │   ├── __init__.py
│       │   ├── load.py
│       │   └── schema.py
│       ├── models/      # 模型接口和实现
│       │   ├── __init__.py
│       │   ├── base.py
│       │   └── sdv_ctgan.py
│       ├── train.py     # 训练入口
│       └── sample.py    # 采样入口
├── scripts/
│   ├── make_demo_data.py  # 生成演示数据脚本
│   └── smoke_test.sh      # Smoke test 脚本
├── tests/
│   └── test_smoke.py   # Smoke test 测试
├── data/               # 数据目录（gitignore）
├── outputs/            # 输出目录（gitignore）
├── .gitignore
├── pyproject.toml
└── README.md
```

## 配置说明

### 训练配置 (configs/train.yaml)

- `data`: 数据文件路径和配置
- `model`: 模型类型和参数
- `output_dir`: 输出目录
- `seed`: 随机种子

### 采样配置 (configs/sample.yaml)

- `model_path`: 模型文件路径
- `num_rows`: 生成的行数
- `output_path`: 输出 CSV 路径

## 输出文件

训练完成后，`outputs/` 目录会包含：

- `model.pkl`: 保存的模型
- `schema.json`: 推断的 schema
- `run_config.yaml`: 运行配置
- `train.log`: 训练日志

采样完成后会生成：

- `synthetic.csv`: 合成的数据文件

## 开发

```bash
# 运行测试
pytest

# 代码格式化（如果安装了 black）
black src/

# 代码检查（如果安装了 ruff）
ruff check src/
```

## 许可证

MIT
