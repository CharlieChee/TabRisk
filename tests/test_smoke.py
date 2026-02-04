"""Smoke test for the synthetic data generation pipeline."""

import os
import subprocess
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from synthgen.data.load import load_csv
from synthgen.data.schema import infer_schema, Schema
from synthgen.models import SynthCityCTGANModel, load_model
from synthgen.synthetic_backend import create_plugin


@pytest.fixture
def demo_data():
    """创建演示数据（与 make_demo_data.py 一致）。"""
    np.random.seed(42)
    n_rows = 500

    # 与 make_demo_data.py 保持一致的数据结构
    age = np.random.randint(18, 81, n_rows)
    income = np.random.normal(50000, 20000, n_rows)
    income = np.clip(income, 0, 200000)
    education_levels = ["小学", "初中", "高中", "本科", "硕士", "博士"]
    education = np.random.choice(education_levels, n_rows, p=[0.1, 0.15, 0.25, 0.3, 0.15, 0.05])
    gender = np.random.choice(["男", "女"], n_rows, p=[0.52, 0.48])
    countries = ["中国", "美国", "日本", "德国", "法国", "英国", "其他"]
    country = np.random.choice(countries, n_rows, p=[0.4, 0.2, 0.1, 0.1, 0.05, 0.05, 0.1])
    target_probs = 1 / (1 + np.exp(-(income / 10000 - 3 + (pd.Categorical(education).codes / 2))))
    target = np.random.binomial(1, target_probs, n_rows).astype(bool)

    data = {
        "age": age,
        "income": income.round(2),
        "education": education,
        "gender": gender,
        "country": country,
        "target": target,
    }

    df = pd.DataFrame(data)
    return df


@pytest.fixture
def temp_dir():
    """创建临时目录。"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


def test_onehotencoder_compat():
    """OneHotEncoder(sparse=...) 在 sklearn>=1.2 下应通过 compat 层正常创建。"""
    pytest.importorskip("sklearn")
    from synthgen.compat.sklearn import apply_sklearn_compat

    apply_sklearn_compat()
    import sklearn.preprocessing

    # 模拟 synthcity 旧 API：OneHotEncoder(sparse=True)
    enc = sklearn.preprocessing.OneHotEncoder(sparse=True)
    assert enc is not None


def test_create_ctgan_plugin_with_verbose():
    """创建 ctgan plugin 时传入 verbose 不应触发 TypeError。"""
    plugin = create_plugin(
        "ctgan",
        random_state=42,
        n_iter=2,
        batch_size=32,
        verbose=True,  # 当前 SynthCity 不支持，应被过滤
    )
    assert plugin is not None


def test_data_loading(demo_data, temp_dir):
    """测试数据加载。"""
    # 保存演示数据
    csv_path = temp_dir / "demo.csv"
    demo_data.to_csv(csv_path, index=False)

    # 加载数据
    df = load_csv(str(csv_path))
    assert len(df) > 0
    assert len(df.columns) > 0


def test_schema_inference(demo_data):
    """测试 schema 推断。"""
    schema = infer_schema(demo_data)

    assert schema.total_rows == len(demo_data)
    assert schema.total_columns == len(demo_data.columns)
    assert len(schema.categorical_columns) > 0
    assert len(schema.continuous_columns) > 0

    # 验证列类型
    assert "education" in schema.categorical_columns
    assert "gender" in schema.categorical_columns
    assert "income" in schema.continuous_columns or "income" in schema.categorical_columns


def test_schema_save_load(demo_data, temp_dir):
    """测试 schema 保存和加载。"""
    schema = infer_schema(demo_data)
    schema_path = temp_dir / "schema.json"

    schema.save(str(schema_path))
    assert schema_path.exists()

    loaded_schema = Schema.load(str(schema_path))
    assert loaded_schema.total_rows == schema.total_rows
    assert loaded_schema.total_columns == schema.total_columns


def test_model_training_and_sampling(demo_data, temp_dir):
    """测试模型训练和采样（SynthCity CTGAN）。"""
    # 推断 schema
    schema = infer_schema(demo_data)

    # 创建模型（使用较少的 n_iter 以加快测试）
    model = SynthCityCTGANModel(n_iter=10, batch_size=100, verbose=False, random_state=42)

    # 训练
    model.fit(demo_data, schema)

    # 采样
    synthetic = model.sample(num_rows=100)
    assert len(synthetic) == 100
    assert set(synthetic.columns) == set(demo_data.columns)


def test_model_save_load(demo_data, temp_dir):
    """测试模型保存和加载（SynthCity 后端）。"""
    schema = infer_schema(demo_data)
    model = SynthCityCTGANModel(n_iter=10, batch_size=100, verbose=False, random_state=42)
    model.fit(demo_data, schema)

    # 保存
    model_path = temp_dir / "model.pkl"
    model.save(str(model_path))
    assert model_path.exists()

    # 通过统一 load_model 加载
    loaded_model = load_model(str(model_path))
    assert loaded_model._plugin is not None

    # 验证可以采样
    synthetic = loaded_model.sample(num_rows=50)
    assert len(synthetic) == 50


def test_end_to_end_pipeline(demo_data, temp_dir):
    """端到端测试：完整流程。"""
    # 1. 保存数据
    csv_path = temp_dir / "demo.csv"
    demo_data.to_csv(csv_path, index=False)

    # 2. 加载数据
    df = load_csv(str(csv_path))
    assert len(df) > 0

    # 3. 推断 schema
    schema = infer_schema(df)
    schema_path = temp_dir / "schema.json"
    schema.save(str(schema_path))

    # 4. 训练模型（SynthCity CTGAN）
    model = SynthCityCTGANModel(n_iter=10, batch_size=100, verbose=False, random_state=42)
    model.fit(df, schema)

    # 5. 保存模型
    model_path = temp_dir / "model.pkl"
    model.save(str(model_path))

    # 6. 通过统一 load_model 加载
    loaded_model = load_model(str(model_path))

    # 7. 生成合成数据
    synthetic = loaded_model.sample(num_rows=200)
    assert len(synthetic) == 200

    # 8. 保存合成数据
    synthetic_path = temp_dir / "synthetic.csv"
    synthetic.to_csv(synthetic_path, index=False)
    assert synthetic_path.exists()

    # 验证合成数据可以加载
    loaded_synthetic = pd.read_csv(synthetic_path)
    assert len(loaded_synthetic) == 200
    assert set(loaded_synthetic.columns) == set(df.columns)


def test_train_with_data_path_override(demo_data, temp_dir):
    """测试通过 +data.path=... 覆盖数据路径时，struct mode 下不会触发 ConfigTypeError。"""
    csv_path = temp_dir / "demo.csv"
    demo_data.to_csv(csv_path, index=False)

    project_root = Path(__file__).parent.parent
    env = {**os.environ, "PYTHONPATH": str(project_root / "src")}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "synthgen.train",
            f"+data.path={csv_path.resolve()}",
            "epochs=2",
        ],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"train 失败 (exit={result.returncode})\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
