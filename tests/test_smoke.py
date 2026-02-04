"""Smoke test for the synthetic data generation pipeline."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from synthgen.data.load import load_csv
from synthgen.data.schema import infer_schema, Schema
from synthgen.models.sdv_ctgan import SDVCTGANModel


@pytest.fixture
def demo_data():
    """创建演示数据。"""
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
    return df


@pytest.fixture
def temp_dir():
    """创建临时目录。"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


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
    assert "city" in schema.categorical_columns
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
    """测试模型训练和采样。"""
    # 推断 schema
    schema = infer_schema(demo_data)

    # 创建模型（使用较少的 epochs 以加快测试）
    model = SDVCTGANModel(epochs=10, batch_size=100, verbose=False)

    # 训练
    model.fit(demo_data, schema)

    # 采样
    synthetic = model.sample(num_rows=100)
    assert len(synthetic) == 100
    assert set(synthetic.columns) == set(demo_data.columns)


def test_model_save_load(demo_data, temp_dir):
    """测试模型保存和加载。"""
    schema = infer_schema(demo_data)
    model = SDVCTGANModel(epochs=10, batch_size=100, verbose=False)
    model.fit(demo_data, schema)

    # 保存
    model_path = temp_dir / "model.pkl"
    model.save(str(model_path))
    assert model_path.exists()

    # 加载
    loaded_model = SDVCTGANModel.load(str(model_path))
    assert loaded_model.model is not None

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

    # 4. 训练模型
    model = SDVCTGANModel(epochs=10, batch_size=100, verbose=False)
    model.fit(df, schema)

    # 5. 保存模型
    model_path = temp_dir / "model.pkl"
    model.save(str(model_path))

    # 6. 加载模型
    loaded_model = SDVCTGANModel.load(str(model_path))

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
