"""Tests for the decoupled evaluate subcommand."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

from synthgen.data.schema import Schema, infer_schema
from synthgen.evaluation.runner import (
    _ensure_run_dir_files,
    _infer_target_and_task,
    _resolve_run_dir,
    run_evaluation,
)


@pytest.fixture
def minimal_run_dir(tmp_path):
    """创建最小的 run_dir：schema.json, original.csv, synthetic.csv, train_config.yaml。"""
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0],
        "b": [4, 5, 6],
        "target": [0, 1, 0],
    })
    schema = infer_schema(df)
    schema.save(str(tmp_path / "schema.json"))
    df.to_csv(tmp_path / "original.csv", index=False)
    df.to_csv(tmp_path / "synthetic.csv", index=False)
    OmegaConf.save(
        config=OmegaConf.create({"data": {"params": {"target": "target"}}}),
        f=tmp_path / "train_config.yaml",
    )
    return tmp_path


def test_resolve_run_dir_requires_run_dir():
    cfg = OmegaConf.create({"run_dir": None})
    with pytest.raises(ValueError, match="必须指定 run_dir"):
        _resolve_run_dir(cfg, Path.cwd())


def test_ensure_run_dir_files_ok(minimal_run_dir):
    _ensure_run_dir_files(minimal_run_dir)


def test_ensure_run_dir_files_missing(tmp_path):
    (tmp_path / "schema.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="缺少文件"):
        _ensure_run_dir_files(tmp_path)


def test_infer_target_from_config(minimal_run_dir):
    schema = Schema.load(str(minimal_run_dir / "schema.json"))
    cfg = OmegaConf.create({"target_col": None, "task_type": None})
    target, task = _infer_target_and_task(
        minimal_run_dir, schema, OmegaConf.load(minimal_run_dir / "train_config.yaml"), cfg
    )
    assert target == "target"
    assert task in ("classification", "regression")


def test_infer_target_from_common_names(minimal_run_dir):
    schema = Schema.load(str(minimal_run_dir / "schema.json"))
    cfg = OmegaConf.create({"target_col": None, "task_type": None})
    # 无 train_config 时从列名推断
    target, task = _infer_target_and_task(minimal_run_dir, schema, None, cfg)
    assert target == "target"
    assert task is not None


def test_run_evaluation_synthcity(minimal_run_dir):
    """运行 evaluator=synthcity，应生成 eval/metrics.json。"""
    cfg = OmegaConf.create({
        "run_dir": str(minimal_run_dir),
        "evaluator": {"name": "synthcity", "stats": ["ks", "wasserstein", "corr"], "utility_metric": "logreg", "seed": 42},
        "target_col": "target",
        "task_type": "classification",
        "output_subdir": "eval",
    })
    run_evaluation(cfg)
    eval_dir = minimal_run_dir / "eval"
    assert eval_dir.exists()
    assert (eval_dir / "metrics.json").exists()
    with open(eval_dir / "metrics.json") as f:
        data = json.load(f)
    assert isinstance(data, dict)
    assert (eval_dir / "eval_config.yaml").exists()


def test_run_evaluation_sdv(minimal_run_dir):
    """运行 evaluator=sdv，需安装 sdmetrics。"""
    pytest.importorskip("sdmetrics")
    cfg = OmegaConf.create({
        "run_dir": str(minimal_run_dir),
        "evaluator": {"name": "sdv", "diagnostic": False, "seed": 42},
        "target_col": None,
        "task_type": None,
        "output_subdir": "eval",
    })
    run_evaluation(cfg)
    eval_dir = minimal_run_dir / "eval"
    assert eval_dir.exists()
    assert (eval_dir / "metrics.json").exists()
    with open(eval_dir / "metrics.json") as f:
        data = json.load(f)
    assert isinstance(data, dict)
    assert "overall_score" in data or "quality_overall" in data or "quality_error" in data
