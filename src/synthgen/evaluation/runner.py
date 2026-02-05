"""评估流程：校验 run_dir、加载数据、推断 target/task_type、调用评估器、保存结果。"""

from pathlib import Path
from typing import Any, Optional

# 项目根目录（src/synthgen/evaluation/runner.py -> parents[3]）
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from synthgen.data.schema import Schema


# 常见目标列名（用于推断）
COMMON_TARGET_NAMES = ("target", "label", "class", "outcome", "y", "income")


def _resolve_run_dir(cfg: DictConfig, project_root: Path) -> Path:
    """将 run_dir 解析为绝对路径。"""
    run_dir = cfg.get("run_dir")
    if run_dir is None or str(run_dir).strip() in ("", "null", "None"):
        raise ValueError("必须指定 run_dir，例如: run_dir=outputs/smoke_iris")
    path = Path(run_dir)
    if not path.is_absolute():
        # 先相对于 cwd，再相对于项目根
        if path.exists():
            return path.resolve()
        path = project_root / path
    return path.resolve()


def _ensure_run_dir_files(run_dir: Path) -> None:
    """校验 run_dir 内必要文件存在。"""
    required = ["schema.json", "original.csv", "synthetic.csv"]
    missing = [f for f in required if not (run_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"run_dir 缺少文件: {missing}。需要: {required}。"
            f"train_config.yaml 可选，用于推断 target_col / task_type。"
        )


def _load_train_config(run_dir: Path) -> Optional[DictConfig]:
    """若存在则加载 train_config.yaml。"""
    path = run_dir / "train_config.yaml"
    if not path.exists():
        return None
    return OmegaConf.load(path)


def _infer_target_and_task(
    run_dir: Path,
    schema: Schema,
    train_config: Optional[DictConfig],
    cfg: DictConfig,
) -> tuple[Optional[str], Optional[str]]:
    """
    推断 target_col 与 task_type。
    优先级：1) 用户传入 2) train_config.data.params.target 3) schema + 常见列名 4) 失败则要求用户传 target_col。
    """
    target_col = cfg.get("target_col")
    if target_col is not None and str(target_col).strip() and str(target_col).lower() not in ("null", "none"):
        target_col = str(target_col).strip()
    else:
        target_col = None

    task_type = cfg.get("task_type")
    if task_type is not None and str(task_type).strip() and str(task_type).lower() not in ("null", "none"):
        task_type = str(task_type).strip().lower()
        if task_type not in ("classification", "regression"):
            task_type = None
    else:
        task_type = None

    # 从 train_config 取 target
    if target_col is None and train_config is not None:
        try:
            t = OmegaConf.select(train_config, "data.params.target", default=None)
            if t is not None and str(t).strip() and str(t).lower() not in ("null", "none"):
                target_col = str(t).strip()
        except Exception:
            pass

    # 从 schema 列名推断 target
    if target_col is None:
        col_names = [c.name for c in schema.columns]
        for name in COMMON_TARGET_NAMES:
            if name in col_names:
                target_col = name
                break
        if target_col is None and len(col_names) == 1:
            target_col = col_names[0]

    # 推断 task_type
    if task_type is None and target_col is not None:
        task_type = "classification" if target_col in schema.categorical_columns else "regression"

    return target_col, task_type


def _require_target_for_evaluator(evaluator_name: str, target_col: Optional[str], task_type: Optional[str]) -> None:
    """synthcity 的 utility 评估需要 target_col 与 task_type；缺失时提示用户。"""
    if evaluator_name != "synthcity":
        return
    if not target_col or not task_type:
        raise ValueError(
            "evaluator=synthcity 的 utility 评估需要目标列与任务类型。"
            "请指定 target_col=... 和 task_type=classification|regression，"
            "或确保 run_dir 内 train_config.yaml 的 data.params.target 已设置且可推断 task_type。"
        )


def run_evaluation(cfg: DictConfig) -> None:
    """
    主流程：校验 run_dir、读 original/synthetic、可选读 schema/train_config、
    推断 target_col/task_type、按 evaluator 调用 synthcity 或 sdv、写入 run_dir/eval/。
    """
    run_dir = _resolve_run_dir(cfg, _PROJECT_ROOT)
    _ensure_run_dir_files(run_dir)

    original_path = run_dir / "original.csv"
    synthetic_path = run_dir / "synthetic.csv"
    schema_path = run_dir / "schema.json"

    original_df = pd.read_csv(original_path)
    synthetic_df = pd.read_csv(synthetic_path)
    schema = Schema.load(str(schema_path))
    train_config = _load_train_config(run_dir)

    target_col, task_type = _infer_target_and_task(run_dir, schema, train_config, cfg)
    evaluator_name = OmegaConf.select(cfg, "evaluator.name", default=None)
    if not evaluator_name:
        evaluator_name = "synthcity"
    evaluator_name = str(evaluator_name).strip().lower()

    _require_target_for_evaluator(evaluator_name, target_col, task_type)

    evaluator_cfg = OmegaConf.to_container(cfg.get("evaluator") or {}, resolve=True) or {}
    if evaluator_name == "synthcity":
        from synthgen.evaluation.synthcity_eval import run_synthcity_eval
        metrics_dict, details = run_synthcity_eval(
            original_df, synthetic_df, schema,
            target_col=target_col, task_type=task_type, evaluator_cfg=evaluator_cfg,
        )
    elif evaluator_name == "sdv":
        from synthgen.evaluation.sdv_eval import run_sdv_eval
        metrics_dict, details = run_sdv_eval(
            original_df, synthetic_df, schema,
            evaluator_cfg=evaluator_cfg,
        )
    else:
        raise ValueError(f"不支持的 evaluator: {evaluator_name}。可选: synthcity | sdv")

    # 输出目录
    output_subdir = cfg.get("output_subdir", "eval")
    eval_dir = run_dir / output_subdir
    eval_dir.mkdir(parents=True, exist_ok=True)

    # 可序列化：把 non-float 转成可 JSON 的类型
    def _serialize(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_serialize(x) for x in obj]
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        if hasattr(obj, "item"):
            return obj.item()
        return str(obj)

    metrics_serializable = _serialize(metrics_dict)
    if details:
        metrics_serializable["_details"] = _serialize(details)

    import json
    metrics_json_path = eval_dir / "metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_serializable, f, ensure_ascii=False, indent=2)

    # 可选：metrics.csv（每行一个 metric name, value）
    metrics_csv_path = eval_dir / "metrics.csv"
    rows = []
    for k, v in metrics_dict.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            rows.append({"metric": k, "value": v})
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, (int, float, str, bool)) or v2 is None:
                    rows.append({"metric": f"{k}.{k2}", "value": v2})
    if rows:
        pd.DataFrame(rows).to_csv(metrics_csv_path, index=False)

    # 保存评估配置便于复现
    eval_config_path = eval_dir / "eval_config.yaml"
    OmegaConf.save(config=cfg, f=eval_config_path)

    from loguru import logger
    logger.info(f"评估完成。metrics.json: {metrics_json_path}")
    logger.info(f"eval_config.yaml: {eval_config_path}")
    if rows:
        logger.info(f"metrics.csv: {metrics_csv_path}")
