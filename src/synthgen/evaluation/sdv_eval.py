"""SDMetrics 评估：QualityReport + DiagnosticReport，返回可序列化的 overall 与 breakdown。"""

from typing import Any, Dict, Optional

import pandas as pd

from synthgen.data.schema import Schema

SDV_IMPORT_ERROR = (
    "evaluator=sdv 需要安装 sdmetrics。请执行: pip install sdmetrics"
)


def _ensure_sdmetrics() -> None:
    try:
        import sdmetrics  # noqa: F401
    except ImportError as e:
        raise ImportError(SDV_IMPORT_ERROR) from e


def _schema_to_sdv_metadata(schema: Schema) -> Dict[str, Any]:
    """将项目 Schema 转为 SDMetrics/SDV 单表 metadata 字典。"""
    columns = {}
    for col in schema.columns:
        columns[col.name] = {
            "type": "categorical" if col.is_categorical else "numerical",
        }
    # 单表：仅返回表级 columns，供 QualityReport 使用
    return {"columns": columns}


def run_sdv_eval(
    original_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    schema: Schema,
    evaluator_cfg: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    调用 SDMetrics QualityReport（及可选 DiagnosticReport），生成 overall score 与 per-metric breakdown。
    返回 (metrics_dict, details_dict)，均可序列化。
    """
    _ensure_sdmetrics()
    evaluator_cfg = evaluator_cfg or {}
    run_diagnostic = evaluator_cfg.get("diagnostic", True)

    metadata = _schema_to_sdv_metadata(schema)
    table_meta = metadata

    metrics: Dict[str, Any] = {}
    details: Dict[str, Any] = {}

    try:
        from sdmetrics.reports.single_table import QualityReport
        report = QualityReport()
        try:
            report.generate(original_df, synthetic_df, metadata=table_meta)
        except TypeError:
            report.generate(original_df, synthetic_df, table_meta)
        except Exception as gen_e:
            try:
                from sdv.metadata import SingleTableMetadata
                meta = SingleTableMetadata()
                meta.detect_from_dataframe(original_df)
                table_meta = meta.to_dict()
                report = QualityReport()
                report.generate(original_df, synthetic_df, metadata=table_meta)
            except Exception:
                raise gen_e
        try:
            overall = report.get_score()
            metrics["quality_overall"] = float(overall)
        except Exception:
            pass
        try:
            details_df = report.get_details()
            if details_df is not None and not details_df.empty:
                details["quality_details"] = details_df.to_dict(orient="records")
                for _, row in details_df.iterrows():
                    metric_name = row.get("Metric", row.get("metric", str(_)))
                    value = row.get("Score", row.get("value", row.get("score")))
                    if isinstance(value, (int, float)) and metric_name:
                        metrics[f"quality_{metric_name}"] = float(value)
        except Exception:
            pass
    except Exception as e:
        metrics["quality_error"] = str(e)
        details["quality_error"] = str(e)

    if run_diagnostic:
        try:
            from sdmetrics.reports.single_table import DiagnosticReport
            diag = DiagnosticReport()
            diag.generate(original_df, synthetic_df, metadata=table_meta)
            try:
                overall_d = diag.get_score()
                metrics["diagnostic_overall"] = float(overall_d)
            except Exception:
                pass
            try:
                details_d = diag.get_details()
                if details_d is not None and not details_d.empty:
                    details["diagnostic_details"] = details_d.to_dict(orient="records")
            except Exception:
                pass
        except Exception as e:
            metrics["diagnostic_error"] = str(e)
            details["diagnostic_error"] = str(e)

    if "quality_overall" not in metrics and "quality_error" in metrics:
        metrics["overall_score"] = None
    elif "quality_overall" in metrics:
        metrics["overall_score"] = metrics["quality_overall"]

    return metrics, details if details else None
