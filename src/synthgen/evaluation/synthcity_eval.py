"""SynthCity 评估：统计指标（ks, wasserstein, corr）+ 效用（Train on synthetic / Test on real）。"""

from typing import Any, Dict, List, Optional

import pandas as pd

from synthgen.data.schema import Schema

SYNTHCITY_IMPORT_ERROR = (
    "evaluator=synthcity 需要安装 synthcity。请执行: pip install synthcity"
)


def _ensure_synthcity() -> None:
    try:
        import synthcity  # noqa: F401
    except ImportError as e:
        raise ImportError(SYNTHCITY_IMPORT_ERROR) from e


def run_synthcity_eval(
    original_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    schema: Schema,
    target_col: Optional[str] = None,
    task_type: Optional[str] = None,
    evaluator_cfg: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    调用 SynthCity 的统计与效用评估。
    Utility 使用 Train on synthetic / Test on real 的 protocol。
    返回 (metrics_dict, details_dict)。
    """
    _ensure_synthcity()
    evaluator_cfg = evaluator_cfg or {}
    stats_list = evaluator_cfg.get("stats") or ["ks", "wasserstein", "corr"]
    utility_metric = evaluator_cfg.get("utility_metric", "xgb")
    seed = evaluator_cfg.get("seed", 42)

    metrics: Dict[str, Any] = {}
    details: Dict[str, Any] = {}

    # 统计指标
    try:
        from synthcity.metrics import eval_statistical
        # synthcity 常见 API: eval_statistical(X_gt, X_syn) 或 (real_data, synthetic_data)
        res = eval_statistical(original_df, synthetic_df)
        if isinstance(res, dict):
            for k, v in res.items():
                if isinstance(v, (int, float)):
                    metrics[f"stats_{k}"] = v
                else:
                    details[f"stats_{k}"] = v
        elif hasattr(res, "items"):
            for k, v in res.items():
                metrics[f"stats_{k}"] = v if isinstance(v, (int, float)) else str(v)
        else:
            metrics["stats_score"] = float(res) if res is not None else None
    except Exception as e:
        metrics["stats_error"] = str(e)
        details["stats_error"] = str(e)

    # 效用：Train on synthetic / Test on real（优先 synthcity.eval_performance，否则自实现）
    if target_col and task_type and target_col in original_df.columns and target_col in synthetic_df.columns:
        used_synthcity = False
        try:
            from synthcity.metrics import eval_performance
            res = eval_performance(
                original_df,
                synthetic_df,
                task_type=task_type,
                target_col=target_col,
                random_state=seed,
            )
            if isinstance(res, dict):
                for k, v in res.items():
                    metrics[f"utility_{k}"] = v if isinstance(v, (int, float)) else str(v)
                used_synthcity = True
            elif res is not None:
                metrics["utility_score"] = float(res)
                used_synthcity = True
        except (TypeError, Exception):
            pass
        if not used_synthcity:
            _utility_train_synth_test_real(
                original_df, synthetic_df, target_col, task_type,
                utility_metric=utility_metric, seed=seed, metrics=metrics, details=details,
            )

    return metrics, details if details else None


def _utility_train_synth_test_real(
    original_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    target_col: str,
    task_type: str,
    utility_metric: str = "xgb",
    seed: int = 42,
    metrics: Optional[Dict[str, Any]] = None,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Train on synthetic / Test on real：在合成数据上训练模型，在真实数据上评估。
    直接写入 metrics/details。
    """
    metrics = metrics or {}
    details = details or {}
    feature_cols = [c for c in original_df.columns if c != target_col]
    if not feature_cols:
        metrics["utility_error"] = "no feature columns"
        return
    X_syn = synthetic_df[feature_cols]
    y_syn = synthetic_df[target_col]
    X_real = original_df[feature_cols]
    y_real = original_df[target_col]
    # 简单编码：数值列保持，分类型 LabelEncoder
    from sklearn.preprocessing import LabelEncoder
    import numpy as np
    X_syn = X_syn.copy()
    X_real = X_real.copy()
    for col in feature_cols:
        d = X_syn[col].dtype
        if d == object or (hasattr(d, "name") and d.name == "string") or str(d) == "category":
            le = LabelEncoder()
            combined = pd.concat([X_syn[col].astype(str), X_real[col].astype(str)], ignore_index=True)
            le.fit(combined.astype(str))
            X_syn[col] = le.transform(X_syn[col].astype(str))
            X_real[col] = le.transform(X_real[col].astype(str))
    if getattr(y_syn.dtype, "name", str(y_syn.dtype)) in ("object", "string") or str(y_syn.dtype) == "category":
        le_y = LabelEncoder()
        le_y.fit(pd.concat([y_syn.astype(str), y_real.astype(str)], ignore_index=True))
        y_syn = le_y.transform(y_syn.astype(str))
        y_real = le_y.transform(y_real.astype(str))
    X_syn = X_syn.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_real = X_real.replace([np.inf, -np.inf], np.nan).fillna(0)

    try:
        if task_type == "classification":
            from sklearn.metrics import accuracy_score
            if utility_metric == "xgb":
                try:
                    from xgboost import XGBClassifier
                    model = XGBClassifier(random_state=seed, use_label_encoder=False, eval_metric="logloss")
                except ImportError:
                    from sklearn.ensemble import GradientBoostingClassifier
                    model = GradientBoostingClassifier(random_state=seed)
            else:
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(random_state=seed, max_iter=500)
            model.fit(X_syn, y_syn)
            y_pred = model.predict(X_real)
            acc = accuracy_score(y_real, y_pred)
            metrics["utility_accuracy"] = float(acc)
            metrics["utility_score"] = float(acc)
        else:
            from sklearn.metrics import mean_squared_error
            if utility_metric == "xgb":
                try:
                    from xgboost import XGBRegressor
                    model = XGBRegressor(random_state=seed)
                except ImportError:
                    from sklearn.ensemble import GradientBoostingRegressor
                    model = GradientBoostingRegressor(random_state=seed)
            else:
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
            model.fit(X_syn, y_syn)
            y_pred = model.predict(X_real)
            rmse = float(np.sqrt(mean_squared_error(y_real, y_pred)))
            metrics["utility_rmse"] = rmse
            metrics["utility_score"] = -rmse  # 越高越好，用负 RMSE
    except Exception as e:
        metrics["utility_error"] = str(e)
        details["utility_error"] = str(e)
