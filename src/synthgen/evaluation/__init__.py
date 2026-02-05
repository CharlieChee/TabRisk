"""Utility evaluation 子包：synthcity / sdmetrics 评估，完全解耦于 train。"""

from synthgen.evaluation.runner import run_evaluation

__all__ = ["run_evaluation"]
