#!/usr/bin/env python3
"""
先跑 shadow_pair_metrics，再从其输出目录读取并跑 shadow_delta_analysis，
最后打印所有保存文件的路径与说明。

用法（在项目根目录）：
  python scripts/run_shadow_metrics_pipeline.py --run-dir outputs/train_xxx_candidate100_control_20260218_123456
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# 输出文件说明（与 run_commands.txt 一致，便于维护）
OUTPUT_DESCRIPTIONS = [
    # shadow_pair_metrics.py
    ("pair_metrics.csv", "shadow_pair_metrics", "每个 (target_idx, round_k) 的 in/out 对：MMD、列统计、target 最近邻 delta 等"),
    ("summary_by_target.csv", "shadow_pair_metrics", "按 target 聚合的 in/out 指标（均值/方差等）"),
    ("pair_metrics_control.csv", "shadow_pair_metrics", "（有 control 时）每个 round 的 control_in/control_out 对同样指标"),
    ("summary_by_target_control.csv", "shadow_pair_metrics", "（有 control 时）按 target 聚合的 control 指标"),
    # shadow_delta_analysis.py
    ("global_delta_stats.json", "shadow_delta_analysis", "delta_min_dist 总体统计（均值、置信区间、t/Wilcoxon 检验、Cohen's d）"),
    ("delta_distribution.png", "shadow_delta_analysis", "delta 分布直方图"),
    ("control_comparison.json", "shadow_delta_analysis", "（有 control 时）target delta vs control delta 成对对照（均值、t 检验等）"),
    ("mmd_component_analysis.json", "shadow_delta_analysis", "numeric-only vs mixed MMD 相关性等"),
    ("outlier_targets.json", "shadow_delta_analysis", "异常 target 及 round 级 delta 列表"),
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="依次执行 shadow_pair_metrics 与 shadow_delta_analysis，并输出保存文件说明与位置。",
    )
    parser.add_argument("--run-dir", type=str, required=True, help="run 目录（含 shadow/target_*/、candidate.csv 等）")
    parser.add_argument("--n-jobs", type=int, default=32, help="pair_metrics 并行进程数（默认 32）")
    parser.add_argument("--mmd-max-rows", type=int, default=500, help="pair_metrics 的 MMD 每侧最大行数（默认 500）")
    parser.add_argument("--max-targets", type=int, default=None, help="pair_metrics 仅统计前 N 个 target（默认全量）")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    if not run_dir.exists():
        print(f"错误: run_dir 不存在: {run_dir}", file=sys.stderr)
        return 1

    out_dir = run_dir / "shadow_pair_metrics"
    metrics_script = PROJECT_ROOT / "scripts" / "shadow_pair_metrics.py"
    delta_script = PROJECT_ROOT / "scripts" / "shadow_delta_analysis.py"

    if not metrics_script.exists() or not delta_script.exists():
        print(f"错误: 依赖脚本不存在（需 {metrics_script} 与 {delta_script}）", file=sys.stderr)
        return 1

    # Step 1: shadow_pair_metrics
    cmd1 = [
        sys.executable,
        str(metrics_script),
        "--run-dir", str(run_dir),
        "--n-jobs", str(args.n_jobs),
        "--mmd-max-rows", str(args.mmd_max_rows),
    ]
    if args.max_targets is not None:
        cmd1.extend(["--max-targets", str(args.max_targets)])
    print("Step 1: 运行 shadow_pair_metrics ...")
    ret1 = subprocess.run(cmd1, cwd=str(PROJECT_ROOT))
    if ret1.returncode != 0:
        print(f"shadow_pair_metrics 退出码 {ret1.returncode}，终止流程。", file=sys.stderr)
        return ret1.returncode

    # Step 2: shadow_delta_analysis（从 Step 1 的输出目录读取）
    cmd2 = [sys.executable, str(delta_script), "--run-dir", str(run_dir)]
    print("Step 2: 运行 shadow_delta_analysis（读取 shadow_pair_metrics 输出）...")
    ret2 = subprocess.run(cmd2, cwd=str(PROJECT_ROOT))
    if ret2.returncode != 0:
        print(f"shadow_delta_analysis 退出码 {ret2.returncode}，终止流程。", file=sys.stderr)
        return ret2.returncode

    # 输出保存文件说明与位置
    print("\n" + "=" * 60)
    print("保存文件说明与位置（均在 run_dir/shadow_pair_metrics/ 下）")
    print("=" * 60)
    print(f"输出目录: {out_dir.resolve()}\n")
    for filename, source, desc in OUTPUT_DESCRIPTIONS:
        path = out_dir / filename
        exists = "存在" if path.exists() else "未生成"
        print(f"  {filename}")
        print(f"    来源: {source}")
        print(f"    说明: {desc}")
        print(f"    路径: {path.resolve()}  [{exists}]")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
