#!/usr/bin/env python3
"""
在 synthetic.csv 中统计与给定 target（从 candidate.csv 抽取一条）近似相等的条数。

使用多种近似匹配方法，并在一定精度下输出每种方法匹配到的 synthetic 样本与 target 的对比。

用法示例:
  # 使用 run 目录（内含 synthetic.csv, candidate.csv, schema.json）
  python scripts/match_target_demo.py --run-dir outputs/standard_adult_openml_... --target-row 0

  # 指定文件路径
  python scripts/match_target_demo.py \
    --synthetic outputs/xxx/synthetic.csv \
    --candidate outputs/xxx/candidate.csv \
    --schema outputs/xxx/schema.json \
    --target-row 0

  # 调节精度
  python scripts/match_target_demo.py --run-dir outputs/xxx --epsilon 0.01 --l2-radius 0.3 --fraction 0.85

  # 从 run 目录下指定文件抽取 target（仅写文件名即可，如 original.csv）
  python scripts/match_target_demo.py --run-dir outputs/xxx --target-file original.csv --target-row 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from synthgen.data.schema import Schema
from synthgen.match import run_all_methods

# 展示匹配行时最多显示条数
MAX_SAMPLE_ROWS = 5


def _load_data(args: argparse.Namespace):
    """根据 run_dir 或显式路径加载 synthetic、target 来源（candidate 或 --target-file）、schema。"""
    run_dir = None
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        synthetic_path = run_dir / "synthetic.csv"
        candidate_path = run_dir / "candidate.csv"
        schema_path = run_dir / "schema.json"
    else:
        synthetic_path = Path(args.synthetic)
        candidate_path = Path(args.candidate) if args.candidate else None
        schema_path = Path(args.schema) if args.schema else None
        if not synthetic_path.is_absolute():
            synthetic_path = PROJECT_ROOT / synthetic_path
        if candidate_path and not candidate_path.is_absolute():
            candidate_path = PROJECT_ROOT / candidate_path
        if schema_path and not schema_path.is_absolute():
            schema_path = PROJECT_ROOT / schema_path

    if not synthetic_path.exists():
        raise FileNotFoundError(f"合成数据不存在: {synthetic_path}")
    synthetic = pd.read_csv(synthetic_path)

    # target 来源：--target-file 指定则从该文件取，否则从 candidate 取
    if getattr(args, "target_file", None):
        target_path = Path(args.target_file)
        if run_dir is not None:
            # 指定了 run-dir 时，target-file 视为 run 目录下的文件名（如 original.csv）
            target_path = run_dir / target_path
        elif not target_path.is_absolute():
            target_path = PROJECT_ROOT / target_path
        if not target_path.exists():
            raise FileNotFoundError(f"Target 文件不存在: {target_path}")
        target_source = pd.read_csv(target_path)
        target_source_name = str(target_path)
    else:
        if not candidate_path or not candidate_path.exists():
            raise FileNotFoundError(
                "未指定 --target-file 且 candidate 不存在或未指定（使用 --synthetic 时需 --candidate 或 --target-file）"
            )
        target_source = pd.read_csv(candidate_path)
        target_source_name = str(candidate_path)

    schema = None
    if schema_path and schema_path.exists():
        schema = Schema.load(str(schema_path))

    return synthetic, target_source, target_source_name, schema


def _feature_df(df: pd.DataFrame) -> pd.DataFrame:
    """排除 is_member 等列，便于展示。"""
    drop = [c for c in df.columns if c == "is_member"]
    if drop:
        return df.drop(columns=drop)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="在 synthetic 中统计与 target（来自 candidate 的一行）近似相等的条数，多种方法对比。"
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-dir", type=str, help="run 目录，内含 synthetic.csv, candidate.csv, schema.json")
    g.add_argument("--synthetic", type=str, help="synthetic.csv 路径（需同时指定 --candidate）")
    parser.add_argument("--candidate", type=str, help="candidate.csv 路径（与 --synthetic 一起使用）")
    parser.add_argument("--schema", type=str, default=None, help="schema.json 路径（可选，不指定则从数据推断）")
    parser.add_argument("--target-row", type=int, default=0, help="从 candidate 或 --target-file 中取第几行作为 target（0-based）")
    parser.add_argument("--target-file", type=str, default=None, help="从该 CSV 抽取 target；与 --run-dir 同用时只需写 run 目录下文件名，如 original.csv")
    parser.add_argument("--epsilon", type=float, default=1e-6, help="连续列绝对容差（方法一、三）")
    parser.add_argument("--rel-epsilon", type=float, default=None, help="连续列相对容差（方法一，可选）")
    parser.add_argument("--l2-radius", type=float, default=0.5, help="L2 球半径（方法二，归一化后）")
    parser.add_argument("--fraction", type=float, default=0.9, help="最少匹配列比例（方法三）")
    parser.add_argument("--n-bins", type=int, default=10, help="连续列分箱数（方法四）")
    parser.add_argument("--max-sample-rows", type=int, default=MAX_SAMPLE_ROWS, help="每种方法最多展示的 synthetic 行数")
    parser.add_argument("--output-dir", type=str, default=None, help="将每种方法的 target + 匹配样本写入该目录")
    args = parser.parse_args()

    if args.synthetic and not args.candidate and not args.target_file:
        parser.error("使用 --synthetic 时需指定 --candidate 或 --target-file 之一作为 target 来源")

    synthetic, target_source, target_source_name, schema = _load_data(args)
    target_idx = args.target_row
    if target_idx < 0 or target_idx >= len(target_source):
        print(f"错误: target-row={target_idx} 超出 target 来源行范围 [0, {len(target_source)-1}]")
        sys.exit(1)

    target_row = target_source.iloc[target_idx : target_idx + 1]
    target_display = _feature_df(target_row)

    print("=" * 60)
    print("匹配条件（精度）")
    print("=" * 60)
    print(f"  epsilon (绝对): {args.epsilon}")
    print(f"  rel_epsilon:    {args.rel_epsilon}")
    print(f"  l2_radius:      {args.l2_radius}")
    print(f"  fraction:       {args.fraction}")
    print(f"  n_bins:         {args.n_bins}")
    print()
    print("Target 行（来自 {} 第 {} 行）:".format(target_source_name, target_idx))
    print(target_display.to_string(index=False))
    print()

    results = run_all_methods(
        synthetic,
        target_row.iloc[0],
        schema=schema,
        epsilon=args.epsilon,
        rel_epsilon=args.rel_epsilon,
        l2_radius=args.l2_radius,
        fraction=args.fraction,
        n_bins=args.n_bins,
    )

    method_descriptions = {
        "epsilon": "方法一：分类型精确 + 连续型在 epsilon 内",
        "l2_ball": "方法二：分类型精确 + 连续列归一化 L2 距离 ≤ radius",
        "fraction_match": "方法三：至少 fraction 比例的列在容差内匹配",
        "binned_exact": "方法四：连续列分箱后整行精确匹配",
    }

    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        target_display.to_csv(output_dir / "target.csv", index=False)

    for method_name, (mask, matched_df) in results.items():
        count = int(mask.sum())
        desc = method_descriptions.get(method_name, method_name)
        print("=" * 60)
        print(f"{method_name}: {desc}")
        print("=" * 60)
        print(f"  匹配条数: {count} / {len(synthetic)}")
        sample = _feature_df(matched_df.head(args.max_sample_rows))
        if len(sample) > 0:
            print("  匹配到的 synthetic 样本（前 {} 条）:".format(len(sample)))
            print(sample.to_string(index=False))
        else:
            print("  匹配到的 synthetic 样本: （无）")
        print()

        if output_dir is not None:
            out_path = output_dir / f"matched_{method_name}.csv"
            _feature_df(matched_df).to_csv(out_path, index=False)
            print(f"  已写入: {out_path}")

    if output_dir:
        print()
        print(f"Target 已写入: {output_dir / 'target.csv'}")
        print(f"各方法匹配结果已写入: {output_dir / 'matched_<method>.csv'}")


if __name__ == "__main__":
    main()
