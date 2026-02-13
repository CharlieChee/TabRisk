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

  # 不指定 --target-row 时，遍历 target 文件每一行并汇总（多进程并行，默认 16 进程）
  python scripts/match_target_demo.py --run-dir outputs/xxx --target-file original.csv
  # 参数扫描，找合适容差（仅用前 100 行 target 加速）
  python scripts/match_target_demo.py --run-dir outputs/xxx --target-file original.csv --sweep
  python scripts/match_target_demo.py --run-dir outputs/xxx --target-file original.csv --sweep --sweep-sample 200 --output-dir outputs/xxx
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

# 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

from synthgen.data.schema import Schema
from synthgen.match import run_all_methods

# 展示匹配行时最多显示条数
MAX_SAMPLE_ROWS = 5
DEFAULT_N_JOBS = 16


def _match_chunk_worker(
    chunk: tuple[
        list[int],
        pd.DataFrame,
        pd.DataFrame,
        object,
        float,
        object,
        float,
        float,
        int,
    ],
) -> dict[str, tuple[set[int], list[tuple[int, int]]]]:
    """处理一批 target 行，返回各方法的匹配索引集合与每行匹配数。供多进程调用。"""
    (
        indices,
        synthetic,
        target_chunk,
        schema,
        epsilon,
        rel_epsilon,
        l2_radius,
        fraction,
        n_bins,
    ) = chunk
    union = defaultdict(set)
    per_target = defaultdict(list)
    for k, target_idx in enumerate(indices):
        target_row = target_chunk.iloc[k : k + 1]
        results = run_all_methods(
            synthetic,
            target_row.iloc[0],
            schema=schema,
            epsilon=epsilon,
            rel_epsilon=rel_epsilon,
            l2_radius=l2_radius,
            fraction=fraction,
            n_bins=n_bins,
        )
        for method_name, (mask, _) in results.items():
            idx_set = set(np.where(mask)[0])
            union[method_name].update(idx_set)
            per_target[method_name].append((target_idx, len(idx_set)))
    return {m: (union[m], per_target[m]) for m in union}


def _sweep_worker(task):
    """Sweep 单组参数：对一批 target 做匹配，返回该参数下的各方法匹配数。供多进程调用。"""
    (
        param_name,
        param_value,
        epsilon,
        l2_radius,
        fraction,
        n_bins,
        synthetic,
        target_chunk,
        target_indices,
        schema,
        rel_epsilon,
    ) = task
    union = defaultdict(set)
    for k in range(len(target_indices)):
        target_row = target_chunk.iloc[k : k + 1]
        results = run_all_methods(
            synthetic,
            target_row.iloc[0],
            schema=schema,
            epsilon=epsilon,
            rel_epsilon=rel_epsilon,
            l2_radius=l2_radius,
            fraction=fraction,
            n_bins=n_bins,
        )
        for method_name, (mask, _) in results.items():
            union[method_name].update(set(np.where(mask)[0]))
    return {
        "param_name": param_name,
        "param_value": param_value,
        "epsilon": epsilon,
        "l2_radius": l2_radius,
        "fraction": fraction,
        "n_bins": n_bins,
        "count_epsilon": len(union["epsilon"]),
        "count_l2_ball": len(union["l2_ball"]),
        "count_fraction_match": len(union["fraction_match"]),
        "count_binned_exact": len(union["binned_exact"]),
    }


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


# 默认宽松参数（sweep 时作为未扫描维度的取值）
SWEEP_DEFAULT = {"epsilon": 0.01, "l2_radius": 1.0, "fraction": 0.8, "n_bins": 10}
SWEEP_GRID = {
    "epsilon": [1e-6, 1e-4, 0.01, 0.1, 1.0],
    "l2_radius": [0.2, 0.5, 1.0, 2.0, 5.0],
    "fraction": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "n_bins": [3, 5, 10, 20, 50],
}


def _run_sweep(args, synthetic, target_source, target_source_name, schema):
    """参数扫描：多组参数下跑匹配（多进程并行），输出各方法匹配数表格。"""
    n_sample = min(args.sweep_sample, len(target_source))
    target_indices = list(range(n_sample))
    target_chunk = target_source.iloc[target_indices].reset_index(drop=True)
    n_jobs = max(1, args.n_jobs)
    print("=" * 60)
    print("参数 Sweep（target 前 {} 行，并行进程数 {}）".format(n_sample, n_jobs))
    print("=" * 60)

    tasks = []
    if getattr(args, "sweep_full_grid", False):
        # 全网格：5*5*6*5 = 750 组
        for eps in SWEEP_GRID["epsilon"]:
            for l2 in SWEEP_GRID["l2_radius"]:
                for frac in SWEEP_GRID["fraction"]:
                    for nb in SWEEP_GRID["n_bins"]:
                        tasks.append(
                            (
                                "grid",
                                None,
                                eps,
                                l2,
                                frac,
                                nb,
                                synthetic,
                                target_chunk,
                                target_indices,
                                schema,
                                getattr(args, "rel_epsilon", None),
                            )
                        )
        print("  全网格: {} 组（epsilon×l2_radius×fraction×n_bins）".format(len(tasks)))
    elif getattr(args, "sweep_fraction_epsilon", False):
        # 仅 fraction × epsilon 二维网格（精调唯一有匹配的维度）
        l2, nb = SWEEP_DEFAULT["l2_radius"], SWEEP_DEFAULT["n_bins"]
        for frac in SWEEP_GRID["fraction"]:
            for eps in SWEEP_GRID["epsilon"]:
                tasks.append(
                    (
                        "fraction_epsilon",
                        None,
                        eps,
                        l2,
                        frac,
                        nb,
                        synthetic,
                        target_chunk,
                        target_indices,
                        schema,
                        getattr(args, "rel_epsilon", None),
                    )
                )
        print("  fraction×epsilon 网格: {} 组（精调 fraction_match）".format(len(tasks)))
    else:
        # 单参数扫描：5+5+6+5 = 21 组
        for param_name in ["epsilon", "l2_radius", "fraction", "n_bins"]:
            defaults = dict(SWEEP_DEFAULT)
            for param_value in SWEEP_GRID[param_name]:
                defaults[param_name] = param_value
                eps, l2, frac, nb = defaults["epsilon"], defaults["l2_radius"], defaults["fraction"], defaults["n_bins"]
                tasks.append(
                    (
                        param_name,
                        param_value,
                        eps,
                        l2,
                        frac,
                        nb,
                        synthetic,
                        target_chunk,
                        target_indices,
                        schema,
                        getattr(args, "rel_epsilon", None),
                    )
                )
    sys.stdout.flush()

    rows = []
    with Pool(processes=n_jobs) as pool:
        for row in pool.imap_unordered(_sweep_worker, tasks):
            rows.append(row)
    # 排序：全网格 / fraction_epsilon 按 epsilon,fraction；单参数按 param_name, param_value
    if getattr(args, "sweep_full_grid", False):
        rows.sort(key=lambda r: (r["epsilon"], r["l2_radius"], r["fraction"], r["n_bins"]))
    elif getattr(args, "sweep_fraction_epsilon", False):
        rows.sort(key=lambda r: (r["fraction"], r["epsilon"]))
    else:
        rows.sort(key=lambda r: (["epsilon", "l2_radius", "fraction", "n_bins"].index(r["param_name"]), r["param_value"]))
    df = pd.DataFrame(rows)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(df.to_string(index=False))
    print()

    out_path = Path(args.output_dir) / "sweep_results.csv" if args.output_dir else PROJECT_ROOT / "sweep_results.csv"
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print("Sweep 结果已写入: {}".format(out_path))


def main():
    parser = argparse.ArgumentParser(
        description="在 synthetic 中统计与 target（来自 candidate 的一行）近似相等的条数，多种方法对比。"
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-dir", type=str, help="run 目录，内含 synthetic.csv, candidate.csv, schema.json")
    g.add_argument("--synthetic", type=str, help="synthetic.csv 路径（需同时指定 --candidate）")
    parser.add_argument("--candidate", type=str, help="candidate.csv 路径（与 --synthetic 一起使用）")
    parser.add_argument("--schema", type=str, default=None, help="schema.json 路径（可选，不指定则从数据推断）")
    parser.add_argument("--target-row", type=int, default=None, help="取第几行作为 target（0-based）；与 --target-file 同用且不指定时则遍历整个 CSV 并汇总结果")
    parser.add_argument("--target-file", type=str, default=None, help="从该 CSV 抽取 target；与 --run-dir 同用时只需写 run 目录下文件名，如 original.csv")
    parser.add_argument("--epsilon", type=float, default=0.01, help="连续列绝对容差（方法一、三），默认 0.01 较宽松")
    parser.add_argument("--rel-epsilon", type=float, default=None, help="连续列相对容差（方法一，可选）")
    parser.add_argument("--l2-radius", type=float, default=1.0, help="L2 球半径（方法二，归一化后），默认 1.0 较宽松")
    parser.add_argument("--fraction", type=float, default=0.8, help="最少匹配列比例（方法三），默认 0.8")
    parser.add_argument("--n-bins", type=int, default=10, help="连续列分箱数（方法四）")
    parser.add_argument("--max-sample-rows", type=int, default=MAX_SAMPLE_ROWS, help="每种方法最多展示的 synthetic 行数")
    parser.add_argument("--output-dir", type=str, default=None, help="将每种方法的 target + 匹配样本写入该目录")
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS, help="并行进程数：遍历整个 target 或 sweep 时均生效（默认 16）")
    parser.add_argument("--sweep", action="store_true", help="参数扫描：在多种 epsilon/l2_radius/fraction/n_bins 下跑匹配，输出各组合的匹配数（便于找合适参数）")
    parser.add_argument("--sweep-full-grid", action="store_true", help="sweep 时使用全网格（所有参数组合相乘）；不指定则仅单参数扫描（相加）")
    parser.add_argument("--sweep-fraction-epsilon", action="store_true", help="sweep 仅做 fraction×epsilon 二维网格（针对 fraction_match 有效时精调）")
    parser.add_argument("--sweep-sample", type=int, default=100, help="sweep 时仅用前 N 行 target 以加速（默认 100）")
    args = parser.parse_args()

    if args.synthetic and not args.candidate and not args.target_file:
        parser.error("使用 --synthetic 时需指定 --candidate 或 --target-file 之一作为 target 来源")

    synthetic, target_source, target_source_name, schema = _load_data(args)

    if args.sweep:
        _run_sweep(args, synthetic, target_source, target_source_name, schema)
        return

    # 与 --target-file 同用且未指定 --target-row 时，遍历整个 CSV 并汇总
    use_all_rows = args.target_file and args.target_row is None
    if use_all_rows:
        target_indices = list(range(len(target_source)))
    else:
        target_idx = args.target_row if args.target_row is not None else 0
        if target_idx < 0 or target_idx >= len(target_source):
            print(f"错误: target-row={target_idx} 超出 target 来源行范围 [0, {len(target_source)-1}]")
            sys.exit(1)
        target_indices = [target_idx]

    print("=" * 60)
    print("匹配条件（精度）")
    print("=" * 60)
    print(f"  epsilon (绝对): {args.epsilon}")
    print(f"  rel_epsilon:    {args.rel_epsilon}")
    print(f"  l2_radius:      {args.l2_radius}")
    print(f"  fraction:       {args.fraction}")
    print(f"  n_bins:         {args.n_bins}")
    print()

    method_descriptions = {
        "epsilon": "方法一：分类型精确 + 连续型在 epsilon 内",
        "l2_ball": "方法二：分类型精确 + 连续列归一化 L2 距离 ≤ radius",
        "fraction_match": "方法三：至少 fraction 比例的列在容差内匹配",
        "binned_exact": "方法四：连续列分箱后整行精确匹配",
    }

    if use_all_rows:
        # 多进程遍历：每行作为 target，各方法取匹配索引的并集，并记录每行匹配数
        n_jobs = max(1, args.n_jobs)
        chunk_size = max(1, (len(target_indices) + n_jobs - 1) // n_jobs)
        chunks = []
        for i in range(0, len(target_indices), chunk_size):
            idx_chunk = target_indices[i : i + chunk_size]
            target_chunk = target_source.iloc[idx_chunk].reset_index(drop=True)
            chunks.append(
                (
                    idx_chunk,
                    synthetic,
                    target_chunk,
                    schema,
                    args.epsilon,
                    args.rel_epsilon,
                    args.l2_radius,
                    args.fraction,
                    args.n_bins,
                )
            )
        union_matched = defaultdict(set)
        per_target_counts = defaultdict(list)
        print("Target 来源: {}（共 {} 行，并行进程数 {}）".format(target_source_name, len(target_indices), n_jobs))
        with Pool(processes=n_jobs) as pool:
            for part in pool.imap_unordered(_match_chunk_worker, chunks):
                for method_name, (idx_set, counts) in part.items():
                    union_matched[method_name].update(idx_set)
                    per_target_counts[method_name].extend(counts)
        # 按 target_row 排序，保证与顺序一致
        for method_name in per_target_counts:
            per_target_counts[method_name].sort(key=lambda x: x[0])
        # 汇总结果：每种方法下“至少匹配过任一 target”的 synthetic 条数
        results_aggregated = {}
        for method_name in union_matched:
            idx_set = union_matched[method_name]
            mask_agg = np.zeros(len(synthetic), dtype=bool)
            mask_agg[list(idx_set)] = True
            results_aggregated[method_name] = (mask_agg, synthetic.loc[mask_agg].reset_index(drop=True))
        results = results_aggregated
        print()
    else:
        target_idx = target_indices[0]
        target_row = target_source.iloc[target_idx : target_idx + 1]
        target_display = _feature_df(target_row)
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

    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if not use_all_rows:
            target_display = _feature_df(target_source.iloc[target_indices[0] : target_indices[0] + 1])
            target_display.to_csv(output_dir / "target.csv", index=False)

    for method_name, (mask, matched_df) in results.items():
        count = int(mask.sum())
        desc = method_descriptions.get(method_name, method_name)
        print("=" * 60)
        print(f"{method_name}: {desc}")
        print("=" * 60)
        if use_all_rows:
            print(f"  匹配条数（至少匹配过任一 target）: {count} / {len(synthetic)}")
            if per_target_counts:
                total_matches = sum(c for _, c in per_target_counts[method_name])
                print(f"  各 target 行匹配数之和: {total_matches}")
        else:
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
        if use_all_rows and output_dir is not None and per_target_counts:
            per_path = output_dir / f"per_target_count_{method_name}.csv"
            pd.DataFrame(per_target_counts[method_name], columns=["target_row", "match_count"]).to_csv(
                per_path, index=False
            )
            print(f"  每行 target 匹配数已写入: {per_path}")

    if output_dir:
        print()
        if not use_all_rows:
            print(f"Target 已写入: {output_dir / 'target.csv'}")
        print(f"各方法匹配结果已写入: {output_dir / 'matched_<method>.csv'}")
        if use_all_rows:
            print("各方法 per_target_count_<method>.csv 为每行 target 的匹配条数。")


if __name__ == "__main__":
    main()
