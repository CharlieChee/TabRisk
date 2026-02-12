#!/usr/bin/env python3
"""
PATEGAN 超参扫描脚本：9 组实验，最多 8 GPU 并行，打印 utility。

用法（在项目根目录）：
    python scripts/sweep_pategan.py
"""

import os
import subprocess
import itertools
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# 项目根目录（脚本在 scripts/ 下）
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_BASE = PROJECT_ROOT / "outputs" / "sweep_pategan"

# Sweep 网格：batch (16, 32)，iter (2, 5, 8, 10, 15, 20)，满足 batch_size <= train_rows/2
BATCH_SIZES = [16, 32]
N_ITERS = [2, 5, 8, 10, 15, 20]

# 并行 GPU 数量（根据实际机器调整）
NUM_GPUS = 8


def run_one(bs: int, n_iter: int, gpu_id: int) -> dict:
    """运行单次 PATEGAN 实验，返回 {bs, n_iter, gpu_id, output_dir, overall_score, error}。"""
    run_name = f"bs{bs}_iter{n_iter}"
    output_dir_rel = f"outputs/sweep_pategan/{run_name}"

    # 这里将 train_rows 设为 2000，确保 batch_size<=train_rows/2 对三种 bs 都成立
    cmd = [
        "python",
        "-m",
        "synthgen.train",
        "data=standard/adult_openml",
        "model=pategan",
        "model.params.device=cuda",
        "preprocess=monotonic",
        "train_rows=2000",
        "synthetic_rows=2000",
        "use_detailed_output_dir=false",
        f"output_dir={output_dir_rel}",
        f"model.params.batch_size={bs}",
        f"model.params.n_iter={n_iter}",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=3600,
        )
    except subprocess.CalledProcessError as e:
        return {
            "bs": bs,
            "n_iter": n_iter,
            "gpu_id": gpu_id,
            "output_dir": output_dir_rel,
            "overall_score": None,
            "error": str(e),
        }
    except subprocess.TimeoutExpired:
        return {
            "bs": bs,
            "n_iter": n_iter,
            "gpu_id": gpu_id,
            "output_dir": output_dir_rel,
            "overall_score": None,
            "error": "Timeout (1h)",
        }

    # 读取 utility (Overall Score)
    metrics_path = PROJECT_ROOT / output_dir_rel / "eval" / "metrics_sdv.csv"
    overall_score = None
    if metrics_path.exists():
        lines = metrics_path.read_text().strip().splitlines()
        if len(lines) > 1:
            for line in lines[1:]:
                parts = line.split(",", 1)
                if len(parts) >= 2:
                    k, v = parts[0].strip(), parts[1].strip()
                    if k in ("quality_overall", "overall_score"):
                        try:
                            overall_score = float(v)
                            break
                        except ValueError:
                            pass

    return {
        "bs": bs,
        "n_iter": n_iter,
        "gpu_id": gpu_id,
        "output_dir": output_dir_rel,
        "overall_score": overall_score,
        "error": None,
    }


def main():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    configs = list(itertools.product(BATCH_SIZES, N_ITERS))
    print(f"[Sweep] 共 {len(configs)} 组 PATEGAN 实验，最多 {NUM_GPUS} GPU 并行")
    print(f"[Sweep] batch_size={BATCH_SIZES}, n_iter={N_ITERS}")
    print()

    results = []
    with ThreadPoolExecutor(max_workers=NUM_GPUS) as ex:
        futures = {}
        for i, (bs, n_iter) in enumerate(configs):
            gpu_id = i % NUM_GPUS
            f = ex.submit(run_one, bs, n_iter, gpu_id)
            futures[f] = (bs, n_iter)

        for f in as_completed(futures):
            bs, n_iter = futures[f]
            try:
                r = f.result()
                results.append(r)
                score_str = (
                    f"{r['overall_score']*100:.2f}%"
                    if r["overall_score"] is not None
                    else "N/A"
                )
                err_str = f" ({r['error']})" if r["error"] else ""
                print(
                    f"[Done] bs={r['bs']} n_iter={r['n_iter']} -> Overall Score: {score_str}{err_str}"
                )
            except Exception as e:
                print(f"[Done] bs={bs} n_iter={n_iter} -> Error: {e}")
                results.append(
                    {
                        "bs": bs,
                        "n_iter": n_iter,
                        "overall_score": None,
                        "error": str(e),
                    }
                )

    # 汇总
    print()
    print("=" * 60)
    print("PATEGAN Utility 汇总 (Overall Score %)")
    print("=" * 60)
    for r in sorted(results, key=lambda x: (x["bs"], x["n_iter"])):
        s = r["overall_score"]
        sc = f"{s*100:.2f}%" if s is not None else "N/A"
        print(f"  bs={r['bs']:4d} n_iter={r['n_iter']:4d} -> {sc}")
    best = max(
        (r for r in results if r["overall_score"] is not None),
        key=lambda x: x["overall_score"],
        default=None,
    )
    if best:
        print()
        print(
            f"Best: bs={best['bs']} n_iter={best['n_iter']} "
            f"-> {best['overall_score']*100:.2f}%"
        )


if __name__ == "__main__":
    main()

