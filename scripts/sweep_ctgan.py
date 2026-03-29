#!/usr/bin/env python3
"""
CTGAN 超参扫描脚本：固定 train_rows=synthetic_rows=1000，仅扫 n_iter。

配置：
  - data=standard/adult_openml
  - model=ctgan
  - preprocess=monotonic
  - model.params.device=cuda
  - train_rows=1000
  - synthetic_rows=1000
  - batch_size=1024 固定
  - n_iter ∈ {10, 20, 30, 50, 100, 300}

用法（在项目根目录）：
    python scripts/sweep_ctgan.py
"""

import os
import subprocess
import itertools
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_BASE = PROJECT_ROOT / "outputs" / "sweep_ctgan"

# Sweep 网格：batch_size 固定 1024，仅扫 n_iter
BATCH_SIZE = 1024
N_ITERS = [150, 200, 250, 300, 350, 400]

# 并行 GPU 数量（根据实际机器调整）
NUM_GPUS = 8


def run_one(n_iter: int, gpu_id: int) -> dict:
    """运行单次 CTGAN 实验，返回 {n_iter, gpu_id, output_dir, overall_score, error}。"""
    run_name = f"bs{BATCH_SIZE}_iter{n_iter}"
    output_dir_rel = f"outputs/sweep_ctgan/{run_name}"

    cmd = [
        "python",
        "-m",
        "synthgen.train",
        "data=standard/adult_openml",
        "model=ctgan",
        "model.params.device=cuda",
        "preprocess=monotonic",
        "train_rows=1000",
        "synthetic_rows=1000",
        "use_detailed_output_dir=false",
        f"output_dir={output_dir_rel}",
        f"model.params.batch_size={BATCH_SIZE}",
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
            "n_iter": n_iter,
            "gpu_id": gpu_id,
            "output_dir": output_dir_rel,
            "overall_score": None,
            "error": str(e),
        }
    except subprocess.TimeoutExpired:
        return {
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
        "n_iter": n_iter,
        "gpu_id": gpu_id,
        "output_dir": output_dir_rel,
        "overall_score": overall_score,
        "error": None,
    }


def main():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    configs = list(N_ITERS)
    print(f"[Sweep] 共 {len(configs)} 组 CTGAN 实验，{NUM_GPUS} GPU 并行")
    print(f"[Sweep] batch_size={BATCH_SIZE}, n_iter={N_ITERS}")
    print()

    results = []
    with ThreadPoolExecutor(max_workers=NUM_GPUS) as ex:
        futures = {}
        for i, n_iter in enumerate(configs):
            gpu_id = i % NUM_GPUS
            f = ex.submit(run_one, n_iter, gpu_id)
            futures[f] = n_iter

        for f in as_completed(futures):
            n_iter = futures[f]
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
                    f"[Done] bs={BATCH_SIZE} n_iter={r['n_iter']} -> Overall Score: {score_str}{err_str}"
                )
            except Exception as e:
                print(f"[Done] bs={BATCH_SIZE} n_iter={n_iter} -> Error: {e}")
                results.append(
                    {
                        "n_iter": n_iter,
                        "overall_score": None,
                        "error": str(e),
                    }
                )

    # 汇总
    print()
    print("=" * 60)
    print("CTGAN Utility 汇总 (Overall Score %)")
    print("=" * 60)
    for r in sorted(results, key=lambda x: x["n_iter"]):
        s = r["overall_score"]
        sc = f"{s*100:.2f}%" if s is not None else "N/A"
        print(f"  bs={BATCH_SIZE:4d} n_iter={r['n_iter']:4d} -> {sc}")


if __name__ == "__main__":
    main()

