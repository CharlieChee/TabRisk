#!/usr/bin/env python3
"""
DPGAN 超参扫描脚本：批量扫 bs / n_iter / epsilon / clipping_value，打印 utility。

用法（在项目根目录）：
    python scripts/sweep_dpgan.py
"""

import os
import subprocess
import itertools
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# 项目根目录（脚本在 scripts/ 下）
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_BASE = PROJECT_ROOT / "outputs" / "sweep_dpgan"

# Sweep 网格：
#   - batch_size: 256, 512
#   - n_iter: 50, 100
#   - epsilon: 0.5, 1.0, 5.0   （隐私预算，越大隐私越弱、utility 越高）
#   - clipping_value: 1, 2, 4   （生成器梯度裁剪阈值 C，DPGAN 中该参数为整数，0=关闭）
# 满足 batch_size <= train_rows/2，用 train_rows=2500
BATCH_SIZES = [256, 512]
N_ITERS = [50, 100]
EPSILONS = [0.5, 1.0, 5.0]
CLIPPING_VALUES = [1, 2, 4]

# 并行 GPU 数量（根据实际机器调整）
NUM_GPUS = 8


def run_one(bs: int, n_iter: int, epsilon: float, clipping_value: float, gpu_id: int) -> dict:
    """运行单次 DPGAN 实验，返回 {bs, n_iter, epsilon, clipping_value, gpu_id, output_dir, overall_score, error}。"""
    run_name = f"bs{bs}_iter{n_iter}_eps{epsilon}_clip{clipping_value}"
    output_dir_rel = f"outputs/sweep_dpgan/{run_name}"

    # train_rows=synthetic_rows=1000
    cmd = [
        "python",
        "-m",
        "synthgen.train",
        "data=standard/adult_openml",
        "model=dpgan",
        "model.params.device=cuda",
        "preprocess=monotonic",
        "train_rows=1000",
        "synthetic_rows=1000",
        "use_detailed_output_dir=false",
        f"output_dir={output_dir_rel}",
        f"model.params.batch_size={bs}",
        f"model.params.n_iter={n_iter}",
        f"+model.params.epsilon={epsilon}",
        f"+model.params.clipping_value={clipping_value}",
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
            "epsilon": epsilon,
            "clipping_value": clipping_value,
            "gpu_id": gpu_id,
            "output_dir": output_dir_rel,
            "overall_score": None,
            "error": str(e),
        }
    except subprocess.TimeoutExpired:
        return {
            "bs": bs,
            "n_iter": n_iter,
            "epsilon": epsilon,
            "clipping_value": clipping_value,
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
        "epsilon": epsilon,
        "clipping_value": clipping_value,
        "gpu_id": gpu_id,
        "output_dir": output_dir_rel,
        "overall_score": overall_score,
        "error": None,
    }


def main():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    configs = list(itertools.product(BATCH_SIZES, N_ITERS, EPSILONS, CLIPPING_VALUES))
    print(f"[Sweep] 共 {len(configs)} 组 DPGAN 实验，最多 {NUM_GPUS} GPU 并行")
    print(f"[Sweep] batch_size={BATCH_SIZES}, n_iter={N_ITERS}, epsilon={EPSILONS}, clipping_value={CLIPPING_VALUES}")
    print()

    results = []
    with ThreadPoolExecutor(max_workers=NUM_GPUS) as ex:
        futures = {}
        for i, (bs, n_iter, eps, clip) in enumerate(configs):
            gpu_id = i % NUM_GPUS
            f = ex.submit(run_one, bs, n_iter, eps, clip, gpu_id)
            futures[f] = (bs, n_iter, eps, clip)

        for f in as_completed(futures):
            bs, n_iter, eps, clip = futures[f]
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
                print(f"[Done] bs={bs} n_iter={n_iter} eps={eps} clip={clip} -> Error: {e}")
                results.append(
                    {
                        "bs": bs,
                        "n_iter": n_iter,
                        "epsilon": eps,
                        "clipping_value": clip,
                        "overall_score": None,
                        "error": str(e),
                    }
                )

    # 汇总
    print()
    print("=" * 60)
    print("DPGAN Utility 汇总 (Overall Score %)")
    print("=" * 60)
    for r in sorted(results, key=lambda x: (x["bs"], x["n_iter"], x["epsilon"], x["clipping_value"])):
        s = r["overall_score"]
        sc = f"{s*100:.2f}%" if s is not None else "N/A"
        print(
            f"  bs={r['bs']:4d} n_iter={r['n_iter']:4d} "
            f"eps={r['epsilon']:<4} clip={r['clipping_value']:<4} -> {sc}"
        )
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
