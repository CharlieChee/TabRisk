#!/usr/bin/env python3
"""
DDPM n_iter 扫描：固定 train_rows=synthetic_rows=1000, batch_size=256，仅扫 n_iter。

配置：
  - data=standard/adult_openml
  - model=ddpm
  - model.params.device=cuda
  - preprocess=monotonic
  - train_rows=1000
  - synthetic_rows=1000
  - model.params.batch_size=256
  - n_iter ∈ {500, 1000, 2000, 4000}

输出：记录每组实验的 utility (Overall Score)，写入 outputs/sweep_ddpm_iter/ 并打印汇总。

用法（项目根目录）：
    python scripts/sweep_ddpm_iter.py
"""

import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_BASE = PROJECT_ROOT / "outputs" / "sweep_ddpm_iter"

BATCH_SIZE = 256
N_ITERS = [500, 1000, 2000, 4000]
NUM_GPUS = 8


def run_one(n_iter: int, gpu_id: int) -> dict:
    """运行单次 DDPM 实验，返回 {n_iter, gpu_id, output_dir, overall_score, error}。"""
    run_name = f"bs{BATCH_SIZE}_iter{n_iter}"
    output_dir_rel = f"outputs/sweep_ddpm_iter/{run_name}"

    cmd = [
        "python",
        "-m",
        "synthgen.train",
        "data=standard/adult_openml",
        "model=ddpm",
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
            timeout=7200,  # 2h per run，DDPM 训练可能较慢
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
            "error": "Timeout (2h)",
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
    print(f"[Sweep] 共 {len(N_ITERS)} 组 DDPM 实验，{NUM_GPUS} GPU 并行")
    print(f"[Sweep] batch_size={BATCH_SIZE}, n_iter={N_ITERS}")
    print()

    results = []
    with ThreadPoolExecutor(max_workers=min(NUM_GPUS, len(N_ITERS))) as ex:
        futures = {}
        for i, n_iter in enumerate(N_ITERS):
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
                    f"[Done] bs={BATCH_SIZE} n_iter={r['n_iter']:4d} -> Utility: {score_str}{err_str}"
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
    print("DDPM Utility 汇总 (Overall Score %)")
    print("=" * 60)
    for r in sorted(results, key=lambda x: x["n_iter"]):
        s = r["overall_score"]
        sc = f"{s*100:.2f}%" if s is not None else "N/A"
        print(f"  n_iter={r['n_iter']:4d} -> {sc}")

    # 保存汇总到 CSV 方便后续分析
    summary_path = OUTPUT_BASE / "sweep_summary.csv"
    with open(summary_path, "w") as fp:
        fp.write("n_iter,overall_score,error\n")
        for r in sorted(results, key=lambda x: x["n_iter"]):
            sc = f"{r['overall_score']}" if r["overall_score"] is not None else ""
            err = (r.get("error") or "").replace(",", ";")
            fp.write(f"{r['n_iter']},{sc},{err}\n")
    print()
    print(f"汇总已保存: {summary_path}")

    best = max(
        (r for r in results if r["overall_score"] is not None),
        key=lambda x: x["overall_score"],
        default=None,
    )
    if best:
        print(f"Best: n_iter={best['n_iter']} -> {best['overall_score']*100:.2f}%")


if __name__ == "__main__":
    main()
