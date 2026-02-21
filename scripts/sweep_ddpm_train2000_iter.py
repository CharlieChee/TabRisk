#!/usr/bin/env python3
"""
DDPM sweep：train_rows=2000, n_iter ∈ {2000, 4000, 6000, 8000}，输出 utility。

配置：
  - data=standard/adult_openml
  - model=ddpm
  - model.params.device=cuda
  - preprocess=monotonic
  - train_rows=2000
  - synthetic_rows=2000
  - model.params.batch_size=256
  - n_iter ∈ {2000, 4000, 6000, 8000}

用法（项目根目录）：
    python scripts/sweep_ddpm_train2000_iter.py
"""

import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_BASE = PROJECT_ROOT / "outputs" / "sweep_ddpm_train2000_iter"

TRAIN_ROWS = 2000
BATCH_SIZE = 256
N_ITERS = [2000, 4000, 6000, 8000]
NUM_GPUS = 8


def run_one(n_iter: int, gpu_id: int) -> dict:
    """运行单次 DDPM 实验，返回 {n_iter, gpu_id, output_dir, overall_score, error}。"""
    run_name = f"train{TRAIN_ROWS}_bs{BATCH_SIZE}_iter{n_iter}"
    output_dir_rel = f"outputs/sweep_ddpm_train2000_iter/{run_name}"

    cmd = [
        "python",
        "-m",
        "synthgen.train",
        "data=standard/adult_openml",
        "model=ddpm",
        "model.params.device=cuda",
        "preprocess=monotonic",
        f"train_rows={TRAIN_ROWS}",
        f"synthetic_rows={TRAIN_ROWS}",
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
            timeout=7200,
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

    metrics_path = PROJECT_ROOT / output_dir_rel / "eval" / "metrics_sdv.csv"
    overall_score = None
    if metrics_path.exists():
        for line in metrics_path.read_text().strip().splitlines()[1:]:
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
    print(f"[Sweep] train_rows={TRAIN_ROWS}, 共 {len(N_ITERS)} 组 DDPM 实验，{NUM_GPUS} GPU 并行")
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
                score_str = f"{r['overall_score']*100:.2f}%" if r["overall_score"] is not None else "N/A"
                err_str = f" ({r['error']})" if r["error"] else ""
                print(f"[Done] train={TRAIN_ROWS} n_iter={r['n_iter']:4d} -> Utility: {score_str}{err_str}")
            except Exception as e:
                print(f"[Done] train={TRAIN_ROWS} n_iter={n_iter} -> Error: {e}")
                results.append({"n_iter": n_iter, "overall_score": None, "error": str(e)})

    print()
    print("=" * 60)
    print("DDPM Utility 汇总 (train_rows=2000)")
    print("=" * 60)
    for r in sorted(results, key=lambda x: x["n_iter"]):
        s = r["overall_score"]
        sc = f"{s*100:.2f}%" if s is not None else "N/A"
        print(f"  n_iter={r['n_iter']:4d} -> {sc}")

    summary_path = OUTPUT_BASE / "sweep_summary.csv"
    with open(summary_path, "w") as fp:
        fp.write("train_rows,n_iter,overall_score,error\n")
        for r in sorted(results, key=lambda x: x["n_iter"]):
            sc = f"{r['overall_score']}" if r["overall_score"] is not None else ""
            err = (r.get("error") or "").replace(",", ";")
            fp.write(f"{TRAIN_ROWS},{r['n_iter']},{sc},{err}\n")
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
