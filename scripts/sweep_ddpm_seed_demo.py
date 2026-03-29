#!/usr/bin/env python3
"""
DDPM 三组配置 × 5 seed 复现脚本：15 个实验，8 GPU 并行。

配置：
  1. bs=128, n_iter=1000, lr=1e-3  (best)
  2. bs=256, n_iter=1000, lr=1e-3  (次优，同 lr)
  3. bs=128, n_iter=1000, lr=3e-4  (对照：同 iter/batch，lr 更小)

用法：
    python scripts/sweep_ddpm_seed_demo.py
"""

import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_BASE = PROJECT_ROOT / "outputs" / "sweep_ddpm_seed"
NUM_GPUS = 8
NUM_SEEDS = 5

CONFIGS = [
    {"bs": 128, "n_iter": 1000, "lr": 1e-3},   # best
    {"bs": 256, "n_iter": 1000, "lr": 1e-3},   # 次优
    {"bs": 128, "n_iter": 1000, "lr": 3e-4},   # 对照
]


def run_one(cfg: dict, seed: int, gpu_id: int) -> dict:
    """运行单次实验。"""
    bs, n_iter, lr = cfg["bs"], cfg["n_iter"], cfg["lr"]
    run_name = f"bs{bs}_iter{n_iter}_lr{lr}_seed{seed}"
    output_dir_rel = f"outputs/sweep_ddpm_seed/{run_name}"

    cmd = [
        "python", "-m", "synthgen.train",
        "data=standard/adult_openml",
        "model=ddpm",
        "model.params.device=cuda",
        "preprocess=monotonic",
        "train_rows=1000",
        "synthetic_rows=1000",
        "use_detailed_output_dir=false",
        f"output_dir={output_dir_rel}",
        f"seed={seed}",
        f"model.params.batch_size={bs}",
        f"model.params.n_iter={n_iter}",
        f"+model.params.lr={lr}",
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
        return {**cfg, "seed": seed, "gpu_id": gpu_id, "output_dir": output_dir_rel, "overall_score": None, "error": str(e)}
    except subprocess.TimeoutExpired:
        return {**cfg, "seed": seed, "gpu_id": gpu_id, "output_dir": output_dir_rel, "overall_score": None, "error": "Timeout (1h)"}

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

    return {**cfg, "seed": seed, "gpu_id": gpu_id, "output_dir": output_dir_rel, "overall_score": overall_score, "error": None}


def main():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    seeds = list(range(NUM_SEEDS))
    tasks = [(cfg, seed) for cfg in CONFIGS for seed in seeds]
    total = len(tasks)

    print(f"[Sweep] 3 组 × {NUM_SEEDS} seed = {total} 个实验，{NUM_GPUS} GPU 并行")
    for i, c in enumerate(CONFIGS):
        print(f"  组{i+1}: bs={c['bs']} n_iter={c['n_iter']} lr={c['lr']}")
    print()

    results = []
    with ThreadPoolExecutor(max_workers=NUM_GPUS) as ex:
        futures = {}
        for idx, (cfg, seed) in enumerate(tasks):
            gpu_id = idx % NUM_GPUS
            f = ex.submit(run_one, cfg, seed, gpu_id)
            futures[f] = (cfg, seed)

        for f in as_completed(futures):
            cfg, seed = futures[f]
            try:
                r = f.result()
                results.append(r)
                sc = f"{r['overall_score']*100:.2f}%" if r["overall_score"] is not None else "N/A"
                err = f" ({r['error']})" if r["error"] else ""
                print(f"[Done] bs={r['bs']} iter={r['n_iter']} lr={r['lr']} seed={r['seed']} -> {sc}{err}")
            except Exception as e:
                print(f"[Done] bs={cfg['bs']} seed={seed} -> Error: {e}")
                results.append({**cfg, "seed": seed, "overall_score": None, "error": str(e)})

    # 按组汇总 mean ± std
    print()
    print("=" * 60)
    print("Utility 汇总 (Overall Score %, mean ± std)")
    print("=" * 60)
    for cfg in CONFIGS:
        subset = [r for r in results if r["bs"] == cfg["bs"] and r["n_iter"] == cfg["n_iter"] and r["lr"] == cfg["lr"]]
        scores = [r["overall_score"] for r in subset if r["overall_score"] is not None]
        if scores:
            import statistics
            mean = statistics.mean(scores) * 100
            std = statistics.stdev(scores) * 100 if len(scores) > 1 else 0
            print(f"  bs={cfg['bs']} n_iter={cfg['n_iter']} lr={cfg['lr']}: {mean:.2f} ± {std:.2f}%  (n={len(scores)})")
        else:
            print(f"  bs={cfg['bs']} n_iter={cfg['n_iter']} lr={cfg['lr']}: N/A (all failed)")


if __name__ == "__main__":
    main()
