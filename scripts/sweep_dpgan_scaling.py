#!/usr/bin/env python3
"""
DPGAN scaling sweep 脚本：

对每个 N ∈ {2000, 5000, 10000}（train_rows = synthetic_rows = N），
按你给出的 A-line / B-line / C-line 结构，扫：

  - A-line: bs=256, clip=2,  n_iter 按 A 等效表，eps ∈ {0.5, 1, 2, 5}
  - B-line: bs=512, clip=2,  n_iter 按 B 等效表，eps ∈ {0.5, 1, 2, 5}
  - C-line: bs=512, clip=4,  n_iter 按 C 等效表，eps ∈ {0.5, 1, 2, 5}

等效表（我按“总梯度步数近似不变”的原则给出一个具体数值，你可以在下面直接改）：

  记 steps ≈ n_iter * (N / bs)，希望在不同 N、bs 下 steps 大致接近。

  A-line（bs=256, clip=2）:
      N=2000  -> n_iter=50
      N=5000  -> n_iter=20
      N=10000 -> n_iter=10

  B-line（bs=512, clip=2）:
      N=2000  -> n_iter=100
      N=5000  -> n_iter=40
      N=10000 -> n_iter=20

  C-line（bs=512, clip=4）:
      默认采用与 B-line 相同的 n_iter（只改 clip），
      若你希望 C 更强训练，可以单独调下面的 C_EQ_N_ITERS。

共 3(N) × 12(每个 N) = 36 个实验。

用法（在项目根目录）：
    python scripts/sweep_dpgan_scaling.py
"""

import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_BASE = PROJECT_ROOT / "outputs" / "sweep_dpgan_scaling"

# N 值（train_rows = synthetic_rows = N）
N_VALUES = [2000, 5000, 10000]

# epsilon 网格
EPSILONS = [0.5, 1.0, 2.0, 5.0]

# A/B/C 等效表（可根据需要自行修改）
A_EQ_N_ITERS = {2000: 50, 5000: 20, 10000: 10}   # bs=256, clip=2
B_EQ_N_ITERS = {2000: 100, 5000: 40, 10000: 20}  # bs=512, clip=2
C_EQ_N_ITERS = {2000: 100, 5000: 40, 10000: 20}  # bs=512, clip=4（当前与 B 相同）

# 并行 GPU 数量（根据实际机器调整）
NUM_GPUS = 8


def build_configs():
    """根据 N / A-line / B-line / C-line 构造配置列表。"""
    configs = []
    for N in N_VALUES:
        # A-line: bs=256, clip=2
        for eps in EPSILONS:
            configs.append(
                {
                    "line": "A",
                    "N": N,
                    "bs": 256,
                    "clip": 2,
                    "n_iter": A_EQ_N_ITERS[N],
                    "epsilon": eps,
                }
            )
        # B-line: bs=512, clip=2
        for eps in EPSILONS:
            configs.append(
                {
                    "line": "B",
                    "N": N,
                    "bs": 512,
                    "clip": 2,
                    "n_iter": B_EQ_N_ITERS[N],
                    "epsilon": eps,
                }
            )
        # C-line: bs=512, clip=4
        for eps in EPSILONS:
            configs.append(
                {
                    "line": "C",
                    "N": N,
                    "bs": 512,
                    "clip": 4,
                    "n_iter": C_EQ_N_ITERS[N],
                    "epsilon": eps,
                }
            )
    return configs


def run_one(cfg: dict, gpu_id: int) -> dict:
    """运行单次 DPGAN 实验。cfg 中包含 N/bs/clip/n_iter/epsilon。"""
    N = cfg["N"]
    bs = cfg["bs"]
    n_iter = cfg["n_iter"]
    eps = cfg["epsilon"]
    clip = cfg["clip"]
    line = cfg["line"]

    run_name = f"N{N}_{line}_bs{bs}_iter{n_iter}_eps{eps}_clip{clip}"
    output_dir_rel = f"outputs/sweep_dpgan_scaling/{run_name}"

    cmd = [
        "python",
        "-m",
        "synthgen.train",
        "data=standard/adult_openml",
        "model=dpgan",
        "model.params.device=cuda",
        "preprocess=monotonic",
        f"train_rows={N}",
        f"synthetic_rows={N}",
        "use_detailed_output_dir=false",
        f"output_dir={output_dir_rel}",
        f"model.params.batch_size={bs}",
        f"model.params.n_iter={n_iter}",
        f"+model.params.epsilon={eps}",
        f"+model.params.clipping_value={clip}",
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
            **cfg,
            "gpu_id": gpu_id,
            "output_dir": output_dir_rel,
            "overall_score": None,
            "error": str(e),
        }
    except subprocess.TimeoutExpired:
        return {
            **cfg,
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
        **cfg,
        "gpu_id": gpu_id,
        "output_dir": output_dir_rel,
        "overall_score": overall_score,
        "error": None,
    }


def main():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    configs = build_configs()

    print(f"[Sweep] 共 {len(configs)} 组 DPGAN 实验，{NUM_GPUS} GPU 并行")
    print("  N ∈", N_VALUES)
    print("  eps ∈", EPSILONS)
    print("  A-line n_iter:", A_EQ_N_ITERS)
    print("  B-line n_iter:", B_EQ_N_ITERS)
    print("  C-line n_iter:", C_EQ_N_ITERS)
    print()

    results = []
    with ThreadPoolExecutor(max_workers=NUM_GPUS) as ex:
        futures = {}
        for i, cfg in enumerate(configs):
            gpu_id = i % NUM_GPUS
            f = ex.submit(run_one, cfg, gpu_id)
            futures[f] = cfg

        for f in as_completed(futures):
            cfg = futures[f]
            try:
                r = f.result()
                results.append(r)
                sc = (
                    f"{r['overall_score']*100:.2f}%"
                    if r["overall_score"] is not None
                    else "N/A"
                )
                err = f" ({r['error']})" if r["error"] else ""
                print(
                    f"[Done] N={r['N']} line={r['line']} "
                    f"bs={r['bs']} n_iter={r['n_iter']} "
                    f"eps={r['epsilon']} clip={r['clip']} -> {sc}{err}"
                )
            except Exception as e:
                print(
                    f"[Done] N={cfg['N']} line={cfg['line']} "
                    f"bs={cfg['bs']} n_iter={cfg['n_iter']} "
                    f"eps={cfg['epsilon']} clip={cfg['clip']} -> Error: {e}"
                )
                results.append({**cfg, "overall_score": None, "error": str(e)})

    # 简单汇总：按 (N, line, bs, n_iter, epsilon, clip) 排序打印
    print()
    print("=" * 60)
    print("DPGAN scaling Utility 汇总 (Overall Score %)")
    print("=" * 60)
    for r in sorted(
        results,
        key=lambda x: (
            x["N"],
            x["line"],
            x["bs"],
            x["n_iter"],
            x["epsilon"],
            x["clip"],
        ),
    ):
        s = r["overall_score"]
        sc = f"{s*100:.2f}%" if s is not None else "N/A"
        print(
            f"  N={r['N']:5d} line={r['line']} "
            f"bs={r['bs']:4d} n_iter={r['n_iter']:4d} "
            f"eps={r['epsilon']:<4} clip={r['clip']:<2} -> {sc}"
        )


if __name__ == "__main__":
    main()

