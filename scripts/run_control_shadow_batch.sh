#!/usr/bin/env bash
# 三组实验 × 2 个 seed，共 6 次顺序运行（跑完一个再跑下一个）。输出目录名会包含 seed。
# 三组：(train/synth, iter, bs) = (200, 1000, 200) | (500, 300, 256) | (1000, 1000, 256)
# 种子：123, 456（每组跑 2 次，不含 42）
# 在项目根目录执行：bash scripts/run_control_shadow_batch.sh

set -e
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"

BASE_ARGS="data=standard/adult_openml model=ctgan model.params.device=cuda preprocess=monotonic"
SHADOW_ARGS="shadow_model=true shadow.num_shadow_rounds=100 shadow.max_targets=100"
SHADOW_ENGINE="shadow.engine=worker shadow.control_branch=true"
# Hydra 需收到列表而非字符串，故用 [0,1,...] 不要外层引号
SHADOW_GPU="shadow.gpu_ids=[0,1,2,3,4,5,6,7] shadow.worker_concurrency_per_gpu=4"
SHADOW_REUSE="shadow.reuse_mode=true shadow.reuse_num_bases=10 shadow.reuse_num_out_candidates=10 shadow.reuse_num_control_candidates=10"

SEEDS=(123 456)

# 1) 200 / 1000iter / bs200
for seed in "${SEEDS[@]}"; do
  log="${LOG_DIR}/train_200_1000iter_bs200_seed${seed}.log"
  echo "Starting: train_rows=200 1000iter bs200 seed=${seed} -> $log"
  python -m synthgen.train \
    $BASE_ARGS \
    train_rows=200 synthetic_rows=200 \
    model.params.n_iter=1000 model.params.batch_size=200 \
    $SHADOW_ARGS shadow.random_seed=$seed \
    $SHADOW_ENGINE $SHADOW_GPU $SHADOW_REUSE \
    >> "$log" 2>&1
  echo "Done: seed=${seed} (200/1000iter/bs200)"
done

# 2) 500 / 300iter / bs256
for seed in "${SEEDS[@]}"; do
  log="${LOG_DIR}/train_500_300iter_bs256_seed${seed}.log"
  echo "Starting: train_rows=500 300iter bs256 seed=${seed} -> $log"
  python -m synthgen.train \
    $BASE_ARGS \
    train_rows=500 synthetic_rows=500 \
    model.params.n_iter=300 model.params.batch_size=256 \
    $SHADOW_ARGS shadow.random_seed=$seed \
    $SHADOW_ENGINE $SHADOW_GPU $SHADOW_REUSE \
    >> "$log" 2>&1
  echo "Done: seed=${seed} (500/300iter/bs256)"
done

# 3) 1000 / 1000iter / bs256
for seed in "${SEEDS[@]}"; do
  log="${LOG_DIR}/train_1000_1000iter_bs256_seed${seed}.log"
  echo "Starting: train_rows=1000 1000iter bs256 seed=${seed} -> $log"
  python -m synthgen.train \
    $BASE_ARGS \
    train_rows=1000 synthetic_rows=1000 \
    model.params.n_iter=1000 model.params.batch_size=256 \
    $SHADOW_ARGS shadow.random_seed=$seed \
    $SHADOW_ENGINE $SHADOW_GPU $SHADOW_REUSE \
    >> "$log" 2>&1
  echo "Done: seed=${seed} (1000/1000iter/bs256)"
done

echo "All 6 runs finished. Logs under: $LOG_DIR"
