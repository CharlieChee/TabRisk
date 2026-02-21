#!/usr/bin/env bash
# DDPM shadow 实验：5 个 train_rows × 5 seeds = 25 个 run，顺序执行，可 nohup 后台
#
# 配置：类似 CTGAN A1-A4，但 model=ddpm
#   train_rows: 200, 500, 1000, 1500, 2000
#   batch_size: 128, 256, 256, 256, 256（对应上述 train_rows）
#   n_iter=1000, rounds=20, seeds=42,43,44,45,46
#
# 用法：
#   chmod +x scripts/run_ddpm_shadow_5seeds.sh
# 后台运行（推荐）：
#   cd /Users/changlong.ji/Desktop/project/TabRisk
#   nohup bash scripts/run_ddpm_shadow_5seeds.sh > run_ddpm_shadow_5seeds.log 2>&1 &
# 前台运行：
#   bash scripts/run_ddpm_shadow_5seeds.sh

set -e
cd "$(dirname "$0")/.."
LOG_DIR="${LOG_DIR:-logs_ddpm_shadow_5seeds}"
mkdir -p "$LOG_DIR"

SEEDS=(42 43 44 45 46)

run_one() {
  local train_rows=$1
  local batch_size=$2
  local seed=$3
  local done_file="$LOG_DIR/train${train_rows}_bs${batch_size}_seed${seed}.done"
  if [ -f "$done_file" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Skip (already done): train_rows=$train_rows bs=$batch_size seed=$seed"
    return 0
  fi
  local log_file="$LOG_DIR/train${train_rows}_bs${batch_size}_seed${seed}.log"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Start: train_rows=$train_rows bs=$batch_size seed=$seed -> $log_file"
  python -m synthgen.train \
    data=standard/adult_openml \
    model=ddpm model.params.device=cuda \
    preprocess=monotonic \
    train_rows="$train_rows" synthetic_rows="$train_rows" \
    model.params.n_iter=1000 model.params.batch_size="$batch_size" \
    shadow_model=true \
    shadow.num_shadow_rounds=20 shadow.max_targets=100 shadow.random_seed="$seed" \
    shadow.engine=worker \
    shadow.control_branch=true \
    shadow.gpu_ids="[0,1,2,3,4,5,6,7]" \
    shadow.worker_concurrency_per_gpu=4 \
    shadow.reuse_mode=true \
    shadow.reuse_num_bases=5 \
    shadow.reuse_num_out_candidates=5 \
    shadow.reuse_num_control_candidates=5 \
    >> "$log_file" 2>&1
  touch "$done_file"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done:  train_rows=$train_rows bs=$batch_size seed=$seed"
}

echo "========== DDPM shadow: train_rows=200, bs=128, 5 seeds =========="
for s in "${SEEDS[@]}"; do run_one 200 128 "$s"; done

echo "========== DDPM shadow: train_rows=500, bs=256, 5 seeds =========="
for s in "${SEEDS[@]}"; do run_one 500 256 "$s"; done

echo "========== DDPM shadow: train_rows=1000, bs=256, 5 seeds =========="
for s in "${SEEDS[@]}"; do run_one 1000 256 "$s"; done

echo "========== DDPM shadow: train_rows=1500, bs=256, 5 seeds =========="
for s in "${SEEDS[@]}"; do run_one 1500 256 "$s"; done

echo "========== DDPM shadow: train_rows=2000, bs=256, 5 seeds =========="
for s in "${SEEDS[@]}"; do run_one 2000 256 "$s"; done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All 25 runs finished. Logs under: $LOG_DIR"
