#!/usr/bin/env bash
# DDPM shadow：仅 train_rows=2000，5 seeds，逻辑与 run_ddpm_shadow_5seeds.sh 一致，仅用 GPU 4,5,6,7
#
# 开头设置 CUDA_VISIBLE_DEVICES=4,5,6,7，进程内只看到 4 张卡（编号 0,1,2,3），故 shadow.gpu_ids="[0,1,2,3]"。
#
# 用法：
#   nohup bash scripts/run_ddpm_shadow_2000_4gpu.sh > run_ddpm_shadow_2000_4gpu.log 2>&1 &
# 或前台：bash scripts/run_ddpm_shadow_2000_4gpu.sh

set -e
cd "$(dirname "$0")/.."
LOG_DIR="${LOG_DIR:-logs_ddpm_shadow_2000_4gpu}"
mkdir -p "$LOG_DIR"

# 只用物理 GPU 4,5,6,7（进程内为 0,1,2,3）
export CUDA_VISIBLE_DEVICES=4,5,6,7

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
    shadow.gpu_ids="[0,1,2,3]" \
    shadow.worker_concurrency_per_gpu=4 \
    shadow.reuse_mode=true \
    shadow.reuse_num_bases=5 \
    shadow.reuse_num_out_candidates=5 \
    shadow.reuse_num_control_candidates=5 \
    >> "$log_file" 2>&1
  touch "$done_file"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done:  train_rows=$train_rows bs=$batch_size seed=$seed"
}

echo "========== DDPM shadow: train_rows=2000, bs=256, 5 seeds (GPUs 4,5,6,7) =========="
for s in "${SEEDS[@]}"; do run_one 2000 256 "$s"; done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All 5 runs finished. Logs under: $LOG_DIR"
