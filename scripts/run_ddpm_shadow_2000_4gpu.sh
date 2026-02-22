#!/usr/bin/env bash
# DDPM shadow：仅 train_rows=2000，5 seeds，只用物理 GPU 4,5,6,7
#
# 通过 CUDA_VISIBLE_DEVICES=4,5,6,7 限制进程只看到这 4 张卡（进程内编号为 0,1,2,3），
# 再传 shadow.gpu_ids=[0,1,2,3] 让 shadow worker 用满这 4 张。避免仅改 gpu_ids 时
# Hydra 解析成字符串导致仍用默认 0,1,2,3。
#
# 用法：
#   nohup bash scripts/run_ddpm_shadow_2000_4gpu.sh > run_ddpm_shadow_2000_4gpu.log 2>&1 &
# 或前台：bash scripts/run_ddpm_shadow_2000_4gpu.sh

set -e
cd "$(dirname "$0")/.."
LOG_DIR="${LOG_DIR:-logs_ddpm_shadow_2000_4gpu}"
mkdir -p "$LOG_DIR"

# 仅让当前进程及其子进程看到物理 GPU 4,5,6,7（进程内为 0,1,2,3）
export CUDA_VISIBLE_DEVICES=4,5,6,7

SEEDS=(42 43 44 45 46)
TRAIN_ROWS=2000
BATCH_SIZE=256

run_one() {
  local seed=$1
  local done_file="$LOG_DIR/train${TRAIN_ROWS}_bs${BATCH_SIZE}_seed${seed}.done"
  if [ -f "$done_file" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Skip (already done): train_rows=$TRAIN_ROWS seed=$seed"
    return 0
  fi
  local log_file="$LOG_DIR/train${TRAIN_ROWS}_bs${BATCH_SIZE}_seed${seed}.log"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Start: train_rows=$TRAIN_ROWS seed=$seed (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES) -> $log_file"
  python -m synthgen.train \
    data=standard/adult_openml \
    model=ddpm model.params.device=cuda \
    preprocess=monotonic \
    train_rows="$TRAIN_ROWS" synthetic_rows="$TRAIN_ROWS" \
    model.params.n_iter=1000 model.params.batch_size="$BATCH_SIZE" \
    shadow_model=true \
    shadow.num_shadow_rounds=20 shadow.max_targets=100 shadow.random_seed="$seed" \
    shadow.engine=worker \
    shadow.control_branch=true \
    'shadow.gpu_ids=[0,1,2,3]' \
    shadow.worker_concurrency_per_gpu=4 \
    shadow.reuse_mode=true \
    shadow.reuse_num_bases=5 \
    shadow.reuse_num_out_candidates=5 \
    shadow.reuse_num_control_candidates=5 \
    >> "$log_file" 2>&1
  touch "$done_file"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done:  train_rows=$TRAIN_ROWS seed=$seed"
}

echo "========== DDPM shadow: train_rows=2000, bs=256, GPUs 4,5,6,7, 5 seeds =========="
for s in "${SEEDS[@]}"; do run_one "$s"; done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All 5 runs finished. Logs under: $LOG_DIR"
