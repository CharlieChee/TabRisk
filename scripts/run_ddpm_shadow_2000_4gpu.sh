#!/usr/bin/env bash
# DDPM shadow：train_rows=2000，5 seeds，物理 GPU 4,5,6,7，每张卡跑 4 个任务（不修改 run_shadow.py）
#
# 单进程内多卡 + spawn worker 会在子进程里导致 GPU 不生效，故不 export 多卡。改为：每个 seed
# 单独一个进程且只绑一张卡（CUDA_VISIBLE_DEVICES=4|5|6|7），shadow.gpu_ids=[0]，worker_concurrency_per_gpu=4，
# 即每张 GPU 跑 4 个任务。前 4 个 seed 并行占满 4 张卡，第 5 个等位。
#
# 用法：
#   nohup bash scripts/run_ddpm_shadow_2000_4gpu.sh > run_ddpm_shadow_2000_4gpu.log 2>&1 &
# 或前台：bash scripts/run_ddpm_shadow_2000_4gpu.sh

set -e
cd "$(dirname "$0")/.."
LOG_DIR="${LOG_DIR:-logs_ddpm_shadow_2000_4gpu}"
mkdir -p "$LOG_DIR"

PHYSICAL_GPUS=(4 5 6 7)
SEEDS=(42 43 44 45 46)
TRAIN_ROWS=2000
BATCH_SIZE=256

# 参数：seed, 物理 GPU 号。每进程只看到 1 张卡，故 shadow.gpu_ids=[0]，该卡上跑 4 个任务。
run_one() {
  local seed=$1
  local gpu_id=$2
  local done_file="$LOG_DIR/train${TRAIN_ROWS}_bs${BATCH_SIZE}_seed${seed}.done"
  if [ -f "$done_file" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Skip (already done): train_rows=$TRAIN_ROWS seed=$seed"
    return 0
  fi
  local log_file="$LOG_DIR/train${TRAIN_ROWS}_bs${BATCH_SIZE}_seed${seed}.log"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Start: train_rows=$TRAIN_ROWS seed=$seed GPU=$gpu_id (4 tasks on this GPU) -> $log_file"
  CUDA_VISIBLE_DEVICES="$gpu_id" python -m synthgen.train \
    data=standard/adult_openml \
    model=ddpm model.params.device=cuda \
    preprocess=monotonic \
    train_rows="$TRAIN_ROWS" synthetic_rows="$TRAIN_ROWS" \
    model.params.n_iter=1000 model.params.batch_size="$BATCH_SIZE" \
    shadow_model=true \
    shadow.num_shadow_rounds=20 shadow.max_targets=100 shadow.random_seed="$seed" \
    shadow.engine=worker \
    shadow.control_branch=true \
    'shadow.gpu_ids=[0]' \
    shadow.worker_concurrency_per_gpu=4 \
    shadow.reuse_mode=true \
    shadow.reuse_num_bases=5 \
    shadow.reuse_num_out_candidates=5 \
    shadow.reuse_num_control_candidates=5 \
    >> "$log_file" 2>&1
  touch "$done_file"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done:  train_rows=$TRAIN_ROWS seed=$seed"
}

echo "========== DDPM shadow: train_rows=2000, bs=256, GPUs 4,5,6,7 (4 tasks/GPU, script-only), 5 seeds =========="
n_gpu=${#PHYSICAL_GPUS[@]}
for i in "${!SEEDS[@]}"; do
  if [ "$i" -ge "$n_gpu" ]; then
    wait -n 2>/dev/null || true
  fi
  run_one "${SEEDS[i]}" "${PHYSICAL_GPUS[i % n_gpu]}" &
done
wait

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All 5 runs finished. Logs under: $LOG_DIR"
