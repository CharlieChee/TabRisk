#!/usr/bin/env bash
# 实验 A1–A4，仅 seed=45 和 46：按顺序执行 8 个 run（4 配置 × 2 seeds）
# A1: 200, 1000, 128, 20 | A2: 500, 1000, 256, 20 | A3: 1000, 1000, 256, 20 | A4: 2000, 1000, 256, 20
#
# 可执行权限（可选）：
#   chmod +x scripts/run_exp_A1_A4_seed45_46.sh
# 后台运行（推荐）：
#   cd /Users/changlong.ji/Desktop/project/TabRisk
#   nohup bash scripts/run_exp_A1_A4_seed45_46.sh > run_exp_A1_A4_seed45_46.log 2>&1 &
# 前台运行：
#   bash scripts/run_exp_A1_A4_seed45_46.sh

set -e
cd "$(dirname "$0")/.."
LOG_DIR="${LOG_DIR:-logs_exp_A1_A4_seed45_46}"
mkdir -p "$LOG_DIR"

SEEDS=(45 46)

run_one() {
  local exp_id=$1
  local train_rows=$2
  local n_iter=$3
  local batch_size=$4
  local rounds=$5
  local seed=$6
  local done_file="$LOG_DIR/${exp_id}_rows${train_rows}_seed${seed}.done"
  if [ -f "$done_file" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Skip (already done): $exp_id train_rows=$train_rows seed=$seed"
    return 0
  fi
  local log_file="$LOG_DIR/${exp_id}_rows${train_rows}_seed${seed}.log"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Start: $exp_id train_rows=$train_rows seed=$seed -> $log_file"
  python -m synthgen.train \
    data=standard/adult_openml \
    model=ctgan model.params.device=cuda \
    preprocess=monotonic \
    train_rows="$train_rows" synthetic_rows="$train_rows" \
    model.params.n_iter="$n_iter" model.params.batch_size="$batch_size" \
    shadow_model=true \
    shadow.num_shadow_rounds="$rounds" shadow.max_targets=100 shadow.random_seed="$seed" \
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
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done:  $exp_id train_rows=$train_rows seed=$seed"
}

echo "========== Exp A1 (train_rows=200, n_iter=1000, batch_size=128, rounds=20) =========="
for s in "${SEEDS[@]}"; do run_one A1 200 1000 128 20 "$s"; done

echo "========== Exp A2 (train_rows=500, n_iter=1000, batch_size=256, rounds=20) =========="
for s in "${SEEDS[@]}"; do run_one A2 500 1000 256 20 "$s"; done

echo "========== Exp A3 (train_rows=1000, n_iter=1000, batch_size=256, rounds=20) =========="
for s in "${SEEDS[@]}"; do run_one A3 1000 1000 256 20 "$s"; done

echo "========== Exp A4 (train_rows=2000, n_iter=1000, batch_size=256, rounds=20) =========="
for s in "${SEEDS[@]}"; do run_one A4 2000 1000 256 20 "$s"; done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All 8 runs finished."
