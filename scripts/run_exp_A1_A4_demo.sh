#!/usr/bin/env bash
# 实验 A1–A4：按顺序执行 12 个 run（4 配置 × 3 seeds），可 nohup 后台运行
#
# 可执行权限（可选，有则可直接 ./scripts/...）：
#   chmod +x scripts/run_exp_A1_A4_demo.sh
# 后台运行（推荐）：
#   cd /Users/changlong.ji/Desktop/project/TabRisk
#   nohup bash scripts/run_exp_A1_A4_demo.sh > run_exp_A1_A4.log 2>&1 &
# 前台运行：
#   bash scripts/run_exp_A1_A4_demo.sh
# 断点续跑：已完成的 run 会跳过（依据 logs_exp_A1_A4/*.done）。若中途杀进程，
# 可先根据旧 log 标记已完成再重跑：bash scripts/resume_exp_A1_A4_from_log.sh

set -e
cd "$(dirname "$0")/.."
LOG_DIR="${LOG_DIR:-logs_exp_A1_A4}"
mkdir -p "$LOG_DIR"

# 实验配置: Exp ID | train_rows | n_iter | batch_size | rounds | num_seeds
# A1: 200, 1000, 128, 20, 3
# A2: 500, 1000, 256, 20, 3
# A3: 1000, 1000, 256, 20, 3
# A4: 2000, 1000, 256, 20, 3
SEEDS=(42 43 44)

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

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All 12 runs finished."
