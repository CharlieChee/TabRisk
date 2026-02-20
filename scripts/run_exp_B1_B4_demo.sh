#!/usr/bin/env bash
# 实验 B1–B4：按顺序执行 8 个 run（4 配置 × 2 seeds），可 nohup 后台运行
# B1–B4 仅 rounds 不同：5, 10, 20, 40；其余 train_rows=500, n_iter=1000, batch_size=256
#
# 可执行权限（可选，有则可直接 ./scripts/...）：
#   chmod +x scripts/run_exp_B1_B4_demo.sh
# 后台运行（推荐）：
#   cd /Users/changlong.ji/Desktop/project/TabRisk
#   nohup bash scripts/run_exp_B1_B4_demo.sh > run_exp_B1_B4.log 2>&1 &
# 前台运行：
#   bash scripts/run_exp_B1_B4_demo.sh

set -e
cd "$(dirname "$0")/.."
LOG_DIR="${LOG_DIR:-logs_exp_B1_B4}"
mkdir -p "$LOG_DIR"

# 实验配置: Exp ID | train_rows | n_iter | batch_size | rounds | num_seeds
# B1: 500, 1000, 256, 5,  2
# B2: 500, 1000, 256, 10, 2
# B3: 500, 1000, 256, 20, 2
# B4: 500, 1000, 256, 40, 2
SEEDS=(42 43)

run_one() {
  local exp_id=$1
  local train_rows=$2
  local n_iter=$3
  local batch_size=$4
  local rounds=$5
  local seed=$6
  local log_file="$LOG_DIR/${exp_id}_rounds${rounds}_seed${seed}.log"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Start: $exp_id rounds=$rounds seed=$seed -> $log_file"
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
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done:  $exp_id rounds=$rounds seed=$seed"
}

echo "========== Exp B1 (train_rows=500, n_iter=1000, batch_size=256, rounds=5) =========="
for s in "${SEEDS[@]}"; do run_one B1 500 1000 256 5 "$s"; done

echo "========== Exp B2 (train_rows=500, n_iter=1000, batch_size=256, rounds=10) =========="
for s in "${SEEDS[@]}"; do run_one B2 500 1000 256 10 "$s"; done

echo "========== Exp B3 (train_rows=500, n_iter=1000, batch_size=256, rounds=20) =========="
for s in "${SEEDS[@]}"; do run_one B3 500 1000 256 20 "$s"; done

echo "========== Exp B4 (train_rows=500, n_iter=1000, batch_size=256, rounds=40) =========="
for s in "${SEEDS[@]}"; do run_one B4 500 1000 256 40 "$s"; done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All 8 runs finished."
