#!/usr/bin/env bash
# 实验 D1–D6 (Iter 曲线, N=500)：按顺序执行 9 个 run，可 nohup 后台运行
# D1: iter=200, 1 seed | D2: iter=300, 2 seeds | D3: iter=600, 1 seed
# D4: iter=1000, 2 seeds | D5: iter=2000, 2 seeds | D6: iter=4000, 1 seed
# 固定: train_rows=500, batch_size=256, rounds=20
#
# 可执行权限（可选）：
#   chmod +x scripts/run_exp_D1_D6_demo.sh
# 后台运行（推荐）：
#   cd /Users/changlong.ji/Desktop/project/TabRisk
#   nohup bash scripts/run_exp_D1_D6_demo.sh > run_exp_D1_D6.log 2>&1 &
# 前台运行：
#   bash scripts/run_exp_D1_D6_demo.sh

set -e
cd "$(dirname "$0")/.."
LOG_DIR="${LOG_DIR:-logs_exp_D1_D6}"
mkdir -p "$LOG_DIR"

run_one() {
  local exp_id=$1
  local train_rows=$2
  local n_iter=$3
  local batch_size=$4
  local rounds=$5
  local seed=$6
  local done_file="$LOG_DIR/${exp_id}_iter${n_iter}_seed${seed}.done"
  if [ -f "$done_file" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Skip (already done): $exp_id n_iter=$n_iter seed=$seed"
    return 0
  fi
  local log_file="$LOG_DIR/${exp_id}_iter${n_iter}_seed${seed}.log"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Start: $exp_id n_iter=$n_iter seed=$seed -> $log_file"
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
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done:  $exp_id n_iter=$n_iter seed=$seed"
}

echo "========== Exp D1 (train_rows=500, n_iter=200, batch_size=256, rounds=20, 1 seed) =========="
run_one D1 500 200 256 20 42

echo "========== Exp D2 (train_rows=500, n_iter=300, batch_size=256, rounds=20, 2 seeds) =========="
for s in 42 43; do run_one D2 500 300 256 20 "$s"; done

echo "========== Exp D3 (train_rows=500, n_iter=600, batch_size=256, rounds=20, 1 seed) =========="
run_one D3 500 600 256 20 42

echo "========== Exp D4 (train_rows=500, n_iter=1000, batch_size=256, rounds=20, 2 seeds) =========="
for s in 42 43; do run_one D4 500 1000 256 20 "$s"; done

echo "========== Exp D5 (train_rows=500, n_iter=2000, batch_size=256, rounds=20, 2 seeds) =========="
for s in 42 43; do run_one D5 500 2000 256 20 "$s"; done

echo "========== Exp D6 (train_rows=500, n_iter=4000, batch_size=256, rounds=20, 1 seed) =========="
run_one D6 500 4000 256 20 42

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All 9 runs finished."
