#!/usr/bin/env bash
# 实验 C1–C5：按顺序执行 6 个 run，可 nohup 后台运行
# C1: N=500, R=80, 2 seeds | C2: N=500, R=160, 1 seed
# C3: N=1000, R=10, 1 seed | C4: N=1000, R=40, 1 seed | C5: N=1000, R=80, 1 seed
#
# 可执行权限（可选）：
#   chmod +x scripts/run_exp_C1_C5_demo.sh
# 后台运行（推荐）：
#   cd /Users/changlong.ji/Desktop/project/TabRisk
#   nohup bash scripts/run_exp_C1_C5_demo.sh > run_exp_C1_C5.log 2>&1 &
# 前台运行：
#   bash scripts/run_exp_C1_C5_demo.sh

set -e
cd "$(dirname "$0")/.."
LOG_DIR="${LOG_DIR:-logs_exp_C1_C5}"
mkdir -p "$LOG_DIR"

run_one() {
  local exp_id=$1
  local train_rows=$2
  local n_iter=$3
  local batch_size=$4
  local rounds=$5
  local seed=$6
  local done_file="$LOG_DIR/${exp_id}_rounds${rounds}_seed${seed}.done"
  if [ -f "$done_file" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Skip (already done): $exp_id rounds=$rounds seed=$seed"
    return 0
  fi
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
  touch "$done_file"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done:  $exp_id rounds=$rounds seed=$seed"
}

echo "========== Exp C1 (train_rows=500, n_iter=1000, batch_size=256, rounds=80, 2 seeds) =========="
for s in 42 43; do run_one C1 500 1000 256 80 "$s"; done

echo "========== Exp C2 (train_rows=500, n_iter=1000, batch_size=256, rounds=160, 1 seed) =========="
run_one C2 500 1000 256 160 42

echo "========== Exp C3 (train_rows=1000, n_iter=1000, batch_size=256, rounds=10, 1 seed) =========="
run_one C3 1000 1000 256 10 42

echo "========== Exp C4 (train_rows=1000, n_iter=1000, batch_size=256, rounds=40, 1 seed) =========="
run_one C4 1000 1000 256 40 42

echo "========== Exp C5 (train_rows=1000, n_iter=1000, batch_size=256, rounds=80, 1 seed) =========="
run_one C5 1000 1000 256 80 42

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All 6 runs finished."
