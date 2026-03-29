#!/usr/bin/env bash
# 单组实验：train_rows=1500, n_iter=1000, batch_size=256, rounds=20, 5 seeds，共 5 个 run
#
# 可执行权限（可选）：
#   chmod +x scripts/run_exp_train1500_5seeds.sh
# 后台运行（推荐）：
#   cd /Users/changlong.ji/Desktop/project/TabRisk
#   nohup bash scripts/run_exp_train1500_5seeds.sh > run_exp_train1500_5seeds.log 2>&1 &
# 前台运行：
#   bash scripts/run_exp_train1500_5seeds.sh

set -e
cd "$(dirname "$0")/.."
LOG_DIR="${LOG_DIR:-logs_exp_train1500_5seeds}"
mkdir -p "$LOG_DIR"

SEEDS=(42 43 44 45 46)

run_one() {
  local seed=$1
  local done_file="$LOG_DIR/train1500_rounds20_seed${seed}.done"
  if [ -f "$done_file" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Skip (already done): train1500 rounds=20 seed=$seed"
    return 0
  fi
  local log_file="$LOG_DIR/train1500_rounds20_seed${seed}.log"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Start: train1500 rounds=20 seed=$seed -> $log_file"
  python -m synthgen.train \
    data=standard/adult_openml \
    model=ctgan model.params.device=cuda \
    preprocess=monotonic \
    train_rows=1500 synthetic_rows=1500 \
    model.params.n_iter=1000 model.params.batch_size=256 \
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
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done:  train1500 rounds=20 seed=$seed"
}

echo "========== train_rows=1500, n_iter=1000, batch_size=256, rounds=20, 5 seeds =========="
for s in "${SEEDS[@]}"; do run_one "$s"; done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All 5 runs finished."
