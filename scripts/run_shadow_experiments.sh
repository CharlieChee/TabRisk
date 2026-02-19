#!/bin/bash
# 按顺序执行 6 组 shadow 实验
# 使用方式: 在项目根目录执行 bash scripts/run_shadow_experiments.sh

set -e
cd "$(dirname "$0")/.."

BASE_OPTS=(
  data=standard/adult_openml
  model=ctgan
  model.params.device=cuda
  preprocess=monotonic
  shadow_model=true
  shadow.num_shadow_rounds=100
  shadow.max_targets=100
  shadow.random_seed=42
  shadow.engine=worker
  shadow.control_branch=true
  'shadow.gpu_ids=[0,1,2,3,4,5,6,7]'
  shadow.worker_concurrency_per_gpu=4
  shadow.reuse_mode=true
  shadow.reuse_num_bases=10
  shadow.reuse_num_out_candidates=10
  shadow.reuse_num_control_candidates=10
)

run_experiment() {
  local exp_id=$1
  local log_file=$2
  shift 2
  echo "=========================================="
  echo "开始实验 $exp_id: $*"
  echo "日志: $log_file"
  echo "=========================================="
  python -m synthgen.train "${BASE_OPTS[@]}" "$@" 2>&1 | tee "$log_file"
  echo "实验 $exp_id 完成."
  echo ""
}

# 1. train_rows=1000 synthetic_rows=1000 n_iter=300 batch_size=256
run_experiment 1 "train_ctgan_shadow_exp1.log" \
  train_rows=1000 synthetic_rows=1000 \
  model.params.n_iter=300 model.params.batch_size=256

# 2. train_rows=500 synthetic_rows=500 n_iter=300 batch_size=256
run_experiment 2 "train_ctgan_shadow_exp2.log" \
  train_rows=500 synthetic_rows=500 \
  model.params.n_iter=300 model.params.batch_size=256

# 3. train_rows=200 synthetic_rows=200 n_iter=1000 batch_size=200
run_experiment 3 "train_ctgan_shadow_exp3.log" \
  train_rows=200 synthetic_rows=200 \
  model.params.n_iter=1000 model.params.batch_size=200

# 4. train_rows=200 synthetic_rows=200 n_iter=2000 batch_size=200
run_experiment 4 "train_ctgan_shadow_exp4.log" \
  train_rows=200 synthetic_rows=200 \
  model.params.n_iter=2000 model.params.batch_size=200

# 5. train_rows=1000 synthetic_rows=1000 n_iter=1000 batch_size=128
run_experiment 5 "train_ctgan_shadow_exp5.log" \
  train_rows=1000 synthetic_rows=1000 \
  model.params.n_iter=1000 model.params.batch_size=128

# 6. train_rows=1000 synthetic_rows=1000 n_iter=300 batch_size=128
run_experiment 6 "train_ctgan_shadow_exp6.log" \
  train_rows=1000 synthetic_rows=1000 \
  model.params.n_iter=300 model.params.batch_size=128

echo "=========================================="
echo "全部 6 组实验已完成."
echo "=========================================="
