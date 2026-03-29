#!/usr/bin/env bash
# 对 6 个 run_dir 按顺序执行 shadow metrics pipeline，可整体 nohup 后台运行。
# 用法: nohup bash scripts/run_shadow_metrics_batch_demo.sh > run_shadow_metrics_batch.log 2>&1 &

set -e
cd "$(dirname "$0")/.."

RUN_DIRS=(
  "outputs/train_adult_openml_preprocess_monotonic_train200_synth200_ctgan_1000iter_bs128_rounds20_candidate100_control_seed42_20260220_201046/"
  "outputs/train_adult_openml_preprocess_monotonic_train200_synth200_ctgan_1000iter_bs128_rounds20_candidate100_control_seed43_20260220_204426/"
  "outputs/train_adult_openml_preprocess_monotonic_train200_synth200_ctgan_1000iter_bs128_rounds20_candidate100_control_seed44_20260220_211718/"
  "outputs/train_adult_openml_preprocess_monotonic_train500_synth500_ctgan_1000iter_bs256_rounds20_candidate100_control_seed42_20260220_223351/"
  "outputs/train_adult_openml_preprocess_monotonic_train500_synth500_ctgan_1000iter_bs256_rounds20_candidate100_control_seed43_20260220_230949/"
  "outputs/train_adult_openml_preprocess_monotonic_train500_synth500_ctgan_1000iter_bs256_rounds20_candidate100_control_seed44_20260220_230509/"
)

N_JOBS=32
MAX_TARGETS=100
MMD_MAX_ROWS=1000

for i in "${!RUN_DIRS[@]}"; do
  run_dir="${RUN_DIRS[$i]}"
  echo "========== [$((i+1))/6] run_shadow_metrics_pipeline: $run_dir =========="
  python scripts/run_shadow_metrics_pipeline.py \
    --run-dir "$run_dir" \
    --n-jobs "$N_JOBS" \
    --max-targets "$MAX_TARGETS" \
    --mmd-max-rows "$MMD_MAX_ROWS"
  echo "========== [$((i+1))/6] 完成 =========="
done

echo "========== 全部 6 个 run_dir 的 pipeline 已按顺序执行完毕 =========="
