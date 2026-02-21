#!/usr/bin/env bash
# 批量对多个 run_dir 执行 shadow metrics pipeline，后台运行，各自独立日志

RUN_DIRS=(
  "train_adult_openml_preprocess_monotonic_train500_synth500_ctgan_1000iter_bs256_rounds10_candidate100_control_seed42_20260220_212035"
  "train_adult_openml_preprocess_monotonic_train500_synth500_ctgan_1000iter_bs256_rounds10_candidate100_control_seed43_20260220_215606"
  "train_adult_openml_preprocess_monotonic_train500_synth500_ctgan_1000iter_bs256_rounds160_candidate100_control_seed42_20260221_023303"
  "train_adult_openml_preprocess_monotonic_train500_synth500_ctgan_1000iter_bs256_rounds20_candidate100_control_seed42_20260220_155728"
  "train_adult_openml_preprocess_monotonic_train500_synth500_ctgan_1000iter_bs256_rounds20_candidate100_control_seed42_20260220_215113"
  "train_adult_openml_preprocess_monotonic_train500_synth500_ctgan_1000iter_bs256_rounds20_candidate100_control_seed42_20260220_223351"
  "train_adult_openml_preprocess_monotonic_train500_synth500_ctgan_1000iter_bs256_rounds20_candidate100_control_seed42_20260221_074449"
  "train_adult_openml_preprocess_monotonic_train500_synth500_ctgan_1000iter_bs256_rounds20_candidate100_control_seed43_20260220_163409"
  "train_adult_openml_preprocess_monotonic_train500_synth500_ctgan_1000iter_bs256_rounds20_candidate100_control_seed43_20260220_222644"
  "train_adult_openml_preprocess_monotonic_train500_synth500_ctgan_1000iter_bs256_rounds20_candidate100_control_seed43_20260220_230949"
  "train_adult_openml_preprocess_monotonic_train500_synth500_ctgan_1000iter_bs256_rounds20_candidate100_control_seed43_20260221_081850"
  "train_adult_openml_preprocess_monotonic_train500_synth500_ctgan_1000iter_bs256_rounds20_candidate100_control_seed44_20260220_171311"
  "train_adult_openml_preprocess_monotonic_train500_synth500_ctgan_1000iter_bs256_rounds20_candidate100_control_seed44_20260220_230509"
  "train_adult_openml_preprocess_monotonic_train500_synth500_ctgan_1000iter_bs256_rounds40_candidate100_control_seed42_20260220_234635"
  "train_adult_openml_preprocess_monotonic_train500_synth500_ctgan_1000iter_bs256_rounds40_candidate100_control_seed43_20260221_002202"
  "train_adult_openml_preprocess_monotonic_train500_synth500_ctgan_1000iter_bs256_rounds5_candidate100_control_seed42_20260220_200431"
  "train_adult_openml_preprocess_monotonic_train500_synth500_ctgan_1000iter_bs256_rounds5_candidate100_control_seed43_20260220_204152"
  "train_adult_openml_preprocess_monotonic_train500_synth500_ctgan_1000iter_bs256_rounds80_candidate100_control_seed42_20260221_011958"
  "train_adult_openml_preprocess_monotonic_train500_synth500_ctgan_1000iter_bs256_rounds80_candidate100_control_seed43_20260221_015556"
  "train_adult_openml_preprocess_monotonic_train500_synth500_ctgan_2000iter_bs256_rounds20_candidate100_control_seed42_20260221_085440"
  "train_adult_openml_preprocess_monotonic_train500_synth500_ctgan_2000iter_bs256_rounds20_candidate100_control_seed43_20260221_093805"
  "train_adult_openml_preprocess_monotonic_train500_synth500_ctgan_200iter_bs256_rounds20_candidate100_control_seed42_20260221_064151"
  "train_adult_openml_preprocess_monotonic_train500_synth500_ctgan_300iter_bs256_rounds20_candidate100_control_seed42_20260221_065205"
  "train_adult_openml_preprocess_monotonic_train500_synth500_ctgan_300iter_bs256_rounds20_candidate100_control_seed43_20260221_070610"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="${PROJECT_ROOT}/logs_shadow_metrics_batch"
mkdir -p "$LOG_DIR"

echo "批量启动 shadow metrics pipeline，共 ${#RUN_DIRS[@]} 个 run_dir"
echo "日志目录: $LOG_DIR"
echo ""

for i in "${!RUN_DIRS[@]}"; do
  name="${RUN_DIRS[$i]}"
  run_dir="outputs/${name}/"
  log_file="${LOG_DIR}/${name}.log"
  echo "[$((i+1))/${#RUN_DIRS[@]}] 启动: $name"
  nohup python scripts/run_shadow_metrics_pipeline.py \
    --run-dir "$run_dir" \
    --n-jobs 32 \
    --max-targets 100 \
    --mmd-max-rows 1000 \
    >> "$log_file" 2>&1 &
  sleep 2
done

echo ""
echo "已全部在后台启动。查看进度: tail -f $LOG_DIR/<run_dir_name>.log"
