#!/usr/bin/env bash
# 对 6 个 run_dir 按顺序执行 shadow metrics pipeline，一键后台运行。
# mmd-max-rows = min(train_rows, 1000)。

set -e
cd "$(dirname "$0")/.."
LOG_DIR="${LOG_DIR:-.}"
mkdir -p "$LOG_DIR"

RUNS=(
  "outputs/train_adult_openml_preprocess_monotonic_train1000_synth1000_ctgan_1000iter_bs128_candidate100_control_20260220_025014/"
  "outputs/train_adult_openml_preprocess_monotonic_train1000_synth1000_ctgan_300iter_bs128_candidate100_control_20260220_063750/"
  "outputs/train_adult_openml_preprocess_monotonic_train200_synth200_ctgan_1000iter_bs200_candidate100_control_20260220_011714/"
  "outputs/train_adult_openml_preprocess_monotonic_train200_synth200_ctgan_2000iter_bs200_candidate100_control_20260220_020314/"
  "outputs/train_adult_openml_preprocess_monotonic_train1000_synth1000_ctgan_300iter_bs256_candidate100_control_20260219_233544/"
  "outputs/train_adult_openml_preprocess_monotonic_train500_synth500_ctgan_300iter_bs256_candidate100_control_20260220_004442/"
)

# run_dir -> mmd-max-rows (min(train_rows, 1000))
MMD_ROWS=(1000 1000 200 200 1000 500)

N_JOBS=32
MAX_TARGETS=100

for i in "${!RUNS[@]}"; do
  run_dir="${RUNS[$i]}"
  mmd="${MMD_ROWS[$i]}"
  log="$LOG_DIR/run_shadow_metrics_$((i+1)).log"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$((i+1))/6] $run_dir (--mmd-max-rows $mmd) -> $log"
  python scripts/run_shadow_metrics_pipeline.py \
    --run-dir "$run_dir" \
    --n-jobs "$N_JOBS" \
    --max-targets "$MAX_TARGETS" \
    --mmd-max-rows "$mmd" \
    >> "$log" 2>&1
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All 6 pipelines done."
