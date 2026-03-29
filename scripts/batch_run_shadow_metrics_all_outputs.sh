#!/usr/bin/env bash
# 对 outputs/ 下所有文件夹执行 shadow metrics pipeline（与 run_commands.txt 90-95 一致）
# 规则：
#   1) 若已存在 shadow_pair_metrics/ → 跳过（说明之前执行过）
#   2) 若 shadow/ 下 target 数量 < 100 → 记录到 insufficient_targets.txt，不执行
#   3) 其余目录才执行（后台运行，各自独立日志）

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUTS_DIR="${PROJECT_ROOT}/outputs"
LOG_DIR="${PROJECT_ROOT}/logs_shadow_metrics_batch"
REQUIRED_TARGETS=100
N_JOBS=32
MAX_TARGETS=100
MMD_MAX_ROWS=1000

mkdir -p "$LOG_DIR"
INSUFFICIENT_LOG="${LOG_DIR}/insufficient_targets.txt"
: > "$INSUFFICIENT_LOG"

# 已执行过：存在 shadow_pair_metrics/ 目录（且有关键输出更稳妥）
is_already_done() {
  local run_dir="$1"
  [[ -d "${run_dir}/shadow_pair_metrics" ]]
}

# 统计 shadow/ 下 target_* 目录数量（只计目录，且名为 target_数字）
count_shadow_targets() {
  local run_dir="$1"
  local shadow_dir="${run_dir}/shadow"
  local count=0
  if [[ ! -d "$shadow_dir" ]]; then
    echo 0
    return
  fi
  for d in "$shadow_dir"/target_*; do
    if [[ -d "$d" ]]; then
      name="${d##*/}"
      suffix="${name#target_}"
      if [[ "$suffix" =~ ^[0-9]+$ ]]; then
        ((count++)) || true
      fi
    fi
  done
  echo "$count"
}

cd "$PROJECT_ROOT"
if [[ ! -d "$OUTPUTS_DIR" ]]; then
  echo "outputs 目录不存在: $OUTPUTS_DIR"
  exit 1
fi

total=0
skipped_done=0
skipped_insufficient=0
launched=0

echo "扫描 outputs/ 下所有目录，未执行且 shadow 至少 ${REQUIRED_TARGETS} 个 target 的才执行 pipeline"
echo "日志目录: $LOG_DIR"
echo "target 不足的记录: $INSUFFICIENT_LOG"
echo ""

for run_dir in "$OUTPUTS_DIR"/*/; do
  [[ -d "$run_dir" ]] || continue
  name="$(basename "$run_dir")"
  ((total++)) || true

  if is_already_done "$run_dir"; then
    echo "[跳过-已完成] $name"
    ((skipped_done++)) || true
    continue
  fi

  n_targets=$(count_shadow_targets "$run_dir")
  if [[ "$n_targets" -lt "$REQUIRED_TARGETS" ]]; then
    echo "[跳过-target不足] $name (shadow 下 target 数: $n_targets, 需要 >= $REQUIRED_TARGETS)"
    echo "$name	targets=$n_targets	required=$REQUIRED_TARGETS" >> "$INSUFFICIENT_LOG"
    ((skipped_insufficient++)) || true
    continue
  fi

  log_file="${LOG_DIR}/${name}.log"
  echo "[启动] $name (targets=$n_targets)"
  nohup python scripts/run_shadow_metrics_pipeline.py \
    --run-dir "$run_dir" \
    --n-jobs "$N_JOBS" \
    --max-targets "$MAX_TARGETS" \
    --mmd-max-rows "$MMD_MAX_ROWS" \
    >> "$log_file" 2>&1 &
  ((launched++)) || true
  sleep 2
done

echo ""
echo "汇总: 共 $total 个目录 | 已存在 shadow_pair_metrics 跳过 $skipped_done | target 不足跳过 $skipped_insufficient | 新启动 $launched 个"
echo "target 不足列表见: $INSUFFICIENT_LOG"
echo "查看进度: tail -f $LOG_DIR/<run_dir_name>.log"
