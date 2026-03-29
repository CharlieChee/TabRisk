#!/usr/bin/env bash
# 根据 run_exp_A1_A4.log 里已出现的 "Done:" 行，为对应 run 打上 .done 标记，
# 然后执行 run_exp_A1_A4_demo.sh，只补跑未完成的 run。
#
# 可执行权限（可选）：
#   chmod +x scripts/resume_exp_A1_A4_from_log.sh
# 用法（在项目根目录执行）：
#   bash scripts/resume_exp_A1_A4_from_log.sh
# 或后台续跑：
#   nohup bash scripts/resume_exp_A1_A4_from_log.sh >> run_exp_A1_A4.log 2>&1 &

set -e
cd "$(dirname "$0")/.."
LOG_DIR="${LOG_DIR:-logs_exp_A1_A4}"
mkdir -p "$LOG_DIR"
MAIN_LOG="run_exp_A1_A4.log"

if [ -f "$MAIN_LOG" ]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 从 $MAIN_LOG 读取已完成的 run 并打 .done 标记..."
  while IFS= read -r line; do
    # 解析 "Done:  A1 train_rows=200 seed=42" 形式的行（兼容不同空格）
    if [[ "$line" =~ Done:[[:space:]]+([A-Z0-9]+)[[:space:]]+train_rows=([0-9]+)[[:space:]]+seed=([0-9]+) ]]; then
      exp_id="${BASH_REMATCH[1]}"
      train_rows="${BASH_REMATCH[2]}"
      seed="${BASH_REMATCH[3]}"
      done_file="$LOG_DIR/${exp_id}_rows${train_rows}_seed${seed}.done"
      if [ ! -f "$done_file" ]; then
        touch "$done_file"
        echo "  标记已完成: $exp_id train_rows=$train_rows seed=$seed"
      fi
    fi
  done < <(grep "Done:" "$MAIN_LOG" || true)
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 标记完成，开始续跑（未完成的 run）..."
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 未找到 $MAIN_LOG，直接执行完整 demo（无跳过）。"
fi

exec bash scripts/run_exp_A1_A4_demo.sh
