#!/usr/bin/env bash
# 四卡任务队列: 用 FIFO 令牌池实现真正的"用完即补", 任一 GPU 空出立刻接下一个任务.
# 用法:
#   bash scripts/gpu_queue.sh joblist_e3.txt            # 每卡 1 个任务
#   PER_GPU=2 bash scripts/gpu_queue.sh joblist_e3.txt  # 每卡 2 个任务 (显存够时吞吐更高)
#   NGPU=4 LOGDIR=logs/e3 bash scripts/gpu_queue.sh joblist_e3.txt
set -uo pipefail
JOBS=${1:?用法: bash scripts/gpu_queue.sh <joblist.txt>}
NGPU=${NGPU:-4}
PER_GPU=${PER_GPU:-1}
LOGDIR=${LOGDIR:-logs}
mkdir -p "$LOGDIR"

FIFO=$(mktemp -u); mkfifo "$FIFO"; exec 9<>"$FIFO"; rm -f "$FIFO"
for ((r=0; r<PER_GPU; r++)); do for ((g=0; g<NGPU; g++)); do echo "$g" >&9; done; done

TOTAL=$(grep -cve '^\s*$' -e '^\s*#' "$JOBS")
echo "[queue] $TOTAL 个任务, $NGPU 卡 × $PER_GPU 并发, 日志 -> $LOGDIR/" | tee -a "$LOGDIR/queue.log"
i=0
while IFS= read -r cmd; do
  [[ -z "${cmd//[[:space:]]/}" ]] && continue
  [[ "$cmd" =~ ^[[:space:]]*# ]] && continue
  i=$((i+1))
  read -u 9 gpu
  (
    tag=$(sed -n 's/.*--out[= ]\([^ ]*\).*/\1/p' <<<"$cmd" | tr '/' '_')
    log="$LOGDIR/$(printf '%03d' "$i")_${tag:-job}_gpu${gpu}.log"
    echo "[$(date '+%F %T')] GPU$gpu <- #$i  $cmd" | tee -a "$LOGDIR/queue.log"
    CUDA_VISIBLE_DEVICES=$gpu bash -c "$cmd" > "$log" 2>&1
    rc=$?
    echo "[$(date '+%F %T')] GPU$gpu -> #$i 完成 rc=$rc  ($log)" | tee -a "$LOGDIR/queue.log"
    echo "$gpu" >&9
  ) &
done < "$JOBS"
wait
echo "[queue] 全部 $i 个任务结束: $(date '+%F %T')" | tee -a "$LOGDIR/queue.log"
grep -c 'rc=0' "$LOGDIR/queue.log" | xargs echo "[queue] 成功计数:"
