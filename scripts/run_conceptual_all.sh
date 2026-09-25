#!/usr/bin/env bash
# 概念模型 + 基准 (纯 CPU 多进程, 与 GPU 训练同时跑, 互不冲突)
set -e
BF=data/processed/basins_aus.txt
W=${1:-32}
MAXN=${MAXN:-8000}
for M in gr4j hbv; do
  for OBJ in kge lnnse; do
    echo "=== $M / $OBJ ==="
    python models/conceptual/run_conceptual.py --dataset camels_aus --model $M --obj $OBJ \
        --basins_file $BF --out runs/aus_${M}_${OBJ} --workers $W --maxn $MAXN
  done
done
python models/conceptual/run_conceptual.py --dataset camels_aus --model benchmarks \
    --basins_file $BF --out runs/aus_bench --workers $W
echo "概念模型与基准完成"
