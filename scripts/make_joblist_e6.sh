#!/usr/bin/env bash
# E6 损失消融: {lstm, transformer} × 5 损失 × 4 种子 = 40 个单卡任务
BF=data/processed/basins_aus.txt
EP=${EPOCHS:-30}
SUB=${SUBSAMPLE:-0.35}
OUT=joblist_e6.txt; : > $OUT
for ARCH in lstm transformer; do
  for LOSS in mse mae nse logmse delta; do
    for SEED in 0 1 2 3; do
      echo "python models/dl/train.py --arch $ARCH --loss $LOSS --seed $SEED --regime temporal --basins_file $BF --epochs $EP --subsample $SUB --out runs/aus_${ARCH}_${LOSS}_s${SEED}" >> $OUT
    done
  done
done
wc -l $OUT
