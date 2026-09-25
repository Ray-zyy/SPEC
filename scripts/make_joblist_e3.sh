#!/usr/bin/env bash
# E3 主实验训练清单: 5 架构 × 8 种子 (temporal) + 5 架构 × 5 折 (PUB) = 65 个单卡任务
BF=data/processed/basins_aus.txt
EP=${EPOCHS:-30}
SUB=${SUBSAMPLE:-0.35}
OUT=joblist_e3.txt; : > $OUT
for ARCH in lstm gru transformer patchtst tcn; do
  for SEED in 0 1 2 3 4 5 6 7; do
    echo "python models/dl/train.py --arch $ARCH --loss nse --seed $SEED --regime temporal --basins_file $BF --epochs $EP --subsample $SUB --out runs/aus_${ARCH}_nse_s${SEED}" >> $OUT
  done
done
for ARCH in lstm gru transformer patchtst tcn; do
  for FOLD in 0 1 2 3 4; do
    echo "python models/dl/train.py --arch $ARCH --loss nse --seed 0 --regime pub --fold $FOLD --basins_file $BF --epochs $EP --subsample $SUB --out runs/aus_${ARCH}_nse_pub${FOLD}" >> $OUT
  done
done
wc -l $OUT
