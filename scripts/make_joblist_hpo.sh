#!/usr/bin/env bash
# 超参随机搜索: 每架构 N 组 (默认 30), 等预算. 只跑 12 轮, 用验证集 MSE 选优.
# 用法: bash scripts/make_joblist_hpo.sh 30 && bash scripts/gpu_queue.sh joblist_hpo.txt
N=${1:-30}
BF=data/processed/basins_aus.txt
OUT=joblist_hpo.txt; : > $OUT
python - "$N" >> $OUT << 'PY'
import sys, numpy as np
n = int(sys.argv[1]); rng = np.random.default_rng(0)
for arch in ["lstm", "gru", "transformer", "patchtst", "tcn"]:
    for k in range(n):
        lr = float(10 ** rng.uniform(-3.7, -2.6))
        hid = int(rng.choice([64, 128, 192, 256]))
        dr = float(rng.choice([0.1, 0.2, 0.3, 0.4, 0.5]))
        bs = int(rng.choice([128, 256, 512]))
        print(f"python models/dl/train.py --arch {arch} --loss nse --seed 0 --regime temporal "
              f"--basins_file data/processed/basins_aus.txt --epochs 12 --patience 4 --subsample 0.2 "
              f"--lr {lr:.5f} --hidden {hid} --dropout {dr} --batch {bs} "
              f"--out runs/hpo/{arch}_{k:02d}")
PY
wc -l $OUT
