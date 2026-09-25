#!/usr/bin/env bash
# 训练完成后一键跑完 E1-E8 分析与出图 (纯 CPU)
set -e
W=${1:-16}
BF=data/processed/basins_aus.txt
CF=data/processed/core_catchments_aus.csv
MAIN="aus_lstm_nse_s0,aus_gr4j_kge,aus_gr4j_lnnse"

python scripts/make_f1_f2.py
python experiments/e1_perturbation.py  --dataset camels_aus --basins_file $BF --n_basins 60 --workers $W --contaminate 0.15
python experiments/e2_variants.py      --dataset camels_aus --basins_file $BF --runs "$MAIN" --B 200 --workers $W
python experiments/e3_benchmark.py     --dataset camels_aus --regime temporal
python experiments/e3_benchmark.py     --dataset camels_aus --regime pub
python experiments/e4_case_studies.py  --dataset camels_aus --runs "$MAIN" --core_file $CF --B 1000
python experiments/e5_uncertainty.py   --dataset camels_aus --runs "aus_lstm_nse_s0,aus_gr4j_kge" --core_file $CF --B 1000 --workers $W
python experiments/e6_loss_ablation.py --dataset camels_aus --archs lstm,transformer
python experiments/e7_robustness.py    --dataset camels_aus --runs "aus_lstm_nse_s0,aus_gr4j_kge" --basins_file $BF --workers $W
python experiments/e8_decision.py      --dataset camels_aus --regime temporal --workers $W
python scripts/make_paired_figures.py  --all
echo "全部分析完成 -> results/ 与 figures/"
