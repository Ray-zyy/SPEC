# Curated result tables

This directory contains the key numerical outputs extracted from the supplied E1--E8 experiment archive. Large training checkpoints, prediction parquet files, temporary reports, and redundant intermediate tables are intentionally excluded.

| Experiment | Main files | Purpose |
|---|---|---|
| E1 | `e1_identifiability.csv`, `e1_metrics.csv` | Controlled perturbations and identifiability |
| E2 | `e2_a_undefined_by_zero_stratum.csv`, `e2_bcd_stability.csv`, `e2_e_branch_correlation.csv` | Kernel definedness and stability |
| E3 | `e3_summary_*`, `T3_summary_*`, `e3_ranking_flips_*`, `e3_friedman_*`, `e3_attribute_rf_*` | Large-sample model comparison and ranking disagreement |
| E4 | `e4_crosscheck.csv` | Catchment-scale hydrological cross-checks |
| E5 | `e5_a_band_width.csv`, `e5_b_convergence.csv`, `e5_c_paired_significance.csv`, `e5_d_rating_uncertainty.csv` | Sampling uncertainty and record-length sensitivity |
| E6 | `e6_pareto.csv`, `e6_summary.csv` | Training-objective ablation |
| E7 | `e7_sensitivity_summary_camels_aus.csv`, `e7_h_timing.csv` | Implementation robustness and runtime |
| E8 | `e8_regressions_camels_aus_temporal.csv` | Decision-relevance regressions |

The CSVs are frozen outputs from the supplied experiment bundle. Re-run the scripts in `experiments/` to regenerate results from raw data and model predictions.
