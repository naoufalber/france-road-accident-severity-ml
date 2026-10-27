# Original research archive

The six numbered Python scripts and [French project report](original_report_fr.pdf) are preserved unchanged from the December 2025 group submission. They show the original progression: data loading, exploration, feature engineering, Random Forest, Optuna and SHAP.

**Use `../src/train.py` for the corrected, portable benchmark.** Historical scripts are reference material, not the supported entry point.

## How to interpret historical results

The report advertises approximately **79% accuracy and 81% F1**. These are historical claims and were **not reproduced or validated as severe-injury performance** during repository preparation:

1. `03_feature_engineering.py` uses `grav >= 3`, which includes hospitalized and slightly injured people, but excludes fatalities. The intended severe target is `grav in {2, 3}`. The report's prose also reverses the meanings of codes 2 and 3, although both belong in the intended positive class.
2. Joining every location row to every user multiplies observations for 2023 in the supplied snapshot. The corrected pipeline checks many-to-one joins and preserves user counts.
3. Random user-level splits can place people from the same accident in both training and test. The new evaluation holds out the full year 2023.
4. Optuna uses a validation split, then evaluates with a different random split of the same full data, which does not preserve an untouched final test set.
5. Numeric category codes are treated as ordered in the original model. The corrected pipeline uses one-hot encoding fitted only on training data.

The optional dependencies are listed in `../requirements-research.txt`. Original scripts assume a working directory with sibling `data_raw/` and `data_interim/` folders and interactive plots. **Optuna and SHAP were not rerun**; their historical findings cannot be attributed to the corrected model.
