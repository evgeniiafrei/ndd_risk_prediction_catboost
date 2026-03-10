[README.md](https://github.com/user-attachments/files/25866963/README.md)
# CatBoost pipeline

A cleaned Python script derived from an exploratory Jupyter notebook for running CatBoost classification with:
- predefined predictor sets
- hyperparameter tuning on a training split
- model selection on a validation split
- final evaluation on a hold-out test split
- feature importance export and plots

The script is intended for datasets already prepared as `train`, `valid`, and `test` tables.

## Workflow

For each predictor set, the script:

1. tunes CatBoost hyperparameters on `train`
2. fits the tuned model on the full `train` split
3. evaluates candidate model on `valid`
4. selects the best overall candidate based on validation performance
5. refits the best model on `train + valid`
6. evaluates the final selected model on `test`

## Input data

The script expects three input files:

- `train`
- `valid`
- `test`

Supported formats:

- `.csv`
- `.parquet`
- `.pkl` / `.pickle`

The files should already contain:

- the binary target column
- all predictor columns needed by the selected predictor set(s)
- the grouping column if grouped CV is used (e.g. for family data)

## Requirements

Main dependencies:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `catboost`

## Usage

Run all predictor sets:

```bash
python clean_catboost_pipeline.py \
  --train path/to/train.csv \
  --valid path/to/valid.csv \
  --test path/to/test.csv \
  --outdir results \
  --predictor-set all
```

Run a single predictor set:

```bash
python clean_catboost_pipeline.py \
  --train path/to/train.csv \
  --valid path/to/valid.csv \
  --test path/to/test.csv \
  --outdir results \
  --predictor-set 4
```

Optional arguments:

- `--target` (default: `F84`)
- `--group-col` (default: `famID`)
- `--random-state` (default: `42`)
- `--n-splits` (default: `5`)

## Outputs

The script writes one set of files per predictor set during model selection, and a separate final set for the selected winner.

Typical outputs:

- `tuning_predictors_<id>.csv`
- `valid_predictions_predictors_<id>.csv`
- `feature_importance_predictors_<id>.csv`
- `feature_importance_predictors_<id>.png`
- `summary_predictors_<id>.json`
- `model_summary.csv`
- `best_model_summary.json`
- `test_predictions_predictors_<best_id>.csv`
- `feature_importance_test_predictors_<best_id>.csv`
- `feature_importance_test_predictors_<best_id>.png`

Example output tree:

```text
results/
├── tuning_predictors_0.csv
├── tuning_predictors_1.csv
├── valid_predictions_predictors_0.csv
├── valid_predictions_predictors_1.csv
├── feature_importance_predictors_0.csv
├── feature_importance_predictors_0.png
├── feature_importance_predictors_1.csv
├── feature_importance_predictors_1.png
├── summary_predictors_0.json
├── summary_predictors_1.json
├── model_summary.csv
├── best_model_summary.json
├── test_predictions_predictors_1.csv
├── feature_importance_test_predictors_1.csv
└── feature_importance_test_predictors_1.png
```

## Notes

- Predictor sets are defined directly inside the script.
- The script assumes preprocessing has already been done before loading the data files.
- Grouped CV is used when the grouping column is available; otherwise stratified CV is used.
