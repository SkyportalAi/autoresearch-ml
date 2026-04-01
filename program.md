# Program: Bank Marketing Campaign Prediction

## Business objective

Predict whether a customer will subscribe to a term deposit based on direct marketing campaign data from a Portuguese banking institution. This helps optimize marketing efforts by identifying high-potential customers.

## Dataset

### Built-in demo dataset

- **source**: `demo`
- **dataset_name**: `bank_marketing`
- **bundle_name**: `bank_marketing`

## Model families to evaluate

- `logistic_regression`
- `random_forest`
- `extra_trees`
- `xgboost`
- `lightgbm`
- `catboost`

## Primary metric

`average_precision`

## Search constraints (optional, defaults shown)

- **max_total_trials**: `20`
- **min_trials_per_family**: `3`
- **max_trials_per_family**: `8`
- **max_consecutive_non_improvements**: `5`
- **min_improvement**: `0.001`
- **per_run_timeout_minutes**: `10`

## Hardware (optional)

- **preferred_backend**: `auto`
