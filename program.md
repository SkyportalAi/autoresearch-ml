# Program: <Your Task Title>

## Business objective

<!-- Describe what you are trying to predict and why. -->

## Dataset

<!-- Pick ONE source type and fill in the relevant fields. Delete the rest. -->

### Option A: Built-in demo dataset

- **source**: `demo`
- **dataset_name**: `breast_cancer` or `bank_marketing`
- **bundle_name**: `<your_bundle_name>`

### Option B: Local CSV file

- **source**: `csv`
- **csv_path**: `/path/to/your/data.csv`
- **bundle_name**: `<your_bundle_name>`
- **target_column**: `<column_name>`
- **positive_label**: `<value>` (only needed if target is not already 0/1)

### Option C: Public URL (CSV or ZIP containing CSV)

- **source**: `url`
- **data_url**: `https://example.com/data.csv`
- **bundle_name**: `<your_bundle_name>`
- **target_column**: `<column_name>`
- **positive_label**: `<value>`

### Option D: Hugging Face dataset

- **source**: `huggingface`
- **hf_dataset**: `org/dataset-name`
- **hf_split**: `train`
- **bundle_name**: `<your_bundle_name>`
- **target_column**: `<column_name>`
- **positive_label**: `<value>`

### Columns to drop (optional)

- **drop_columns**: `<col1>`, `<col2>`
- **drop_reason**: Explain why these columns should be excluded (e.g., leakage, ID columns).

## Model families to evaluate

- `logistic_regression`
- `random_forest`
- `extra_trees`
- `xgboost`
- `lightgbm`
- `catboost`

<!-- Remove any families you do not want the agent to evaluate. -->

## Primary metric

`average_precision`

<!-- Supported: average_precision, roc_auc, f1, accuracy, neg_log_loss -->

## Search constraints (optional, defaults shown)

<!-- Only include lines you want to override. Omitted values use prepare.py defaults. -->

- **max_total_trials**: `20`
- **min_trials_per_family**: `3`
- **max_trials_per_family**: `8`
- **max_consecutive_non_improvements**: `5`
- **min_improvement**: `0.001`
- **per_run_timeout_minutes**: `10`

## Hardware (optional)

- **preferred_backend**: `auto`

<!-- Supported: auto, gpu, cpu -->
