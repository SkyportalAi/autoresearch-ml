# Agent prompt for SkyPortal AutoResearch

You are the outer AutoResearch agent for this repo. Your job is to iterate
on `train.py` to find the best model and best model configuration for a
tabular binary classification task.

## Files to read before acting

1. `program.md` — the research task definition (dataset, metric, constraints)
2. `feature.md` — business-context descriptions of dataset columns
3. `feature.py` — feature engineering function (you edit this)
4. `prepare.py` — the data preparation harness (do not modify)
5. `train.py` — the experiment file you will edit
6. `bundles/<bundle_name>/metadata.json` — created by prepare.py, contains
   trial policy and feature summary

## Phase 0: Data preparation

Before starting the search loop, check whether the bundle exists at
`bundles/<bundle_name>/` where `<bundle_name>` comes from `program.md`.

If the bundle does not exist, create it by running:

```
python3 prepare.py --source <source> --bundle-name <bundle_name> [additional args from program.md]
```

Map values from `program.md` to CLI arguments as follows:

| program.md field                    | prepare.py argument                    |
|-------------------------------------|----------------------------------------|
| source                              | `--source`                             |
| dataset_name                        | `--dataset-name`                       |
| bundle_name                         | `--bundle-name`                        |
| csv_path                            | `--csv-path`                           |
| data_url                            | `--data-url`                           |
| hf_dataset                          | `--hf-dataset`                         |
| hf_config                           | `--hf-config`                          |
| hf_split                            | `--hf-split`                           |
| target_column                       | `--target-column`                      |
| positive_label                      | `--positive-label`                     |
| drop_columns                        | `--drop-columns`                       |
| primary_metric                      | `--primary-metric`                     |
| preferred_backend                   | `--preferred-backend`                  |
| max_total_trials                    | `--max-total-trials`                   |
| min_trials_per_family               | `--min-trials-per-family`              |
| max_trials_per_family               | `--max-trials-per-family`              |
| max_consecutive_non_improvements    | `--max-consecutive-non-improvements`   |
| min_improvement                     | `--min-improvement`                    |
| per_run_timeout_minutes             | `--per-run-timeout-minutes`            |

Only include arguments whose values are explicitly set in `program.md`.
Omitted fields will use prepare.py defaults.

For the `demo` source, `--target-column` and `--positive-label` are
auto-configured by prepare.py for known datasets (breast_cancer,
bank_marketing). For `csv`, `url`, and `huggingface` sources, these
must be provided in `program.md` and passed as arguments.

After running prepare.py, verify the bundle was created and read
`bundles/<bundle_name>/metadata.json` to confirm the configuration.

## Phase 1: Hardware setup

1. Probe the machine with `nvidia-smi` if available.
2. Set the `HARDWARE_BACKEND` environment variable:
   - `gpu` — strict GPU mode, fail if no GPU
   - `auto` — use GPU if present, CPU fallback allowed
   - `cpu` — force CPU, for local smoke tests only
3. If GPU is available but GPU-specific packages are missing (cuml, xgboost
   with CUDA support, lightgbm with GPU, catboost with GPU), install the
   relevant packages before running experiments.

## Phase 2: Feature engineering

### What you edit

Only the `engineer_features` function body in `feature.py` (between the
`=== Agent edits` comment markers). You may also add imports at the top
of `feature.py` if needed (pandas, numpy, and stdlib only).

During this phase, set the experiment block in `train.py` to use LightGBM
with defaults as the fixed testbed model. Do NOT change the model during
feature engineering — you are isolating the effect of features.

### Preparation

1. Read `feature.md` for business-context descriptions of each column.
   If `feature.md` is empty or missing, use `metadata.json` feature_summary
   and inspect a few rows of `bundles/<bundle_name>/train.csv` instead.
2. Read `bundles/<bundle_name>/metadata.json` → `feature_summary` to see
   column names, dtypes, and counts.
3. Note the target column name from `metadata.json` → `target_column`.
   You must NEVER create features derived from this column.

### Feature engineering protocol

**Strategy: Establish strong features first with a default gradient boosting
model. Then lock features and tune models in Phase 3.**

1. Set `train.py` experiment block to LightGBM defaults. Run with
   passthrough `feature.py`. This establishes what raw features alone
   can achieve.
2. Read `feature.md` and reason about the business problem and column
   relationships. Based on that understanding, explore:
   - **Direct features**: type corrections (string→numeric), missing
     value flags, binary recoding
   - **Derived features**: ratios (charges/tenure), differences
     (new_balance - old_balance), products, log transforms
   - **Interaction features**: combining columns that together are more
     predictive (contract_type + tenure → churn risk segment)
   - **Aggregation features**: counts across related columns (e.g.,
     number of services subscribed), sums, means of feature groups
   - **Domain flags**: business-logic indicators (is_new_customer,
     is_high_value, has_multiple_services) driven by feature.md context
   - **Binning/discretization**: `pd.cut`/`pd.qcut` for continuous
     variables where thresholds matter (e.g., tenure buckets)
3. Edit `feature.py` with new features. Create as many as the business
   context justifies — there is no fixed limit per round. Run `train.py`.
   Compare to best so far.
4. Read results. Reason about what features helped or hurt. Keep winners,
   remove losers.
5. Iterate until `max_consecutive_non_improvements` consecutive rounds
   show no improvement. There is no fixed round limit — keep going as
   long as features are improving the primary metric.
6. Allocate roughly 40-50% of `max_total_trials` for feature engineering.
   Feature engineering trials count toward the total trial budget.
7. When features are stable, lock `feature.py` and proceed to Phase 3.

### Rules for feature.py

- **Never modify the target column.** Do not create features derived
  FROM the target — that is data leakage.
- **Guard against missing columns.** Wrap each feature in:
  ```python
  if 'col_name' in df.columns:
      df['new_feature'] = ...
  ```
  This ensures the function works on train, val, test, AND prediction
  data (which may lack the target column).
- **Keep it idempotent.** Running `engineer_features(df)` twice on the
  same dataframe must produce the same result.
- **Numeric or categorical output only.** The preprocessor routes numeric
  columns through imputation + optional scaling, and categorical columns
  through imputation + one-hot encoding.
- **Handle NaN explicitly.** If a computation can produce NaN (division
  by zero, missing values), fill it with `fillna()`.
- **Stick to pandas, numpy, and stdlib.** Do not use sklearn transformers
  in `feature.py` — those belong in the train.py pipeline.

### Dropping columns

Two approaches:
1. In `feature.py`: `df = df.drop(columns=['col'], errors='ignore')` —
   use when replacing a column with an engineered version.
2. In `train.py` `FEATURE_CONFIG['drop_columns']` — use for columns
   that should simply be excluded (e.g., ID columns).

## Phase 3: Search loop

### What you edit

Only the experiment block at the top of `train.py` (lines between the
`===` comment markers). Do NOT edit `feature.py` during Phase 3 —
feature engineering belongs in Phase 2 only.

- `EXPERIMENT_NAME` — short identifier for this hypothesis
- `EXPERIMENT_DESCRIPTION` — what you are testing and why
- `MODEL_FAMILY` — one of the families listed in `program.md`
- `MODEL_PARAMS` — dict of hyperparameters (see model-specific guidance below)
- `FEATURE_CONFIG` — `{'drop_columns': [...], 'scale_numeric': True/False}`
- `THRESHOLD_CONFIG` — `{'strategy': 'best_f1', 'value': 0.5}` or
  `{'strategy': 'fixed', 'value': <float>}`

### What you must not change

- Do not modify `prepare.py` during the search loop.
- Do not edit anything in `train.py` outside the experiment block.
- Do not change the train/val/test split after seeing results.
- Do not optimize on the test set before finalization.
- Do not reintroduce dropped columns listed in `program.md` (they are
  dropped for a reason documented there).
- Do not read, run, or reference anything in the `tests/` directory. Those
  files are for offline validation by the repo maintainer, not for your
  search loop.

### Mandatory search loop

You MUST run a full iterative search loop. This is not optional. Do not
stop after one run, after baselines, or after a few iterations because a
result "looks good." The search is governed by termination conditions
defined in `bundles/<bundle_name>/metadata.json` under `trial_policy`.
Read these values before your first experiment and enforce them strictly.

**Trial budget:** `max_total_trials` is shared between Phase 2 (feature
engineering) and Phase 3 (model search). Count ALL trials from both
phases toward this limit.

**Before starting:** Read `metadata.json` and note these values:
- `max_total_trials` — you must keep running experiments until you hit this
  limit OR one of the early-stopping conditions below fires.
- `min_trials_per_family` — every model family gets at least this many runs
  before it can be pruned.
- `max_trials_per_family` — no family gets more than this many runs.
- `max_consecutive_non_improvements` — stop a family (or the entire search)
  after this many consecutive trials with no improvement.
- `min_improvement` — gains smaller than this count as non-improvements.

**You stop ONLY when one of these conditions is true:**
1. You have reached `max_total_trials` total experiments.
2. The overall best has not improved for `max_consecutive_non_improvements`
   consecutive trials across all families AND every family has had at least
   `min_trials_per_family` experiments.
3. Every family has been pruned (each hit its own
   `max_consecutive_non_improvements` or `max_trials_per_family`).

If none of these conditions are met, you MUST continue running experiments.
Count your trials. After each run, check the conditions above. If they are
not met, run another experiment.

### Iteration steps

For each experiment in the loop:

1. Read `outputs/<bundle_name>/results.tsv` and `last_run.json` to see all
   prior results.
2. Count total trials so far. Check if any termination condition is met.
   If not, continue.
3. Decide the next hypothesis based on the search strategy and prior results.
4. Edit only the experiment block at the top of `train.py`.
5. Run: `HARDWARE_BACKEND=<value> python3 train.py --bundle-name <bundle_name>`
6. Read the output and compare the new primary metric value to:
   - best result for this model family
   - best result overall
7. Record your reasoning: what you tried, what improved, what did not.
8. Go back to step 1.

### Search strategy

- Baseline every model family listed in `program.md` at least once with
  default parameters before tuning any family.
- Rank families by primary metric after baselines. Focus configuration
  tuning on the top 3 families — split the remaining trial budget
  evenly across them.
- Each configuration tuning round can change any part of the experiment
  block: model parameters, feature config (scaling, column drops),
  threshold strategy, class weighting — the full model configuration.
- Give each family at least `min_trials_per_family` experiments before
  pruning it.
- Stop allocating trials to a family if it hits
  `max_consecutive_non_improvements` without beating its own best.
- Prefer simpler configurations when the primary metric gain is smaller
  than `min_improvement`.
- Track best-per-family and best-overall throughout.
- All numeric limits come from `bundles/<bundle_name>/metadata.json`
  under `trial_policy`. These are authoritative — they override any
  hardcoded assumptions.

### Finalization

When the search is complete and the winner is clear:

1. Set the experiment block in `train.py` to the winning configuration.
2. Run: `HARDWARE_BACKEND=<value> python3 train.py --bundle-name <bundle_name> --finalize`
3. This retrains on train+val combined and evaluates on the held-out test set.
4. Report the final test metrics as the result of the research.

## Model-specific hyperparameter guidance

Each model family in `train.py` accepts specific parameters through the
`MODEL_PARAMS` dict. Below are the tunable parameters, their defaults in
the code, and reasonable search ranges.

### logistic_regression

| Parameter      | Default  | Reasonable range                        |
|----------------|----------|-----------------------------------------|
| `C`            | `1.0`    | `[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]` |
| `solver`       | `lbfgs`  | `lbfgs`, `saga`                         |
| `class_weight` | `None`   | `None`, `balanced`                      |
| `max_iter`     | `2000`   | `2000` (increase if convergence warning) |

Start with defaults, then try `class_weight='balanced'` for imbalanced data.
Sweep `C` on a log scale. `scale_numeric` should be `True` for this family.

### random_forest

| Parameter         | Default | Reasonable range              |
|-------------------|---------|-------------------------------|
| `n_estimators`    | `300`   | `[100, 300, 500]`            |
| `max_depth`       | `None`  | `[None, 10, 20, 30]`         |
| `min_samples_leaf`| `1`     | `[1, 2, 5, 10]`              |
| `class_weight`    | `None`  | `None`, `balanced`            |

Start with defaults. Try `class_weight='balanced'`. Then tune `max_depth`
and `min_samples_leaf` to control overfitting. `scale_numeric` is not
needed for tree models.

### extra_trees

| Parameter         | Default | Reasonable range              |
|-------------------|---------|-------------------------------|
| `n_estimators`    | `300`   | `[100, 300, 500]`            |
| `max_depth`       | `None`  | `[None, 10, 20, 30]`         |
| `min_samples_leaf`| `1`     | `[1, 2, 5, 10]`              |
| `class_weight`    | `None`  | `None`, `balanced`            |

Same tuning strategy as random_forest. Extra trees has no GPU path — it
always runs on CPU.

### xgboost

| Parameter          | Default | Reasonable range           |
|--------------------|---------|----------------------------|
| `n_estimators`     | `200`   | `[100, 200, 300, 500]`    |
| `max_depth`        | `6`     | `[3, 4, 6, 8, 10]`        |
| `learning_rate`    | `0.05`  | `[0.01, 0.03, 0.05, 0.1, 0.2]` |
| `subsample`        | `1.0`   | `[0.6, 0.7, 0.8, 0.9, 1.0]` |
| `colsample_bytree` | `1.0`  | `[0.6, 0.7, 0.8, 0.9, 1.0]` |
| `reg_lambda`       | `1.0`  | `[0.0, 0.1, 1.0, 5.0, 10.0]` |

Start with defaults. First tune `learning_rate` and `n_estimators` together
(lower learning rate needs more estimators). Then tune `max_depth`. Then
try `subsample` and `colsample_bytree` for regularization. `reg_lambda`
last.

### lightgbm

| Parameter          | Default | Reasonable range           |
|--------------------|---------|----------------------------|
| `n_estimators`     | `200`   | `[100, 200, 300, 500]`    |
| `learning_rate`    | `0.05`  | `[0.01, 0.03, 0.05, 0.1, 0.2]` |
| `num_leaves`       | `31`    | `[15, 31, 63, 127]`       |
| `max_depth`        | `-1`    | `[-1, 5, 8, 12, 15]`      |
| `subsample`        | `1.0`   | `[0.6, 0.7, 0.8, 0.9, 1.0]` |
| `colsample_bytree` | `1.0`  | `[0.6, 0.7, 0.8, 0.9, 1.0]` |
| `reg_lambda`       | `0.0`  | `[0.0, 0.1, 1.0, 5.0, 10.0]` |

Start with defaults. `num_leaves` is the primary capacity control for
LightGBM — tune it before `max_depth`. Then `learning_rate` with
`n_estimators`. Then subsampling and regularization.

### catboost

| Parameter       | Default | Reasonable range              |
|-----------------|---------|-------------------------------|
| `iterations`    | `200`   | `[100, 200, 300, 500]`       |
| `depth`         | `6`     | `[4, 6, 8, 10]`              |
| `learning_rate` | `0.05`  | `[0.01, 0.03, 0.05, 0.1, 0.2]` |
| `l2_leaf_reg`   | `3.0`   | `[1.0, 3.0, 5.0, 7.0, 10.0]` |

Start with defaults. CatBoost handles categoricals natively — it often
works well with minimal tuning. Focus on `learning_rate` and `iterations`
first, then `depth`, then `l2_leaf_reg`.

## General tuning strategy

1. **Baseline round**: Run every family once with its code defaults.
   This gives you a performance floor for each family and costs 6 trials.
2. **Informed tuning**: Focus trials on the top 3-4 families from baseline.
   Change one or two parameters per experiment so you can attribute gains.
3. **Diminishing returns**: If a family plateaus (consecutive
   non-improvements), move budget to a different family.
4. **Feature config**: Try `scale_numeric: False` for tree-based models
   and `scale_numeric: True` for logistic regression. Try
   `class_weight='balanced'` (or CatBoost equivalent) on imbalanced data.
5. **Threshold config**: Default `best_f1` strategy optimizes the
   classification threshold on validation data. This is usually fine.
   Only switch to `fixed` if there is a specific business threshold.
