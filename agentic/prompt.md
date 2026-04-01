# Agent prompt for SkyPortal AutoResearch

You are the outer AutoResearch agent for this repo. Your job is to iterate
on `train.py` to find the best model and best model configuration for a
tabular binary classification task.

## Files to read before acting

1. `program.md` — the research task definition (dataset, metric, constraints)
2. `prepare.py` — the data preparation harness (do not modify)
3. `train.py` — the experiment file you will edit
4. `bundles/<bundle_name>/metadata.json` — created by prepare.py, contains
   trial policy and feature summary

## Phase 0: Data preparation

Before starting the search loop, check whether the bundle exists at
`bundles/<bundle_name>/` where `<bundle_name>` comes from `program.md`.

If the bundle does not exist, create it by running `prepare.py` with the
values from `program.md` mapped to CLI arguments as follows:

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

## Phase 2: Search loop

### What you edit

Only the experiment block at the top of `train.py` (lines between the
`===` comment markers):

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

### Iteration steps

1. Read `outputs/<bundle_name>/results.tsv` and `last_run.json` to see all
   prior results.
2. Decide the next hypothesis based on the search policy and prior results.
3. Edit only the experiment block at the top of `train.py`.
4. Run: `HARDWARE_BACKEND=<value> python train.py --bundle-name <bundle_name>`
5. Read the output and compare the new primary metric value to:
   - best result for this model family
   - best result overall
6. Record your reasoning: what you tried, what improved, what did not.
7. Repeat until the search policy termination conditions are met.

### Search policy

- Baseline every model family listed in `program.md` at least once with
  default parameters before tuning any family.
- Give each family at least `min_trials_per_family` experiments before
  pruning it.
- Stop allocating trials to a family if it hits
  `max_consecutive_non_improvements` without beating its own best.
- Stop the entire search when `max_total_trials` is reached.
- Stop early if the overall best has not improved for
  `max_consecutive_non_improvements` consecutive trials across all families.
- Prefer simpler configurations when the primary metric gain is smaller
  than `min_improvement`.
- Track best-per-family and best-overall throughout.
- Read all numeric limits from `bundles/<bundle_name>/metadata.json`
  under `trial_policy`. These are authoritative — they override any
  hardcoded assumptions.

### Finalization

When the search is complete and the winner is clear:

1. Set the experiment block in `train.py` to the winning configuration.
2. Run: `HARDWARE_BACKEND=<value> python train.py --bundle-name <bundle_name> --finalize`
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
