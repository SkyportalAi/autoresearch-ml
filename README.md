# SkyPortal AutoResearch for GPU-first Tabular ML

Inspired by Andrej Karpathy's [AutoResearch](https://github.com/karpathy/autoresearch),
which uses an LLM agent to iteratively improve LLM training runs, this repo
applies the same pattern to **classical machine learning on tabular data**.

Instead of tuning LLM hyperparameters, this repo lets a data scientist
define a business objective (e.g., predict customer churn, score fraud risk),
point it at a dataset, and choose which model families to evaluate. An LLM
agent then takes over — systematically exploring the hyperparameter
configuration space across all selected models, running experiments,
comparing results, and narrowing down to the single best model with its
best configuration. The data scientist defines *what* to solve and *which
models* to consider; the agent figures out the *how*.

Supported model families: logistic regression, random forest, extra trees,
XGBoost, LightGBM, and CatBoost — with GPU acceleration where available
and automatic CPU fallback.

## How it works

```
program.md          The data scientist fills in the task: dataset, metric, models.
    |
    v
prompt.md           The agent reads its operating instructions and guardrails.
    |
    v
prepare.py          The agent runs this once to create an immutable data bundle.
    |
    v
train.py            The agent edits the experiment block, runs, reads results,
    |               and repeats for up to max_total_trials iterations.
    v
predict.py          Score new data with the finalized winning model.
```

## Key files

| File | Role | Who edits it |
|------|------|--------------|
| `program.md` | Task definition: dataset, metric, models, constraints | You (the data scientist) |
| `agentic/prompt.md` | Agent instructions: how to run prepare.py, search policy, model tuning guidance | Repo maintainer |
| `prepare.py` | Data preparation harness: loads data, splits, writes bundle | Nobody during search |
| `train.py` | Single-experiment trainer with an editable config block at the top | The agent (experiment block only) |
| `predict.py` | Score new data with a finalized model | Run manually after research |
| `reset.py` | Reset train.py and program.md to clean starting state | You (between research tasks) |

## Quick start

### 1. Reset to a clean state (if starting fresh)

```bash
python reset.py --program
```

This resets `train.py` to its default experiment block and `program.md` to
the blank template. Run this before starting a new research task.

### 2. Fill in program.md

Open `program.md` and fill in your task. Pick a data source:

**Built-in demo dataset:**
```markdown
- **source**: `demo`
- **dataset_name**: `bank_marketing`
- **bundle_name**: `bank_marketing`
```

**Local CSV:**
```markdown
- **source**: `csv`
- **csv_path**: `/path/to/data.csv`
- **bundle_name**: `my_task`
- **target_column**: `converted`
- **positive_label**: `yes`
```

**Public URL:**
```markdown
- **source**: `url`
- **data_url**: `https://example.com/data.csv`
- **bundle_name**: `my_task`
- **target_column**: `converted`
- **positive_label**: `yes`
```

**Hugging Face:**
```markdown
- **source**: `huggingface`
- **hf_dataset**: `org/dataset-name`
- **hf_split**: `train`
- **bundle_name**: `my_task`
- **target_column**: `label`
- **positive_label**: `1`
```

Choose which model families to evaluate, set your primary metric, and
optionally adjust search constraints. See the template comments for details.

### 3. Point your agent at the repo

The agent reads `program.md` and `agentic/prompt.md`, then:

1. Runs `prepare.py` to create the data bundle (if it doesn't exist)
2. Baselines every model family with default hyperparameters
3. Tunes the top families by editing `train.py`'s experiment block
4. Finalizes the winner on the held-out test set

### 4. Score new data (optional)

```bash
python predict.py --bundle-name <bundle_name> --csv-path new_data.csv --save-path scored.csv
```

## Supported model families

Built into `train.py`:
- `logistic_regression` (GPU via cuML, CPU via scikit-learn)
- `random_forest` (GPU via cuML, CPU via scikit-learn)
- `extra_trees` (CPU only)
- `xgboost` (GPU via CUDA, CPU fallback)
- `lightgbm` (GPU via device_type, CPU fallback)
- `catboost` (GPU via task_type, CPU fallback)

## Hardware

This repo is **GPU-first with CPU fallback**.

| `HARDWARE_BACKEND` | Behavior |
|---------------------|----------|
| `gpu` | Strict GPU mode, fails if no GPU detected |
| `auto` | Uses GPU if available, falls back to CPU |
| `cpu` | Forces CPU, for local testing |

Set via environment variable or `preferred_backend` in `program.md`.

## Data sources supported by prepare.py

| Source | Flag | Required args |
|--------|------|---------------|
| Built-in demo | `--source demo --dataset-name <name>` | `breast_cancer` or `bank_marketing` |
| Local CSV | `--source csv --csv-path <path>` | `--target-column` |
| Public URL | `--source url --data-url <url>` | `--target-column` |
| Hugging Face | `--source huggingface --hf-dataset <name>` | `--target-column` |

## Search constraints

`prepare.py` writes these into `bundles/<name>/metadata.json`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_total_trials` | 20 | Max experiments across all families |
| `min_trials_per_family` | 3 | Min experiments before a family can be pruned |
| `max_trials_per_family` | 8 | Max experiments for any single family |
| `max_consecutive_non_improvements` | 5 | Stop a family after this many non-improvements |
| `min_improvement` | 0.001 | Ignore gains smaller than this |
| `per_run_timeout_minutes` | 10 | Timeout per experiment |

Override defaults via `program.md` or CLI args to `prepare.py`.

## Resetting between tasks

After a research run, `train.py` contains the winning model's configuration
(committed by the agent as part of its search loop). Before starting a new
task on a different dataset:

```bash
python reset.py              # reset train.py experiment block only
python reset.py --program    # also reset program.md to the blank template
```

This surgically resets only the experiment block between the `===` markers
in `train.py` — the rest of the file is untouched. It does not delete
`bundles/` or `outputs/` (these are namespaced by bundle name and gitignored).

## Outputs

Each experiment writes to `outputs/<bundle_name>/`:
- `results.tsv` — leaderboard of all runs
- `last_run.json` — most recent experiment metadata
- `runs/<run_id>/model.joblib` — trained sklearn pipeline
- `runs/<run_id>/summary.json` — full run config and metrics

## MLflow

Local file logging is always on. MLflow is optional — enable it with
`--enable-mlflow` on `prepare.py` if you have a tracking server.

## Tests

The repo includes test scripts under `tests/`:
- `tests/test_local_patch_loop.py` — simulates the agent loop with preset
  hypotheses (deterministic, no LLM call)
- `tests/test_precision_recall.py` — validates model meets minimum PR thresholds
