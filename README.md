# SkyPortal AutoResearch for GPU-first Tabular ML

Inspired by Andrej Karpathy's [AutoResearch](https://github.com/karpathy/autoresearch),
which uses an LLM agent to iteratively improve LLM training runs, this repo
applies the same pattern to **classical machine learning on tabular data**.

Instead of tuning LLM hyperparameters, this repo lets a data scientist
define a business objective (e.g., predict customer churn, score fraud risk),
point it at a dataset, provide business context about the features, and
choose which model families to evaluate. An LLM agent then takes over —
first engineering derived features using business domain knowledge, then
systematically searching the model and hyperparameter space, running
experiments, comparing results, and narrowing down to the single best
model with its best configuration. The data scientist defines *what* to
solve, provides domain context, and selects *which models* to consider;
the agent figures out the *how* — from feature creation to final tuning.

Supported model families: logistic regression, random forest, extra trees,
XGBoost, LightGBM, and CatBoost — with GPU acceleration where available
and automatic CPU fallback.

## How it works

```
program.md          The data scientist fills in the task: dataset, metric, models,
    +               and business context (churn drivers, domain knowledge, hints).
feature.md          Column-level documentation: what each feature means, quirks.
    |
    v
prompt.md           The agent reads its operating instructions and guardrails.
    |
    v
prepare.py          The agent runs this once to create an immutable data bundle.
    |
    v
feature.py          Phase 2: The agent engineers derived features — ratios,
    |               interactions, domain flags — guided by business context
    |               in program.md and column docs in feature.md. Iterates
    |               with a fixed LightGBM testbed until features stabilize.
    |
    v
train.py            Phase 3: The agent searches across model families, tuning
    |               hyperparameters. Edits the experiment block, runs, reads
    |               results, and repeats for up to max_total_trials iterations.
    v
predict.py          Score new data with the finalized winning model.
```

## Key files

| File | Role | Who edits it |
|------|------|--------------|
| `program.md` | Task definition: dataset, metric, models, business context, domain knowledge | You (the data scientist) |
| `feature.md` | Column-level documentation: what each feature means, data types, quirks | You (the data scientist) |
| `agentic/prompt.md` | Agent instructions: feature engineering protocol, search policy, model tuning guidance | Repo maintainer |
| `prepare.py` | Data preparation harness: loads data, splits, writes bundle | Nobody during search |
| `feature.py` | Feature engineering function with agent-editable section | The agent (Phase 2) |
| `train.py` | Single-experiment trainer with an editable config block at the top | The agent (Phase 3) |
| `predict.py` | Score new data with a finalized model (applies feature engineering) | Run manually after research |
| `reset.py` | Reset train.py, feature.py, and program.md to clean starting state | You (between research tasks) |

## Quick start

### 1. Reset to a clean state (if starting fresh)

```bash
python3 reset.py --program
```

This resets `train.py` to its default experiment block and `program.md` to
the blank template. Run this before starting a new research task.

### 2. Fill in program.md and feature.md

Open `program.md` and fill in your task. Beyond the dataset and model
configuration, include **business context** — what drives the target
variable, known high-risk profiles, and hints for feature engineering.
The richer your domain knowledge, the better the agent's feature
engineering will be.

Open `feature.md` and document each column: what it represents, its data
type, encoding quirks (e.g., "stored as string, needs conversion"), and
any domain-specific meaning of special values.

Pick a data source:

**Built-in demo datasets:**

`breast_cancer` (569 rows, 30 features) and `bank_marketing` (45K rows,
15 features) work out of the box — no external setup needed:
```markdown
- **source**: `demo`
- **dataset_name**: `bank_marketing`
- **bundle_name**: `bank_marketing`
```

`home_credit` (307K rows, 120 features) is the feature engineering
showcase dataset — requires Kaggle credentials (see prerequisites below):
```markdown
- **source**: `demo`
- **dataset_name**: `home_credit`
- **bundle_name**: `home_credit`
```

> **Kaggle prerequisites (for `home_credit` demo):** `pip3 install kaggle`
> and set `KAGGLE_API_TOKEN` env var or place `kaggle.json` at `~/.kaggle/`.
> Accept the competition rules at
> https://www.kaggle.com/competitions/home-credit-default-risk/rules

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

**Kaggle competition:**
```markdown
- **source**: `kaggle`
- **kaggle_competition**: `home-credit-default-risk`
- **kaggle_file**: `application_train.csv`
- **bundle_name**: `home_credit`
- **target_column**: `TARGET`
- **positive_label**: `1`
```

**Kaggle dataset:**
```markdown
- **source**: `kaggle`
- **kaggle_dataset**: `uciml/default-of-credit-card-clients`
- **kaggle_file**: `UCI_Credit_Card.csv`
- **bundle_name**: `credit_card`
- **target_column**: `default.payment.next.month`
```

> **Kaggle prerequisites:** `pip3 install kaggle` and set up credentials
> (place `kaggle.json` at `~/.kaggle/` or set `KAGGLE_API_TOKEN` env var).
> For competitions, accept the rules on Kaggle's website first.

Choose which model families to evaluate, set your primary metric, and
optionally adjust search constraints. See the template comments for details.

### 3. Point your agent at the repo

The agent reads `program.md`, `feature.md`, and `agentic/prompt.md`, then:

1. Runs `prepare.py` to create the data bundle (if it doesn't exist)
2. **Phase 2 — Feature engineering**: Reads business context from
   `program.md` and column docs from `feature.md`, then iteratively
   builds derived features in `feature.py` (ratios, interactions,
   domain flags, transforms). Uses LightGBM as a fixed testbed to
   isolate the effect of features. Iterates until features stabilize.
3. **Phase 3 — Model search**: Baselines every model family, then tunes
   the top families by editing `train.py`'s experiment block. All
   families are tested with the locked feature set from Phase 2.
4. Finalizes the winner on the held-out test set

### 4. Score new data (optional)

```bash
python3 predict.py --bundle-name <bundle_name> --csv-path new_data.csv --save-path scored.csv
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
| Built-in demo | `--source demo --dataset-name <name>` | `breast_cancer`, `bank_marketing`, or `home_credit` |
| Local CSV | `--source csv --csv-path <path>` | `--target-column` |
| Public URL | `--source url --data-url <url>` | `--target-column` |
| Hugging Face | `--source huggingface --hf-dataset <name>` | `--target-column` |
| Kaggle | `--source kaggle --kaggle-competition <slug>` | `--kaggle-file`, `--target-column` |
| Kaggle dataset | `--source kaggle --kaggle-dataset <owner/name>` | `--kaggle-file`, `--target-column` |

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

After a research run, `train.py` and `feature.py` contain the winning
configuration and engineered features. Before starting a new task on a
different dataset:

```bash
python3 reset.py              # reset train.py experiment block and feature.py
python3 reset.py --program    # also reset program.md to the blank template
```

This surgically resets only the editable sections between the `===` markers
in `train.py` and `feature.py` — the rest of the files are untouched. It
does not delete `bundles/` or `outputs/` (these are namespaced by bundle
name and gitignored).

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
