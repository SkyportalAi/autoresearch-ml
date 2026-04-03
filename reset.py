#!/usr/bin/env python3
"""Reset train.py experiment block and feature.py to their default states.

Run this once before starting a new research task on a different dataset.
Do NOT run this during an active search loop — it would erase the agent's
current best configuration and engineered features.

Usage:
    python3 reset.py              # reset train.py + feature.py
    python3 reset.py --program    # also reset program.md to the blank template
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

TRAIN_PATH = Path(__file__).parent / 'train.py'
PROGRAM_PATH = Path(__file__).parent / 'program.md'
FEATURE_PATH = Path(__file__).parent / 'feature.py'

# The default experiment block that ships with the repo.
# This must match the content between the === markers in a clean train.py.
DEFAULT_EXPERIMENT_BLOCK = """\
EXPERIMENT_NAME = 'baseline_logreg'
EXPERIMENT_DESCRIPTION = 'Baseline logistic regression with defaults'
MODEL_FAMILY = 'logistic_regression'
MODEL_PARAMS = {'C': 1.0, 'max_iter': 2000}
FEATURE_CONFIG = {'drop_columns': [], 'scale_numeric': True}
THRESHOLD_CONFIG = {'strategy': 'best_f1', 'value': 0.5}"""

# Regex: match everything between the two === marker lines (inclusive).
# Uses \r?\n to handle both Unix (LF) and Windows (CRLF) line endings.
BLOCK_PATTERN = re.compile(
    r'(# ={70,}\r?\n'           # opening marker
    r'# Editable experiment block\.\r?\n'
    r'# The external agent may change only this block during the search loop\.\r?\n'
    r'# ={70,}\r?\n)'           # end of opening marker
    r'(.*?)'                     # the experiment block content (to replace)
    r'(\r?\n# ={70,}\r?\n)',    # closing marker
    re.DOTALL,
)

PROGRAM_TEMPLATE = """\
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
"""


def reset_train(path: Path) -> bool:
    """Reset the experiment block in train.py. Returns True if changed."""
    content = path.read_text()
    match = BLOCK_PATTERN.search(content)
    if not match:
        print(f'ERROR: Could not find experiment block markers in {path}')
        print('Expected the block between # === ... === comment lines.')
        return False

    current_block = match.group(2).strip()
    if current_block == DEFAULT_EXPERIMENT_BLOCK.strip():
        print(f'{path.name}: already at default state, no changes needed.')
        return False

    new_content = BLOCK_PATTERN.sub(
        rf'\g<1>{DEFAULT_EXPERIMENT_BLOCK}\n\g<3>',
        content,
    )
    path.write_text(new_content)
    print(f'{path.name}: experiment block reset to default.')
    return True


DEFAULT_FEATURE_PY = """\
#!/usr/bin/env python3
\"\"\"Feature engineering module for AutoResearch.

The LLM agent edits the `engineer_features` function below to add,
modify, or remove derived features. This function is called on every
data split (train, val, test) before the target column is separated.

Rules:
  - The function receives the FULL dataframe INCLUDING the target column.
  - Do NOT modify or drop the target column.
  - The function must be idempotent — safe to call on any split.
  - Handle missing columns gracefully (use `if col in df.columns` guards).
  - New features should be numeric or categorical (object dtype).
    The existing ColumnTransformer in train.py handles both paths.
  - To drop original columns, either drop them here or add them to
    FEATURE_CONFIG['drop_columns'] in train.py.
\"\"\"
from __future__ import annotations

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Transform raw features into model-ready features.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe including all original columns and the target.

    Returns
    -------
    pd.DataFrame
        Dataframe with engineered features added (and optionally
        original columns removed).
    \"\"\"
    df = df.copy()

    # === Agent edits below this line ===

    # Default: passthrough (no engineered features)

    # === Agent edits above this line ===

    return df
"""


def reset_feature(path: Path) -> bool:
    """Reset feature.py to the default passthrough. Returns True if changed."""
    if path.exists():
        current = path.read_text()
        if current.strip() == DEFAULT_FEATURE_PY.strip():
            print(f'{path.name}: already at default state, no changes needed.')
            return False

    path.write_text(DEFAULT_FEATURE_PY)
    print(f'{path.name}: reset to default passthrough.')
    return True


def reset_program(path: Path) -> bool:
    """Reset program.md to the blank template. Returns True if changed."""
    if path.exists():
        current = path.read_text()
        if current.strip() == PROGRAM_TEMPLATE.strip():
            print(f'{path.name}: already at template state, no changes needed.')
            return False

    path.write_text(PROGRAM_TEMPLATE)
    print(f'{path.name}: reset to blank template.')
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--program', action='store_true',
        help='Also reset program.md to the blank template',
    )
    args = parser.parse_args()

    changed = reset_train(TRAIN_PATH)
    changed = reset_feature(FEATURE_PATH) or changed
    if args.program:
        changed = reset_program(PROGRAM_PATH) or changed

    if changed:
        print('\nReset complete. Next steps:')
        print('  1. Fill in program.md with your new task configuration')
        print('  2. Fill in feature.md with column descriptions for your dataset')
        print('  3. Point your agent at the repo to start a new research run')
    else:
        print('\nEverything is already at default state.')


if __name__ == '__main__':
    main()
