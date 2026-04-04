#!/usr/bin/env python3
"""Feature engineering module for AutoResearch.

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
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw features into model-ready features.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe including all original columns and the target.

    Returns
    -------
    pd.DataFrame
        Dataframe with engineered features added (and optionally
        original columns removed).
    """
    df = df.copy()

    # === Agent edits below this line ===

    # Default: passthrough (no engineered features)

    # === Agent edits above this line ===

    return df
