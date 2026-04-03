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

    # --- Employment sentinel fix (365243 = not employed, not a real day count) ---
    if 'DAYS_EMPLOYED' in df.columns:
        df['employed_anomaly'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
        df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)

    # --- Debt ratios ---
    if 'AMT_ANNUITY' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
        income = df['AMT_INCOME_TOTAL'].replace(0, np.nan)
        df['annuity_income_ratio'] = df['AMT_ANNUITY'] / income
    if 'AMT_CREDIT' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
        df['credit_income_ratio'] = df['AMT_CREDIT'] / income
    if 'AMT_CREDIT' in df.columns and 'AMT_GOODS_PRICE' in df.columns:
        goods = df['AMT_GOODS_PRICE'].replace(0, np.nan)
        df['credit_goods_ratio'] = df['AMT_CREDIT'] / goods

    # --- External score combinations ---
    ext_cols = [c for c in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'] if c in df.columns]
    if ext_cols:
        df['ext_source_mean'] = df[ext_cols].mean(axis=1)
        df['ext_source_min'] = df[ext_cols].min(axis=1)
        df['ext_source_max'] = df[ext_cols].max(axis=1)
        df['ext_source_nancount'] = df[ext_cols].isna().sum(axis=1)
        if len(ext_cols) >= 2:
            df['ext_source_std'] = df[ext_cols].std(axis=1)
        if 'EXT_SOURCE_2' in df.columns and 'EXT_SOURCE_3' in df.columns:
            df['ext_source_2x3'] = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']

    # --- Age and employment derived ---
    if 'DAYS_BIRTH' in df.columns:
        df['age_years'] = df['DAYS_BIRTH'] / -365
    if 'DAYS_EMPLOYED' in df.columns and 'DAYS_BIRTH' in df.columns:
        df['employment_to_age'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']

    # --- External score x debt interactions ---
    if 'ext_source_mean' in df.columns and 'annuity_income_ratio' in df.columns:
        df['ext_score_x_debt'] = df['ext_source_mean'] * df['annuity_income_ratio']
    if 'ext_source_mean' in df.columns and 'age_years' in df.columns:
        df['ext_score_x_age'] = df['ext_source_mean'] * df['age_years']

    # --- Annuity relative to credit (payment intensity) ---
    if 'AMT_ANNUITY' in df.columns and 'AMT_CREDIT' in df.columns:
        credit = df['AMT_CREDIT'].replace(0, np.nan)
        df['payment_rate'] = df['AMT_ANNUITY'] / credit

    # --- Income per family ---
    if 'AMT_INCOME_TOTAL' in df.columns and 'CNT_FAM_MEMBERS' in df.columns:
        fam = df['CNT_FAM_MEMBERS'].replace(0, 1)
        df['income_per_family'] = df['AMT_INCOME_TOTAL'] / fam

    # --- Document count ---
    doc_cols = [c for c in df.columns if c.startswith('FLAG_DOCUMENT_')]
    if doc_cols:
        df['doc_count'] = df[doc_cols].sum(axis=1)

    # --- Bureau recent pressure ---
    recent_bureau = [c for c in df.columns if c.startswith('AMT_REQ_CREDIT_BUREAU_')
                     and any(p in c for p in ['HOUR', 'DAY', 'WEEK', 'MON'])]
    if recent_bureau:
        df['bureau_recent'] = df[recent_bureau].sum(axis=1)

    # === Agent edits above this line ===

    return df
