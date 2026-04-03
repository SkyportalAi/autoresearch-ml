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

    # --- Data cleaning ---
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0)

    # =====================================================================
    # DERIVED CONTINUOUS FEATURES
    # =====================================================================

    # Price shock: current bill vs historical average (did their bill go up?)
    if all(c in df.columns for c in ['MonthlyCharges', 'TotalCharges', 'tenure']):
        avg_hist = df['TotalCharges'] / df['tenure'].replace(0, 1)
        df['price_shock'] = df['MonthlyCharges'] - avg_hist

    # Tenure risk curve: high at start, decays — captures the "trial period" risk
    if 'tenure' in df.columns:
        df['tenure_risk'] = 1.0 / (df['tenure'] + 1)

    # Add-on count → value perception
    addon_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                  'TechSupport', 'StreamingTV', 'StreamingMovies']
    existing_addons = [c for c in addon_cols if c in df.columns]
    if existing_addons:
        df['num_addons'] = sum((df[c] == 'Yes').astype(int) for c in existing_addons)
        # Cost per service: high = overpaying for what you get
        if 'MonthlyCharges' in df.columns:
            df['cost_per_service'] = df['MonthlyCharges'] / (df['num_addons'] + 1)

    # Protection lock-in depth (dependency-creating services, not entertainment)
    protection_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
    prot_exist = [c for c in protection_cols if c in df.columns]
    if prot_exist and 'num_addons' in df.columns:
        df['num_protection'] = sum((df[c] == 'Yes').astype(int) for c in prot_exist)
        df['protection_ratio'] = df['num_protection'] / (df['num_addons'] + 1)

    # Commitment score: contract value x loyalty evidence
    if 'Contract' in df.columns and 'tenure' in df.columns:
        contract_months = df['Contract'].map({
            'Month-to-month': 1, 'One year': 12, 'Two year': 24
        }).fillna(1)
        df['commitment_score'] = contract_months * np.log1p(df['tenure'])

    # Friction-weighted cost: manual payers feel each dollar more
    if all(c in df.columns for c in ['MonthlyCharges', 'PaymentMethod']):
        is_manual = (~df['PaymentMethod'].isin(
            ['Bank transfer (automatic)', 'Credit card (automatic)']
        )).astype(float)
        df['felt_cost'] = df['MonthlyCharges'] * (1 + 0.5 * is_manual)

    # Household anchor: continuous stickiness
    anchor = 0
    if 'Partner' in df.columns:
        anchor = anchor + (df['Partner'] == 'Yes').astype(int)
    if 'Dependents' in df.columns:
        anchor = anchor + (df['Dependents'] == 'Yes').astype(int)
    df['household_anchor'] = anchor

    # === Agent edits above this line ===

    return df
