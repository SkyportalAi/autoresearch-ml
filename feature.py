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

    pay_cols = ['PAY_6', 'PAY_5', 'PAY_4', 'PAY_3', 'PAY_2', 'PAY_0']
    existing_pay = [c for c in pay_cols if c in df.columns]

    # Temporal delay features
    if len(existing_pay) >= 2:
        pay_matrix = df[existing_pay].values
        x = np.arange(len(existing_pay), dtype=float)
        x_centered = x - x.mean()
        denom = (x_centered ** 2).sum()
        df['pay_delay_trend'] = (pay_matrix * x_centered).sum(axis=1) / denom
        df['avg_delay'] = pay_matrix.mean(axis=1)
        df['max_delay'] = pay_matrix.max(axis=1)
        df['months_delayed'] = (pay_matrix > 0).sum(axis=1)
        if 'PAY_0' in df.columns and 'PAY_6' in df.columns:
            df['delay_acceleration'] = df['PAY_0'] - df['PAY_6']

    # Utilization features
    bill_cols = ['BILL_AMT6', 'BILL_AMT5', 'BILL_AMT4', 'BILL_AMT3', 'BILL_AMT2', 'BILL_AMT1']
    existing_bill = [c for c in bill_cols if c in df.columns]
    if 'LIMIT_BAL' in df.columns and len(existing_bill) >= 1:
        limit = df['LIMIT_BAL'].replace(0, 1)
        if 'BILL_AMT1' in df.columns:
            df['util_current'] = df['BILL_AMT1'] / limit
        df['util_avg'] = df[existing_bill].mean(axis=1) / limit
        df['util_max'] = df[existing_bill].max(axis=1) / limit
        if 'BILL_AMT1' in df.columns and 'BILL_AMT6' in df.columns:
            df['util_growth'] = (df['BILL_AMT1'] - df['BILL_AMT6']) / limit

    # Repayment ratio features
    amt_cols = ['PAY_AMT6', 'PAY_AMT5', 'PAY_AMT4', 'PAY_AMT3', 'PAY_AMT2', 'PAY_AMT1']
    existing_amt = [c for c in amt_cols if c in df.columns]
    repay_ratios = []
    for bill_c, amt_c in zip(existing_bill, existing_amt):
        if bill_c in df.columns and amt_c in df.columns:
            bill_safe = df[bill_c].clip(lower=1)
            ratio = (df[amt_c] / bill_safe).clip(upper=5)
            col_name = 'repay_ratio_' + bill_c[-1]
            df[col_name] = ratio
            repay_ratios.append(col_name)
    if repay_ratios:
        df['repay_avg'] = df[repay_ratios].mean(axis=1)
        if len(repay_ratios) >= 2:
            rr_matrix = df[repay_ratios].values
            x = np.arange(len(repay_ratios), dtype=float)
            x_c = x - x.mean()
            d2 = (x_c ** 2).sum()
            if d2 > 0:
                df['repay_trend'] = (rr_matrix * x_c).sum(axis=1) / d2

    # Payment behavior features
    if len(existing_amt) >= 1:
        df['pay_amt_std'] = df[existing_amt].std(axis=1)
        df['pay_amt_avg'] = df[existing_amt].mean(axis=1)
        df['months_zero_payment'] = (df[existing_amt] == 0).sum(axis=1)

    # Balance dynamics
    if len(existing_bill) >= 2:
        df['bill_std'] = df[existing_bill].std(axis=1)
        bill_matrix = df[existing_bill].values
        x = np.arange(len(existing_bill), dtype=float)
        x_c = x - x.mean()
        d2 = (x_c ** 2).sum()
        if d2 > 0:
            df['bill_trend'] = (bill_matrix * x_c).sum(axis=1) / d2

    # Composite risk indicators
    if 'PAY_0' in df.columns and len(existing_pay) >= 4:
        old_pays = [c for c in ['PAY_6', 'PAY_5', 'PAY_4'] if c in df.columns]
        if old_pays:
            df['recent_shock'] = df['PAY_0'] - df[old_pays].mean(axis=1)
    if 'LIMIT_BAL' in df.columns and len(existing_bill) >= 1:
        df['months_over_limit'] = sum((df[c] > df['LIMIT_BAL']).astype(int) for c in existing_bill)

    # === Agent edits above this line ===

    return df
