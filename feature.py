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

    # Payment delay columns in temporal order (oldest → most recent)
    pay_cols = ['PAY_6', 'PAY_5', 'PAY_4', 'PAY_3', 'PAY_2', 'PAY_0']
    bill_cols = ['BILL_AMT6', 'BILL_AMT5', 'BILL_AMT4', 'BILL_AMT3', 'BILL_AMT2', 'BILL_AMT1']
    amt_cols = ['PAY_AMT6', 'PAY_AMT5', 'PAY_AMT4', 'PAY_AMT3', 'PAY_AMT2', 'PAY_AMT1']

    existing_pay = [c for c in pay_cols if c in df.columns]
    existing_bill = [c for c in bill_cols if c in df.columns]
    existing_amt = [c for c in amt_cols if c in df.columns]

    # =====================================================================
    # TEMPORAL TREND FEATURES — the slope of behavior over 6 months
    # =====================================================================

    if len(existing_pay) >= 2:
        pay_matrix = df[existing_pay].values
        # Payment delay trend: positive slope = getting worse
        x = np.arange(len(existing_pay), dtype=float)
        x_centered = x - x.mean()
        denom = (x_centered ** 2).sum()
        df['pay_delay_trend'] = (pay_matrix * x_centered).sum(axis=1) / denom
        # Average delay across all months
        df['avg_delay'] = pay_matrix.mean(axis=1)
        # Max delay in the window
        df['max_delay'] = pay_matrix.max(axis=1)
        # Months with any delay (PAY > 0)
        df['months_delayed'] = (pay_matrix > 0).sum(axis=1)
        # Recent vs old delay (is it getting worse recently?)
        if 'PAY_0' in df.columns and 'PAY_6' in df.columns:
            df['delay_acceleration'] = df['PAY_0'] - df['PAY_6']

    # =====================================================================
    # UTILIZATION FEATURES — how much of the credit limit is used
    # =====================================================================

    if 'LIMIT_BAL' in df.columns and len(existing_bill) >= 1:
        limit = df['LIMIT_BAL'].replace(0, 1)
        # Current utilization
        if 'BILL_AMT1' in df.columns:
            df['util_current'] = df['BILL_AMT1'] / limit
        # Oldest utilization
        if 'BILL_AMT6' in df.columns:
            df['util_oldest'] = df['BILL_AMT6'] / limit
        # Average utilization
        df['util_avg'] = df[existing_bill].mean(axis=1) / limit
        # Max utilization (worst month)
        df['util_max'] = df[existing_bill].max(axis=1) / limit
        # Is utilization growing? (balance growth relative to limit)
        if 'BILL_AMT1' in df.columns and 'BILL_AMT6' in df.columns:
            df['util_growth'] = (df['BILL_AMT1'] - df['BILL_AMT6']) / limit

    # =====================================================================
    # REPAYMENT RATIO FEATURES — how much of the bill is actually paid
    # =====================================================================

    repay_ratios = []
    for bill_c, amt_c in zip(existing_bill, existing_amt):
        if bill_c in df.columns and amt_c in df.columns:
            bill_safe = df[bill_c].clip(lower=1)  # avoid div by 0
            ratio = df[amt_c] / bill_safe
            ratio = ratio.clip(upper=5)  # cap extreme ratios
            col_name = 'repay_ratio_' + bill_c[-1]
            df[col_name] = ratio
            repay_ratios.append(col_name)

    if repay_ratios:
        # Average repayment ratio
        df['repay_avg'] = df[repay_ratios].mean(axis=1)
        # Repayment trend (slope over time)
        if len(repay_ratios) >= 2:
            rr_matrix = df[repay_ratios].values
            x = np.arange(len(repay_ratios), dtype=float)
            x_centered = x - x.mean()
            denom = (x_centered ** 2).sum()
            if denom > 0:
                df['repay_trend'] = (rr_matrix * x_centered).sum(axis=1) / denom

    # =====================================================================
    # PAYMENT BEHAVIOR FEATURES
    # =====================================================================

    if len(existing_amt) >= 1:
        # Payment consistency (std dev of payments — erratic = risky)
        df['pay_amt_std'] = df[existing_amt].std(axis=1)
        # Average payment amount
        df['pay_amt_avg'] = df[existing_amt].mean(axis=1)
        # Months with zero payment
        df['months_zero_payment'] = (df[existing_amt] == 0).sum(axis=1)

    # =====================================================================
    # BALANCE DYNAMICS
    # =====================================================================

    if len(existing_bill) >= 2:
        # Balance volatility
        df['bill_std'] = df[existing_bill].std(axis=1)
        # Balance trend (slope)
        bill_matrix = df[existing_bill].values
        x = np.arange(len(existing_bill), dtype=float)
        x_centered = x - x.mean()
        denom = (x_centered ** 2).sum()
        if denom > 0:
            df['bill_trend'] = (bill_matrix * x_centered).sum(axis=1) / denom

    # =====================================================================
    # COMPOSITE RISK INDICATORS
    # =====================================================================

    # Recent shock: good history but recent deterioration
    if 'PAY_0' in df.columns and len(existing_pay) >= 4:
        old_pays = [c for c in ['PAY_6', 'PAY_5', 'PAY_4'] if c in df.columns]
        if old_pays:
            old_avg = df[old_pays].mean(axis=1)
            df['recent_shock'] = df['PAY_0'] - old_avg

    # Over-limit indicator: any month where bill exceeded limit
    if 'LIMIT_BAL' in df.columns and len(existing_bill) >= 1:
        over_limit = sum((df[c] > df['LIMIT_BAL']).astype(int) for c in existing_bill)
        df['months_over_limit'] = over_limit

    # === Agent edits above this line ===

    return df
