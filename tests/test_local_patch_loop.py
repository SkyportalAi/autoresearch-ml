#!/usr/bin/env python3
"""Deterministic local patch loop that simulates an outer AutoResearch agent.

It patches the editable experiment block in train.py, runs a sequence of
hypotheses, writes a leaderboard, and can finalize the current best run.
"""
from __future__ import annotations

import argparse
import json
import os
import pprint
import re
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = REPO_ROOT / 'train.py'
DEFAULT_OUTPUT_ROOT = REPO_ROOT / 'outputs'

BLOCK_PATTERN = re.compile(
    r"# ============================================================================\n# Editable experiment block\.\n# The external agent may change only this block during the search loop\.\n# ============================================================================\n.*?# ============================================================================\n",
    re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle-name', required=True)
    p.add_argument('--preset', choices=['core3', 'boosters5', 'mixed7'], default='mixed7')
    p.add_argument('--hardware-backend', choices=['cpu', 'auto', 'gpu'], default='cpu')
    p.add_argument('--finalize-best', action='store_true')
    return p.parse_args()


def block_for(h: dict[str, Any]) -> str:
    return f"""# ============================================================================\n# Editable experiment block.\n# The external agent may change only this block during the search loop.\n# ============================================================================\nEXPERIMENT_NAME = {h['name']!r}\nEXPERIMENT_DESCRIPTION = {h['description']!r}\nMODEL_FAMILY = {h['family']!r}\nMODEL_PARAMS = {pprint.pformat(h['params'], sort_dicts=True)}\nFEATURE_CONFIG = {pprint.pformat(h.get('feature_config', {'scale_numeric': False, 'drop_columns': []}), sort_dicts=True)}\nTHRESHOLD_CONFIG = {pprint.pformat(h.get('threshold_config', {'strategy': 'best_f1', 'value': 0.5}), sort_dicts=True)}\n# ============================================================================\n"""


def build_hypotheses(preset: str) -> list[dict[str, Any]]:
    base_fc = {'scale_numeric': False, 'drop_columns': []}
    log_fc = {'scale_numeric': True, 'drop_columns': []}
    if preset == 'core3':
        return [
            {'name': 'logreg_low_c', 'description': 'Low regularization strength baseline', 'family': 'logistic_regression', 'params': {'C': 0.01, 'max_iter': 2000}, 'feature_config': log_fc},
            {'name': 'logreg_high_c', 'description': 'Higher C logistic regression', 'family': 'logistic_regression', 'params': {'C': 10.0, 'max_iter': 2000}, 'feature_config': log_fc},
            {'name': 'rf_baseline', 'description': 'Random forest baseline', 'family': 'random_forest', 'params': {'n_estimators': 120, 'max_depth': 6, 'min_samples_leaf': 1}, 'feature_config': base_fc},
        ]
    if preset == 'boosters5':
        return [
            {'name': 'xgb_small', 'description': 'Small XGBoost baseline', 'family': 'xgboost', 'params': {'n_estimators': 10, 'max_depth': 2, 'learning_rate': 0.2, 'subsample': 1.0, 'colsample_bytree': 1.0}, 'feature_config': base_fc},
            {'name': 'xgb_larger', 'description': 'Larger XGBoost config', 'family': 'xgboost', 'params': {'n_estimators': 40, 'max_depth': 3, 'learning_rate': 0.08, 'subsample': 0.9, 'colsample_bytree': 0.9}, 'feature_config': base_fc},
            {'name': 'lgb_small', 'description': 'LightGBM baseline', 'family': 'lightgbm', 'params': {'n_estimators': 10, 'learning_rate': 0.08, 'num_leaves': 15, 'max_depth': 4}, 'feature_config': base_fc},
            {'name': 'cat_small', 'description': 'CatBoost baseline', 'family': 'catboost', 'params': {'iterations': 20, 'depth': 4, 'learning_rate': 0.08}, 'feature_config': base_fc},
            {'name': 'cat_deeper', 'description': 'CatBoost deeper model', 'family': 'catboost', 'params': {'iterations': 30, 'depth': 6, 'learning_rate': 0.05}, 'feature_config': base_fc},
        ]
    return [
        {'name': 'logreg_low_c', 'description': 'Low regularization logistic regression', 'family': 'logistic_regression', 'params': {'C': 0.01, 'max_iter': 2000}, 'feature_config': log_fc},
        {'name': 'logreg_high_c', 'description': 'High C logistic regression', 'family': 'logistic_regression', 'params': {'C': 10.0, 'max_iter': 2000}, 'feature_config': log_fc},
        {'name': 'rf_baseline', 'description': 'Random forest baseline', 'family': 'random_forest', 'params': {'n_estimators': 120, 'max_depth': 6, 'min_samples_leaf': 1}, 'feature_config': base_fc},
        {'name': 'xgb_small', 'description': 'Small XGBoost baseline', 'family': 'xgboost', 'params': {'n_estimators': 10, 'max_depth': 2, 'learning_rate': 0.2, 'subsample': 1.0, 'colsample_bytree': 1.0}, 'feature_config': base_fc},
        {'name': 'xgb_larger', 'description': 'Larger XGBoost config', 'family': 'xgboost', 'params': {'n_estimators': 40, 'max_depth': 3, 'learning_rate': 0.08, 'subsample': 0.9, 'colsample_bytree': 0.9}, 'feature_config': base_fc},
        {'name': 'lgb_small', 'description': 'LightGBM baseline', 'family': 'lightgbm', 'params': {'n_estimators': 10, 'learning_rate': 0.08, 'num_leaves': 15, 'max_depth': 4}, 'feature_config': base_fc},
        {'name': 'cat_small', 'description': 'CatBoost baseline', 'family': 'catboost', 'params': {'iterations': 20, 'depth': 4, 'learning_rate': 0.08}, 'feature_config': base_fc},
    ]


def patch_train_file(train_text: str, hypothesis: dict[str, Any]) -> str:
    replacement = block_for(hypothesis)
    patched, count = BLOCK_PATTERN.subn(replacement, train_text)
    if count != 1:
        raise RuntimeError('Could not patch the editable experiment block in train.py')
    return patched


def run_train(bundle_name: str, hardware_backend: str, finalize: bool = False) -> None:
    env = os.environ.copy()
    env['HARDWARE_BACKEND'] = hardware_backend
    cmd = ['python3', str(TRAIN_PATH), '--bundle-name', bundle_name]
    if finalize:
        cmd.append('--finalize')
    cp = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        check=False,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if cp.returncode != 0:
        raise RuntimeError(f'Run failed for {cmd}:\nSTDOUT:\n{cp.stdout}\nSTDERR:\n{cp.stderr}')


def main() -> None:
    args = parse_args()
    hypotheses = build_hypotheses(args.preset)
    original = TRAIN_PATH.read_text()
    leaderboard_rows = []
    output_dir = DEFAULT_OUTPUT_ROOT / args.bundle_name
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        for hyp in hypotheses:
            TRAIN_PATH.write_text(patch_train_file(original, hyp))
            run_train(args.bundle_name, args.hardware_backend, finalize=False)
            last = json.loads((output_dir / 'last_run.json').read_text())
            leaderboard_rows.append({
                'experiment_name': last['experiment_name'],
                'model_family': last['model_family'],
                'requested_backend': last['requested_backend'],
                'actual_backend': last['actual_backend'],
                'primary_metric': last['primary_metric_name'],
                'primary_metric_value': last['primary_metric_value'],
                'val_average_precision': last['val_metrics']['average_precision'],
                'val_roc_auc': last['val_metrics']['roc_auc'],
                'val_f1': last['val_metrics']['f1'],
                'params_json': json.dumps(last['model_params'], sort_keys=True),
            })

        leaderboard = pd.DataFrame(leaderboard_rows).sort_values('primary_metric_value', ascending=False).reset_index(drop=True)
        leaderboard_path = output_dir / 'tiny_harness_leaderboard.csv'
        leaderboard.to_csv(leaderboard_path, index=False)

        best_name = str(leaderboard.iloc[0]['experiment_name'])
        if args.finalize_best:
            best_hyp = next(h for h in hypotheses if h['name'] == best_name)
            TRAIN_PATH.write_text(patch_train_file(original, best_hyp))
            run_train(args.bundle_name, args.hardware_backend, finalize=False)
            run_train(args.bundle_name, args.hardware_backend, finalize=True)
            final_payload = json.loads((output_dir / 'last_run.json').read_text())
        else:
            final_payload = {'best_experiment_name': best_name}

        result = {
            'bundle_name': args.bundle_name,
            'preset': args.preset,
            'hardware_backend': args.hardware_backend,
            'best_experiment_name': best_name,
            'leaderboard_path': str(leaderboard_path),
            'final_payload': final_payload,
        }
        (output_dir / 'tiny_harness_result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    finally:
        TRAIN_PATH.write_text(original)


if __name__ == '__main__':
    main()
