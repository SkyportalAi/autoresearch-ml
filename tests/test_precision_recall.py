#!/usr/bin/env python3
"""Recompute precision/recall metrics from the latest run artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score

DEFAULT_BUNDLE_ROOT = Path('bundles')
DEFAULT_OUTPUT_ROOT = Path('outputs')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle-name', required=True)
    p.add_argument('--bundle-root', type=Path, default=DEFAULT_BUNDLE_ROOT)
    p.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument('--split', choices=['val', 'test'], default='test')
    p.add_argument('--min-average-precision', type=float, default=0.80)
    p.add_argument('--min-precision', type=float, default=0.70)
    p.add_argument('--min-recall', type=float, default=0.70)
    p.add_argument('--save-json', type=Path, default=None)
    p.add_argument('--save-pr-curve', type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bundle_dir = args.bundle_root / args.bundle_name
    output_dir = args.output_root / args.bundle_name
    metadata = json.loads((bundle_dir / 'metadata.json').read_text())
    last = json.loads((output_dir / 'last_run.json').read_text())
    model_path = output_dir / 'runs' / last['run_id'] / 'model.joblib'
    pipeline = joblib.load(model_path)

    df = pd.read_csv(bundle_dir / f'{args.split}.csv')
    target_col = metadata['target_column']
    X = df.drop(columns=[target_col])
    X = X.drop(columns=last.get('feature_config', {}).get('drop_columns', []), errors='ignore')
    y = df[target_col].astype(int)

    proba = pipeline.predict_proba(X)
    scores = np.asarray(proba[:, 1] if getattr(proba, 'ndim', 2) == 2 else proba)
    threshold = last.get('test_metrics', last.get('val_metrics', {})).get('threshold', 0.5)
    preds = (scores >= threshold).astype(int)

    ap = float(average_precision_score(y, scores))
    prec = float(precision_score(y, preds, zero_division=0))
    rec = float(recall_score(y, preds, zero_division=0))

    payload = {
        'bundle_name': args.bundle_name,
        'split': args.split,
        'average_precision': ap,
        'precision': prec,
        'recall': rec,
        'threshold': float(threshold),
        'pass': bool(ap >= args.min_average_precision and prec >= args.min_precision and rec >= args.min_recall),
    }

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(payload, indent=2))

    if args.save_pr_curve:
        precision_arr, recall_arr, thresholds = precision_recall_curve(y, scores)
        curve = pd.DataFrame({
            'precision': precision_arr[:-1],
            'recall': recall_arr[:-1],
            'threshold': thresholds,
        })
        args.save_pr_curve.parent.mkdir(parents=True, exist_ok=True)
        curve.to_csv(args.save_pr_curve, index=False)

    print(json.dumps(payload, indent=2))
    if not payload['pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
