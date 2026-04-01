#!/usr/bin/env python3
"""Score a new CSV with the latest finalized model artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

DEFAULT_OUTPUT_ROOT = Path('outputs')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle-name', required=True)
    p.add_argument('--csv-path', type=Path, required=True)
    p.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument('--save-path', type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_root / args.bundle_name
    last_run = json.loads((output_dir / 'last_run.json').read_text())
    model_path = output_dir / 'runs' / last_run['run_id'] / 'model.joblib'
    pipeline = joblib.load(model_path)
    df = pd.read_csv(args.csv_path)
    drop_cols = last_run.get('feature_config', {}).get('drop_columns', [])
    X = df.drop(columns=drop_cols, errors='ignore')
    proba = pipeline.predict_proba(X)
    scores = proba[:, 1] if getattr(proba, 'ndim', 2) == 2 else proba
    result = df.copy()
    result['score'] = scores
    threshold = last_run.get('test_metrics', last_run.get('val_metrics', {})).get('threshold', 0.5)
    result['prediction'] = (result['score'] >= threshold).astype(int)
    if args.save_path:
        result.to_csv(args.save_path, index=False)
        print(f'Saved predictions to {args.save_path}')
    else:
        print(result.head().to_string(index=False))


if __name__ == '__main__':
    main()
