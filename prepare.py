#!/usr/bin/env python3
"""Prepare a fixed tabular bundle for AutoResearch-style experiments.

This file is the stable data/evaluation harness.
The iterative agent should not modify it during the search loop.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

DEFAULT_BUNDLE_ROOT = Path("bundles")
BANK_MARKETING_URL = "https://archive.ics.uci.edu/static/public/222/bank%2Bmarketing.zip"


@dataclass
class PrepareConfig:
    source: str
    bundle_name: str
    bundle_root: Path
    dataset_name: Optional[str]
    csv_path: Optional[Path]
    data_url: Optional[str]
    hf_dataset: Optional[str]
    hf_config: Optional[str]
    hf_split: Optional[str]
    kaggle_competition: Optional[str]
    kaggle_dataset: Optional[str]
    kaggle_file: Optional[str]
    target_column: Optional[str]
    positive_label: Optional[str]
    test_size: float
    val_size: float
    random_state: int
    drop_columns: list[str]
    keep_duration: bool
    primary_metric: str
    preferred_backend: str
    per_run_timeout_minutes: int
    max_total_trials: int
    min_trials_per_family: int
    max_trials_per_family: int
    max_consecutive_non_improvements: int
    min_improvement: float
    mlflow_enabled: bool
    mlflow_tracking_uri: Optional[str]
    mlflow_experiment_name: str
    mlflow_model_name: Optional[str]


def parse_args() -> PrepareConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', choices=['demo', 'csv', 'url', 'huggingface', 'kaggle'], required=True)
    p.add_argument('--bundle-name', required=True)
    p.add_argument('--bundle-root', type=Path, default=DEFAULT_BUNDLE_ROOT)
    p.add_argument('--dataset-name', choices=['breast_cancer', 'bank_marketing', 'home_credit'], default='breast_cancer')
    p.add_argument('--csv-path', type=Path, default=None)
    p.add_argument('--data-url', default=None, help='Public URL to a CSV or ZIP containing a CSV')
    p.add_argument('--hf-dataset', default=None)
    p.add_argument('--hf-config', default=None)
    p.add_argument('--hf-split', default='train')
    p.add_argument('--kaggle-competition', default=None, help='Kaggle competition slug (e.g., home-credit-default-risk)')
    p.add_argument('--kaggle-dataset', default=None, help='Kaggle dataset slug (e.g., uciml/default-of-credit-card-clients)')
    p.add_argument('--kaggle-file', default=None, help='Specific file to download from competition/dataset')
    p.add_argument('--target-column', default=None)
    p.add_argument('--positive-label', default=None)
    p.add_argument('--test-size', type=float, default=0.20)
    p.add_argument('--val-size', type=float, default=0.20)
    p.add_argument('--random-state', type=int, default=42)
    p.add_argument('--drop-columns', nargs='*', default=[])
    p.add_argument('--keep-duration', action='store_true')
    p.add_argument('--primary-metric', choices=['average_precision', 'roc_auc', 'f1', 'accuracy', 'neg_log_loss'], default='average_precision')
    p.add_argument('--preferred-backend', choices=['auto', 'gpu', 'cpu'], default='auto')
    p.add_argument('--per-run-timeout-minutes', type=int, default=10)
    p.add_argument('--max-total-trials', type=int, default=20)
    p.add_argument('--min-trials-per-family', type=int, default=3)
    p.add_argument('--max-trials-per-family', type=int, default=8)
    p.add_argument('--max-consecutive-non-improvements', type=int, default=5)
    p.add_argument('--min-improvement', type=float, default=0.001)
    p.add_argument('--enable-mlflow', action='store_true')
    p.add_argument('--mlflow-tracking-uri', default=None)
    p.add_argument('--mlflow-experiment-name', default=None)
    p.add_argument('--mlflow-model-name', default=None,
                   help='Registered model name in MLflow Model Registry (default: autoresearch_<bundle_name>)')
    args = p.parse_args()

    target_column = args.target_column
    positive_label = args.positive_label
    if args.source == 'demo' and args.dataset_name == 'bank_marketing':
        target_column = 'y'
        positive_label = 'yes'
    elif args.source == 'demo' and args.dataset_name == 'breast_cancer':
        target_column = 'target'
        positive_label = '1'
    elif args.source == 'demo' and args.dataset_name == 'home_credit':
        target_column = 'TARGET'
        positive_label = '1'
        args.drop_columns = list(args.drop_columns) + ['SK_ID_CURR']
    else:
        if not target_column:
            p.error('--target-column is required for csv, url, huggingface, and kaggle sources')

    if args.source == 'csv' and not args.csv_path:
        p.error('--csv-path is required when --source csv')
    if args.source == 'url' and not args.data_url:
        p.error('--data-url is required when --source url')
    if args.source == 'huggingface' and not args.hf_dataset:
        p.error('--hf-dataset is required when --source huggingface')
    if args.source == 'kaggle':
        if not args.kaggle_competition and not args.kaggle_dataset:
            p.error('--kaggle-competition or --kaggle-dataset is required when --source kaggle')
        if args.kaggle_competition and not args.kaggle_file:
            p.error('--kaggle-file is required for Kaggle competitions (they contain multiple files)')

    return PrepareConfig(
        source=args.source,
        bundle_name=args.bundle_name,
        bundle_root=args.bundle_root,
        dataset_name=args.dataset_name,
        csv_path=args.csv_path,
        data_url=args.data_url,
        hf_dataset=args.hf_dataset,
        hf_config=args.hf_config,
        hf_split=args.hf_split,
        kaggle_competition=args.kaggle_competition,
        kaggle_dataset=args.kaggle_dataset,
        kaggle_file=args.kaggle_file,
        target_column=target_column,
        positive_label=positive_label,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        drop_columns=list(args.drop_columns),
        keep_duration=bool(args.keep_duration),
        primary_metric=args.primary_metric,
        preferred_backend=args.preferred_backend,
        per_run_timeout_minutes=args.per_run_timeout_minutes,
        max_total_trials=args.max_total_trials,
        min_trials_per_family=args.min_trials_per_family,
        max_trials_per_family=args.max_trials_per_family,
        max_consecutive_non_improvements=args.max_consecutive_non_improvements,
        min_improvement=args.min_improvement,
        mlflow_enabled=bool(args.enable_mlflow),
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment_name=args.mlflow_experiment_name or f'skyportal_{args.bundle_name}',
        mlflow_model_name=args.mlflow_model_name,
    )


def _read_first_csv_from_zip(zf: zipfile.ZipFile) -> pd.DataFrame | None:
    preferred = [
        'bank/bank-full.csv',
        'bank-full.csv',
        'bank-additional/bank-additional-full.csv',
        'bank-additional-full.csv',
        'bank/bank.csv',
        'bank.csv',
    ]
    for name in preferred:
        if name in zf.namelist():
            with zf.open(name) as f:
                return pd.read_csv(f, sep=';')

    csv_members = [n for n in zf.namelist() if n.lower().endswith('.csv')]
    if csv_members:
        with zf.open(csv_members[0]) as f:
            return pd.read_csv(f)
    return None


def load_bank_marketing() -> pd.DataFrame:
    with TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / 'bank_marketing.zip'
        urllib.request.urlretrieve(BANK_MARKETING_URL, archive_path)
        with zipfile.ZipFile(archive_path) as outer:
            df = _read_first_csv_from_zip(outer)
            if df is not None:
                return df
            nested_zips = [n for n in outer.namelist() if n.lower().endswith('.zip')]
            for nested in nested_zips:
                with outer.open(nested) as f:
                    nested_bytes = f.read()
                with zipfile.ZipFile(io.BytesIO(nested_bytes)) as inner:
                    df = _read_first_csv_from_zip(inner)
                    if df is not None:
                        return df
            raise FileNotFoundError(f'No usable CSV found in {archive_path} members={outer.namelist()}')


def load_demo_dataset(name: str) -> pd.DataFrame:
    if name == 'breast_cancer':
        ds = load_breast_cancer(as_frame=True)
        return ds.frame.copy()
    if name == 'bank_marketing':
        return load_bank_marketing()
    if name == 'home_credit':
        return load_kaggle_dataset('home-credit-default-risk', None, 'application_train.csv')
    raise ValueError(f'Unsupported demo dataset: {name}')


def load_public_url(url: str) -> pd.DataFrame:
    parsed = urllib.parse.urlparse(url)
    filename = Path(parsed.path).name.lower()
    with TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / (filename or 'download.bin')
        urllib.request.urlretrieve(url, local_path)
        if local_path.suffix.lower() == '.csv':
            return pd.read_csv(local_path)
        if local_path.suffix.lower() == '.zip':
            with zipfile.ZipFile(local_path) as zf:
                df = _read_first_csv_from_zip(zf)
                if df is not None:
                    return df
                raise FileNotFoundError(f'No CSV found inside ZIP from {url}')
        try:
            return pd.read_csv(local_path)
        except Exception as exc:  # pragma: no cover
            raise ValueError(f'Unsupported URL payload for {url}: {local_path.name}') from exc


def load_huggingface_dataset(name: str, config_name: Optional[str], split: str) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover
        raise ImportError("Hugging Face datasets support requires `pip install datasets`") from exc

    ds = load_dataset(name, config_name, split=split)
    return ds.to_pandas()


def load_kaggle_dataset(competition: Optional[str], dataset: Optional[str], file: Optional[str]) -> pd.DataFrame:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise ImportError(
            "Kaggle source requires `pip install kaggle` and credentials.\n"
            "Set KAGGLE_API_TOKEN env var or place kaggle.json at ~/.kaggle/kaggle.json.\n"
            "For competitions, accept the rules on Kaggle's website first."
        )

    api = KaggleApi()
    api.authenticate()

    with TemporaryDirectory() as tmpdir:
        if competition:
            api.competition_download_file(competition, file, path=tmpdir)
        else:
            api.dataset_download_file(dataset, file, path=tmpdir)

        downloaded = Path(tmpdir) / file
        # Kaggle may wrap downloads in zip even when the filename is .csv
        if not downloaded.exists():
            for zp in Path(tmpdir).glob('*.zip'):
                with zipfile.ZipFile(zp) as zf:
                    zf.extractall(tmpdir)
        if downloaded.exists() and zipfile.is_zipfile(downloaded):
            extract_dir = Path(tmpdir) / '_extracted'
            extract_dir.mkdir()
            with zipfile.ZipFile(downloaded) as zf:
                zf.extractall(extract_dir)
            extracted_csv = extract_dir / file
            if extracted_csv.exists():
                downloaded = extracted_csv
            else:
                csvs = list(extract_dir.glob('*.csv'))
                if csvs:
                    downloaded = csvs[0]
        if not downloaded.exists():
            raise FileNotFoundError(
                f'Expected file {file} not found after download. '
                f'Contents: {list(Path(tmpdir).iterdir())}'
            )
        return pd.read_csv(downloaded)


def load_source(cfg: PrepareConfig) -> pd.DataFrame:
    if cfg.source == 'demo':
        return load_demo_dataset(cfg.dataset_name)
    if cfg.source == 'csv':
        return pd.read_csv(cfg.csv_path)
    if cfg.source == 'url':
        return load_public_url(cfg.data_url)
    if cfg.source == 'huggingface':
        return load_huggingface_dataset(cfg.hf_dataset, cfg.hf_config, cfg.hf_split)
    if cfg.source == 'kaggle':
        return load_kaggle_dataset(cfg.kaggle_competition, cfg.kaggle_dataset, cfg.kaggle_file)
    raise ValueError(f'Unsupported source: {cfg.source}')


def coerce_binary_target(series: pd.Series, positive_label: Optional[str]) -> tuple[pd.Series, dict[str, object]]:
    non_null = series.dropna()
    values = set(non_null.unique().tolist())
    if values.issubset({0, 1}):
        return series.fillna(0).astype(int), {'negative': 0, 'positive': 1}
    if non_null.dtype.kind == 'b':
        return series.fillna(False).astype(int), {'negative': 0, 'positive': 1}
    if positive_label is None:
        raise ValueError(
            f'Target {series.name!r} is not already binary 0/1. Pass --positive-label. Observed values={sorted(map(str, values))}'
        )
    y = (series.astype(str) == str(positive_label)).astype(int)
    return y, {'negative': f'not_{positive_label}', 'positive': positive_label}


def summarize_features(df: pd.DataFrame, target_column: str) -> dict[str, object]:
    X = df.drop(columns=[target_column])
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return {
        'row_count': int(df.shape[0]),
        'feature_count': int(X.shape[1]),
        'numeric_columns': numeric,
        'categorical_columns': categorical,
        'dtypes': X.dtypes.astype(str).to_dict(),
    }


def main() -> None:
    cfg = parse_args()
    bundle_dir = cfg.bundle_root / cfg.bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    df = load_source(cfg)
    if cfg.source == 'demo' and cfg.dataset_name == 'bank_marketing' and not cfg.keep_duration and 'duration' in df.columns:
        cfg.drop_columns.append('duration')

    missing_drop = [c for c in cfg.drop_columns if c not in df.columns]
    if missing_drop:
        raise KeyError(f'Requested drop columns not present: {missing_drop}')
    if cfg.drop_columns:
        df = df.drop(columns=cfg.drop_columns)

    if cfg.target_column not in df.columns:
        raise KeyError(f'Target column {cfg.target_column!r} not found. Available columns: {df.columns.tolist()}')

    y, label_map = coerce_binary_target(df[cfg.target_column], cfg.positive_label)
    df = df.copy()
    df[cfg.target_column] = y

    train_val_df, test_df = train_test_split(
        df,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=df[cfg.target_column],
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=cfg.val_size,
        random_state=cfg.random_state,
        stratify=train_val_df[cfg.target_column],
    )

    train_df.to_csv(bundle_dir / 'train.csv', index=False)
    val_df.to_csv(bundle_dir / 'val.csv', index=False)
    test_df.to_csv(bundle_dir / 'test.csv', index=False)

    metadata = {
        'bundle_name': cfg.bundle_name,
        'source': cfg.source,
        'source_details': {
            'dataset_name': cfg.dataset_name,
            'csv_path': str(cfg.csv_path) if cfg.csv_path else None,
            'data_url': cfg.data_url,
            'hf_dataset': cfg.hf_dataset,
            'hf_config': cfg.hf_config,
            'hf_split': cfg.hf_split,
        },
        'problem_type': 'binary_classification',
        'target_column': cfg.target_column,
        'label_map': label_map,
        'primary_metric': cfg.primary_metric,
        'split': {
            'test_size': cfg.test_size,
            'val_size': cfg.val_size,
            'random_state': cfg.random_state,
            'train_rows': int(train_df.shape[0]),
            'val_rows': int(val_df.shape[0]),
            'test_rows': int(test_df.shape[0]),
        },
        'feature_summary': summarize_features(df, cfg.target_column),
        'drop_columns': cfg.drop_columns,
        'hardware': {
            'preferred_backend': cfg.preferred_backend,
            'gpu_first': cfg.preferred_backend in {'auto', 'gpu'},
        },
        'trial_policy': {
            'per_run_timeout_minutes': cfg.per_run_timeout_minutes,
            'max_total_trials': cfg.max_total_trials,
            'min_trials_per_family': cfg.min_trials_per_family,
            'max_trials_per_family': cfg.max_trials_per_family,
            'max_consecutive_non_improvements': cfg.max_consecutive_non_improvements,
            'min_improvement': cfg.min_improvement,
        },
        'mlflow': {
            'enabled': cfg.mlflow_enabled,
            'tracking_uri': cfg.mlflow_tracking_uri,
            'experiment_name': cfg.mlflow_experiment_name,
            'registered_model_name': cfg.mlflow_model_name,
        },
    }
    (bundle_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    (bundle_dir / 'prepare_config.json').write_text(json.dumps(asdict(cfg), indent=2, default=str))

    print(f'Prepared bundle at {bundle_dir}')
    print(json.dumps({
        'bundle_name': cfg.bundle_name,
        'primary_metric': cfg.primary_metric,
        'preferred_backend': cfg.preferred_backend,
        'train_rows': int(train_df.shape[0]),
        'val_rows': int(val_df.shape[0]),
        'test_rows': int(test_df.shape[0]),
    }, indent=2))


if __name__ == '__main__':
    main()
