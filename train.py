#!/usr/bin/env python3
"""Single-experiment trainer for GPU-first tabular AutoResearch.

The external agent should edit only the experiment block near the top of this
file. Every invocation runs exactly one hypothesis and writes local artifacts.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

# ============================================================================
# Editable experiment block.
# The external agent may change only this block during the search loop.
# ============================================================================
EXPERIMENT_NAME = 'fe_catboost'
EXPERIMENT_DESCRIPTION = 'R1 derived features + catboost'
MODEL_FAMILY = 'catboost'
MODEL_PARAMS = {'depth': 6, 'iterations': 200, 'learning_rate': 0.05, 'verbose': 0}
FEATURE_CONFIG = {'drop_columns': [], 'scale_numeric': True}
THRESHOLD_CONFIG = {'strategy': 'best_f1', 'value': 0.5}

# ============================================================================

DEFAULT_BUNDLE_ROOT = Path('bundles')
DEFAULT_OUTPUT_ROOT = Path('outputs')


@dataclass
class ModelSpec:
    name: str
    needs_scaling_by_default: bool
    cpu_builder: Callable[[dict[str, Any], int], Any]
    gpu_builder: Optional[Callable[[dict[str, Any], int], Any]] = None
    optional_dependency: Optional[str] = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle-name', required=True)
    p.add_argument('--bundle-root', type=Path, default=DEFAULT_BUNDLE_ROOT)
    p.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument('--finalize', action='store_true', help='Refit on train+val and evaluate the test split')
    return p.parse_args()


def get_git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], text=True, stderr=subprocess.DEVNULL)
        return out.strip()
    except Exception:
        return None


def _hash_feature_py() -> Optional[str]:
    """Return a short hash of feature.py for experiment tracking."""
    import hashlib
    feature_path = Path(__file__).parent / 'feature.py'
    if feature_path.exists():
        content = feature_path.read_text()
        return hashlib.sha256(content.encode()).hexdigest()[:12]
    return None


def load_bundle(bundle_root: Path, bundle_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    bundle_dir = bundle_root / bundle_name
    if not bundle_dir.exists():
        raise FileNotFoundError(f'Bundle not found: {bundle_dir}')
    metadata = json.loads((bundle_dir / 'metadata.json').read_text())
    train_df = pd.read_csv(bundle_dir / 'train.csv')
    val_df = pd.read_csv(bundle_dir / 'val.csv')
    test_df = pd.read_csv(bundle_dir / 'test.csv')
    return train_df, val_df, test_df, metadata


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:  # older sklearn
        return OneHotEncoder(handle_unknown='ignore', sparse=False)


def detect_gpu_available() -> bool:
    if shutil.which('nvidia-smi'):
        try:
            subprocess.check_output(['nvidia-smi', '-L'], text=True, stderr=subprocess.DEVNULL, timeout=5)
            return True
        except Exception:
            pass
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_visible and cuda_visible.strip() not in {'', '-1'}:
        return True
    return False


def choose_backend(metadata: dict[str, Any]) -> tuple[str, bool]:
    requested = os.environ.get('HARDWARE_BACKEND') or metadata.get('hardware', {}).get('preferred_backend', 'auto')
    requested = requested.lower()
    if requested not in {'auto', 'gpu', 'cpu'}:
        raise ValueError(f'Invalid HARDWARE_BACKEND: {requested}')
    gpu_available = detect_gpu_available()
    if requested == 'cpu':
        return 'cpu', gpu_available
    if requested == 'gpu':
        if not gpu_available:
            raise RuntimeError('HARDWARE_BACKEND=gpu was requested, but no GPU was detected.')
        return 'gpu', gpu_available
    return ('gpu' if gpu_available else 'cpu'), gpu_available


def _safe_import(module_name: str):
    __import__(module_name)
    return __import__(module_name)


def _require_xgboost():
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        raise ImportError('xgboost is required for MODEL_FAMILY=xgboost') from exc
    return XGBClassifier


def _require_lightgbm():
    try:
        from lightgbm import LGBMClassifier
    except Exception as exc:
        raise ImportError('lightgbm is required for MODEL_FAMILY=lightgbm') from exc
    return LGBMClassifier


def _require_catboost():
    try:
        from catboost import CatBoostClassifier
    except Exception as exc:
        raise ImportError('catboost is required for MODEL_FAMILY=catboost') from exc
    return CatBoostClassifier


def _require_cuml_logistic():
    try:
        from cuml.linear_model import LogisticRegression as CuMLLogisticRegression
    except Exception as exc:
        raise ImportError('cuml is required for GPU logistic_regression') from exc
    return CuMLLogisticRegression


def _require_cuml_random_forest():
    try:
        from cuml.ensemble import RandomForestClassifier as CuMLRandomForestClassifier
    except Exception as exc:
        raise ImportError('cuml is required for GPU random_forest') from exc
    return CuMLRandomForestClassifier


def get_model_registry() -> dict[str, ModelSpec]:
    return {
        'logistic_regression': ModelSpec(
            name='logistic_regression',
            needs_scaling_by_default=True,
            cpu_builder=lambda p, rs: LogisticRegression(
                C=float(p.get('C', 1.0)),
                solver=str(p.get('solver', 'lbfgs')),
                class_weight=p.get('class_weight', None),
                max_iter=int(p.get('max_iter', 2000)),
                random_state=rs,
            ),
            gpu_builder=lambda p, rs: _require_cuml_logistic()(
                C=float(p.get('C', 1.0)),
                max_iter=int(p.get('max_iter', 2000)),
                fit_intercept=bool(p.get('fit_intercept', True)),
            ),
        ),
        'random_forest': ModelSpec(
            name='random_forest',
            needs_scaling_by_default=False,
            cpu_builder=lambda p, rs: RandomForestClassifier(
                n_estimators=int(p.get('n_estimators', 300)),
                max_depth=None if p.get('max_depth', None) in [None, 'None'] else int(p['max_depth']),
                min_samples_leaf=int(p.get('min_samples_leaf', 1)),
                class_weight=p.get('class_weight', None),
                n_jobs=1,
                random_state=rs,
            ),
            gpu_builder=lambda p, rs: _require_cuml_random_forest()(
                n_estimators=int(p.get('n_estimators', 300)),
                max_depth=int(p.get('max_depth', 16)),
                max_features=float(p.get('max_features', 1.0)) if isinstance(p.get('max_features', 1.0), (int, float)) else 1.0,
                random_state=rs,
            ),
        ),
        'extra_trees': ModelSpec(
            name='extra_trees',
            needs_scaling_by_default=False,
            cpu_builder=lambda p, rs: ExtraTreesClassifier(
                n_estimators=int(p.get('n_estimators', 300)),
                max_depth=None if p.get('max_depth', None) in [None, 'None'] else int(p['max_depth']),
                min_samples_leaf=int(p.get('min_samples_leaf', 1)),
                class_weight=p.get('class_weight', None),
                n_jobs=1,
                random_state=rs,
            ),
            gpu_builder=None,
        ),
        'xgboost': ModelSpec(
            name='xgboost',
            needs_scaling_by_default=False,
            cpu_builder=lambda p, rs: _require_xgboost()(
                n_estimators=int(p.get('n_estimators', 200)),
                max_depth=int(p.get('max_depth', 6)),
                learning_rate=float(p.get('learning_rate', 0.05)),
                subsample=float(p.get('subsample', 1.0)),
                colsample_bytree=float(p.get('colsample_bytree', 1.0)),
                reg_lambda=float(p.get('reg_lambda', 1.0)),
                objective='binary:logistic',
                eval_metric='logloss',
                tree_method='hist',
                n_jobs=1,
                random_state=rs,
            ),
            gpu_builder=lambda p, rs: _require_xgboost()(
                n_estimators=int(p.get('n_estimators', 200)),
                max_depth=int(p.get('max_depth', 6)),
                learning_rate=float(p.get('learning_rate', 0.05)),
                subsample=float(p.get('subsample', 1.0)),
                colsample_bytree=float(p.get('colsample_bytree', 1.0)),
                reg_lambda=float(p.get('reg_lambda', 1.0)),
                objective='binary:logistic',
                eval_metric='logloss',
                tree_method='hist',
                device='cuda',
                n_jobs=1,
                random_state=rs,
            ),
        ),
        'lightgbm': ModelSpec(
            name='lightgbm',
            needs_scaling_by_default=False,
            cpu_builder=lambda p, rs: _require_lightgbm()(
                n_estimators=int(p.get('n_estimators', 200)),
                learning_rate=float(p.get('learning_rate', 0.05)),
                num_leaves=int(p.get('num_leaves', 31)),
                max_depth=int(p.get('max_depth', -1)),
                subsample=float(p.get('subsample', 1.0)),
                colsample_bytree=float(p.get('colsample_bytree', 1.0)),
                reg_lambda=float(p.get('reg_lambda', 0.0)),
                objective='binary',
                random_state=rs,
                verbose=-1,
                n_jobs=1,
            ),
            gpu_builder=lambda p, rs: _require_lightgbm()(
                n_estimators=int(p.get('n_estimators', 200)),
                learning_rate=float(p.get('learning_rate', 0.05)),
                num_leaves=int(p.get('num_leaves', 31)),
                max_depth=int(p.get('max_depth', -1)),
                subsample=float(p.get('subsample', 1.0)),
                colsample_bytree=float(p.get('colsample_bytree', 1.0)),
                reg_lambda=float(p.get('reg_lambda', 0.0)),
                objective='binary',
                device_type=os.environ.get('LIGHTGBM_DEVICE_TYPE', 'gpu'),
                random_state=rs,
                verbose=-1,
                n_jobs=1,
            ),
        ),
        'catboost': ModelSpec(
            name='catboost',
            needs_scaling_by_default=False,
            cpu_builder=lambda p, rs: _require_catboost()(
                iterations=int(p.get('iterations', 200)),
                depth=int(p.get('depth', 6)),
                learning_rate=float(p.get('learning_rate', 0.05)),
                l2_leaf_reg=float(p.get('l2_leaf_reg', 3.0)),
                loss_function='Logloss',
                eval_metric='Logloss',
                random_seed=rs,
                verbose=False,
                allow_writing_files=False,
                thread_count=1,
            ),
            gpu_builder=lambda p, rs: _require_catboost()(
                iterations=int(p.get('iterations', 200)),
                depth=int(p.get('depth', 6)),
                learning_rate=float(p.get('learning_rate', 0.05)),
                l2_leaf_reg=float(p.get('l2_leaf_reg', 3.0)),
                loss_function='Logloss',
                eval_metric='Logloss',
                task_type='GPU',
                devices=os.environ.get('CATBOOST_DEVICES', '0'),
                random_seed=rs,
                verbose=False,
                allow_writing_files=False,
                thread_count=1,
            ),
        ),
    }


class CatBoostCompatWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator):
        self.estimator = estimator
    def fit(self, X, y):
        self.estimator.fit(X, y)
        self.classes_ = np.array([0, 1])
        return self
    def predict(self, X):
        return self.estimator.predict(X)
    def predict_proba(self, X):
        return self.estimator.predict_proba(X)
    def get_params(self, deep=True):
        return {'estimator': self.estimator}
    def set_params(self, **params):
        if 'estimator' in params:
            self.estimator = params['estimator']
        return self
    def __sklearn_is_fitted__(self):
        return hasattr(self, 'classes_')


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool, dropped: list[str]) -> ColumnTransformer:
    X = X.drop(columns=dropped, errors='ignore')
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    num_steps: list[tuple[str, Any]] = [('imputer', SimpleImputer(strategy='median'))]
    if scale_numeric:
        num_steps.append(('scaler', StandardScaler()))

    cat_steps = [
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', make_one_hot_encoder()),
    ]

    return ColumnTransformer(
        transformers=[
            ('num', Pipeline(num_steps), numeric_cols),
            ('cat', Pipeline(cat_steps), categorical_cols),
        ],
        remainder='drop',
        sparse_threshold=0.0,
    )


def maybe_wrap_estimator(estimator):
    if estimator.__class__.__name__.lower().startswith('catboost'):
        return CatBoostCompatWrapper(estimator)
    return estimator


def build_estimator(family: str, params: dict[str, Any], random_state: int, backend: str):
    registry = get_model_registry()
    if family not in registry:
        raise KeyError(f'Unsupported MODEL_FAMILY={family!r}. Supported={sorted(registry)}')
    spec = registry[family]
    if backend == 'gpu':
        if spec.gpu_builder is None:
            raise RuntimeError(f'Model family {family} does not have a configured GPU path')
        estimator = spec.gpu_builder(params, random_state)
    else:
        estimator = spec.cpu_builder(params, random_state)
    return maybe_wrap_estimator(estimator), spec


def make_pipeline(X: pd.DataFrame, family: str, params: dict[str, Any], feature_cfg: dict[str, Any], random_state: int, backend: str) -> tuple[Pipeline, str]:
    estimator, spec = build_estimator(family, params, random_state, backend)
    scale_numeric = bool(feature_cfg.get('scale_numeric', spec.needs_scaling_by_default))
    dropped = list(feature_cfg.get('drop_columns', []))
    preprocessor = build_preprocessor(X, scale_numeric=scale_numeric, dropped=dropped)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', estimator),
    ])
    return pipeline, backend


def safe_predict_proba(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(pipeline, 'predict_proba'):
        proba = pipeline.predict_proba(X)
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        if proba.ndim == 1:
            return proba
    if hasattr(pipeline, 'decision_function'):
        scores = np.asarray(pipeline.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-scores))
    preds = np.asarray(pipeline.predict(X), dtype=float)
    return np.clip(preds, 0.0, 1.0)


def choose_threshold(y_true: pd.Series, y_prob: np.ndarray, cfg: dict[str, Any]) -> float:
    strategy = str(cfg.get('strategy', 'fixed'))
    if strategy == 'fixed':
        return float(cfg.get('value', 0.5))
    if strategy == 'best_f1':
        thresholds = np.unique(np.round(y_prob, 6))
        if thresholds.size == 0:
            return 0.5
        best_t = 0.5
        best_score = -1.0
        for t in thresholds:
            y_pred = (y_prob >= float(t)).astype(int)
            score = f1_score(y_true, y_pred, zero_division=0)
            if score > best_score:
                best_score = float(score)
                best_t = float(t)
        return best_t
    raise ValueError(f'Unknown threshold strategy: {strategy}')


def compute_metrics(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        'roc_auc': float(roc_auc_score(y_true, y_prob)),
        'average_precision': float(average_precision_score(y_true, y_prob)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'log_loss': float(log_loss(y_true, y_prob)),
        'brier': float(brier_score_loss(y_true, y_prob)),
        'threshold': float(threshold),
    }


def primary_metric_value(metrics: dict[str, float], primary_metric: str) -> float:
    return -float(metrics['log_loss']) if primary_metric == 'neg_log_loss' else float(metrics[primary_metric])


def maybe_init_mlflow(metadata: dict[str, Any]):
    mlcfg = metadata.get('mlflow', {})
    if not mlcfg.get('enabled'):
        return None
    try:
        import mlflow
    except Exception:
        return None
    tracking_uri = mlcfg.get('tracking_uri')
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    exp_name = mlcfg.get('experiment_name') or f"skyportal_{metadata['bundle_name']}"
    mlflow.set_experiment(exp_name)
    return mlflow


def fit_with_backend_fallback(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    family: str,
    params: dict[str, Any],
    feature_cfg: dict[str, Any],
    random_state: int,
    backend_request: str,
) -> tuple[Pipeline, str, Optional[str]]:
    pipeline, actual_backend = make_pipeline(X_train, family, params, feature_cfg, random_state, backend_request)
    try:
        pipeline.fit(X_train.drop(columns=feature_cfg.get('drop_columns', []), errors='ignore'), y_train)
        return pipeline, actual_backend, None
    except Exception as exc:
        if backend_request == 'gpu' and os.environ.get('HARDWARE_BACKEND', '').lower() == 'gpu':
            raise
        if backend_request == 'gpu':
            cpu_pipeline, cpu_backend = make_pipeline(X_train, family, params, feature_cfg, random_state, 'cpu')
            cpu_pipeline.fit(X_train.drop(columns=feature_cfg.get('drop_columns', []), errors='ignore'), y_train)
            return cpu_pipeline, cpu_backend, f'GPU path failed, fell back to CPU: {exc}'
        raise


def write_local_result(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / 'results.tsv'
    run_dir = output_dir / 'runs' / payload['run_id']
    run_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(payload['artifact_pipeline'], run_dir / 'model.joblib')
    (run_dir / 'summary.json').write_text(json.dumps({k: v for k, v in payload.items() if k != 'artifact_pipeline'}, indent=2))

    serializable = {k: v for k, v in payload.items() if k != 'artifact_pipeline'}
    (output_dir / 'last_run.json').write_text(json.dumps(serializable, indent=2))

    flat_row = {
        'run_id': payload['run_id'],
        'experiment_name': payload['experiment_name'],
        'model_family': payload['model_family'],
        'requested_backend': payload['requested_backend'],
        'actual_backend': payload['actual_backend'],
        'primary_metric': payload['primary_metric_name'],
        'primary_metric_value': payload['primary_metric_value'],
        'val_average_precision': payload['val_metrics']['average_precision'],
        'val_roc_auc': payload['val_metrics']['roc_auc'],
        'val_f1': payload['val_metrics']['f1'],
        'val_precision': payload['val_metrics']['precision'],
        'val_recall': payload['val_metrics']['recall'],
        'git_sha': payload['git_sha'],
        'warning': payload.get('warning'),
        'timestamp': payload['timestamp'],
        'params_json': json.dumps(payload['model_params'], sort_keys=True),
    }
    df_new = pd.DataFrame([flat_row])
    if results_path.exists():
        existing = pd.read_csv(results_path, sep='\t')
        merged = pd.concat([existing, df_new], ignore_index=True)
    else:
        merged = df_new
    merged.to_csv(results_path, sep='\t', index=False)


def maybe_log_mlflow(mlflow_mod, payload: dict[str, Any], output_dir: Path,
                     metadata: dict[str, Any]):
    if mlflow_mod is None:
        return
    with mlflow_mod.start_run(run_name=payload['experiment_name']):
        mlflow_mod.log_params({
            'bundle_name': payload['bundle_name'],
            'model_family': payload['model_family'],
            'requested_backend': payload['requested_backend'],
            'actual_backend': payload['actual_backend'],
            **{f'model__{k}': v for k, v in payload['model_params'].items()},
            **{f'feature__{k}': v for k, v in payload['feature_config'].items()},
            **{f'threshold__{k}': v for k, v in payload['threshold_config'].items()},
        })
        mlflow_mod.log_metrics({
            **{f'val_{k}': v for k, v in payload['val_metrics'].items()},
            'primary_metric_value': payload['primary_metric_value'],
        })
        run_dir = output_dir / 'runs' / payload['run_id']
        mlflow_mod.log_artifact(str(run_dir / 'summary.json'))

        if payload.get('finalize') and payload.get('artifact_pipeline') is not None:
            mlcfg = metadata.get('mlflow', {})
            model_name = (mlcfg.get('registered_model_name')
                          or f"autoresearch_{payload['bundle_name']}")
            mlflow_mod.sklearn.log_model(
                sk_model=payload['artifact_pipeline'],
                artifact_path='model',
                registered_model_name=model_name,
            )
        else:
            mlflow_mod.log_artifact(str(run_dir / 'model.joblib'))


def main() -> None:
    args = parse_args()
    train_df, val_df, test_df, metadata = load_bundle(args.bundle_root, args.bundle_name)

    from feature import engineer_features
    train_df = engineer_features(train_df)
    val_df = engineer_features(val_df)
    test_df = engineer_features(test_df)

    target_col = metadata['target_column']
    random_state = int(metadata['split']['random_state'])
    primary_metric_name = metadata['primary_metric']
    output_dir = args.output_root / args.bundle_name

    requested_backend, gpu_available = choose_backend(metadata)
    mlflow_mod = maybe_init_mlflow(metadata)

    if args.finalize:
        last_path = output_dir / 'last_run.json'
        if not last_path.exists():
            raise FileNotFoundError('No previous run found. Run train.py once before --finalize.')
        last = json.loads(last_path.read_text())
        family = last['model_family']
        params = last['model_params']
        feature_cfg = last['feature_config']
        threshold_cfg = last['threshold_config']
        experiment_name = f"finalize_{last['experiment_name']}"
    else:
        family = MODEL_FAMILY
        params = MODEL_PARAMS
        feature_cfg = FEATURE_CONFIG
        threshold_cfg = THRESHOLD_CONFIG
        experiment_name = EXPERIMENT_NAME

    run_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    timestamp = pd.Timestamp.now('UTC').isoformat()

    fit_df = pd.concat([train_df, val_df], ignore_index=True) if args.finalize else train_df
    eval_df = test_df if args.finalize else val_df

    X_fit = fit_df.drop(columns=[target_col])
    y_fit = fit_df[target_col].astype(int)
    X_eval = eval_df.drop(columns=[target_col])
    y_eval = eval_df[target_col].astype(int)

    t0 = time.time()
    pipeline, actual_backend, warning = fit_with_backend_fallback(
        X_fit,
        y_fit,
        family,
        params,
        feature_cfg,
        random_state,
        requested_backend,
    )
    fit_seconds = time.time() - t0

    X_eval_effective = X_eval.drop(columns=feature_cfg.get('drop_columns', []), errors='ignore')
    y_prob = safe_predict_proba(pipeline, X_eval_effective)
    threshold = choose_threshold(y_eval, y_prob, threshold_cfg)
    metrics = compute_metrics(y_eval, y_prob, threshold)
    pmv = primary_metric_value(metrics, primary_metric_name)

    payload = {
        'run_id': run_id,
        'timestamp': timestamp,
        'bundle_name': args.bundle_name,
        'experiment_name': experiment_name,
        'experiment_description': EXPERIMENT_DESCRIPTION,
        'model_family': family,
        'model_params': params,
        'feature_config': feature_cfg,
        'threshold_config': threshold_cfg,
        'requested_backend': requested_backend,
        'actual_backend': actual_backend,
        'gpu_detected': gpu_available,
        'fit_seconds': fit_seconds,
        'primary_metric_name': primary_metric_name,
        'primary_metric_value': pmv,
        'val_metrics': metrics if not args.finalize else {},
        'test_metrics': metrics if args.finalize else {},
        'warning': warning,
        'git_sha': get_git_sha(),
        'feature_engineering_hash': _hash_feature_py(),
        'finalize': bool(args.finalize),
        'artifact_pipeline': pipeline,
    }

    if args.finalize:
        payload['val_metrics'] = last.get('val_metrics', {}) if 'last' in locals() else {}
        payload['test_metrics'] = metrics
    else:
        payload['val_metrics'] = metrics

    write_local_result(output_dir, payload)
    maybe_log_mlflow(mlflow_mod, payload, output_dir, metadata)

    printable = {k: v for k, v in payload.items() if k != 'artifact_pipeline'}
    print(json.dumps(printable, indent=2))


if __name__ == '__main__':
    main()
