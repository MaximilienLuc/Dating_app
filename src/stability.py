"""Session-level stability analysis for the group's frozen data contract."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def session_sample(groups, rng):
    """Draw whole sessions with replacement, preserving multiplicity and pairs."""
    groups = np.asarray(groups)
    unique = np.unique(groups)
    return np.concatenate([np.flatnonzero(groups == g)
                           for g in rng.choice(unique, len(unique), replace=True)])


def model_features(name, contract):
    """Preserve the contract's feature order, independently for each estimator."""
    selected = contract.get('features_logit', contract['features']) if name == 'logit' else contract['features']
    if not isinstance(selected, list) or not selected or not all(isinstance(c, str) for c in selected):
        raise ValueError(f'{name}: feature list must be a nonempty list of column names')
    if len(selected) != len(set(selected)):
        raise ValueError(f'{name}: duplicate features')
    forbidden = set(contract['protected']) | {contract['target'], 'iid', 'pid', 'wave', 'dec', 'match', 'dec_o'}
    if forbidden.intersection(selected):
        raise ValueError('Protected attributes, identifiers or outcomes in features')
    return list(selected)


def validate_contract(data, contract, split):
    features = list(dict.fromkeys(model_features('logit', contract) + model_features('xgb', contract)))
    if len(features) != len(set(features)):
        raise ValueError('Duplicate features')
    forbidden = set(contract['protected']) | {'iid', 'pid', 'wave', 'dec', 'match', 'dec_o'}
    if forbidden.intersection(features):
        raise ValueError('Protected attributes, identifiers or outcomes in features')
    required = set(features) | {'iid', 'pid', 'wave', contract['target']}
    if required - set(data):
        raise ValueError(f'Missing columns: {required - set(data)}')
    if split['group'] != 'wave':
        raise ValueError('The analysis requires session-level splits')
    train_waves, test_waves = set(split['train_waves']), set(split['test_waves'])
    if not train_waves or not test_waves or train_waves & test_waves:
        raise ValueError('Train/test sessions must be nonempty and disjoint')
    if set(data.wave.unique()) != train_waves | test_waves:
        raise ValueError('Split does not cover the data exactly')
    train, test = data[data.wave.isin(train_waves)].copy(), data[data.wave.isin(test_waves)].copy()
    train_people = set(train.iid) | set(train.pid)
    test_people = set(test.iid) | set(test.pid)
    if train_people & test_people:
        raise ValueError('Participant overlap between train and test')
    for frame in (train, test):
        if frame[['iid', 'pid', 'wave', contract['target']]].isna().any().any():
            raise ValueError('Missing identifier, session or target')
        if set(frame[contract['target']].unique()) != {0, 1}:
            raise ValueError('Each split must contain both binary classes')
        if not all(pd.api.types.is_numeric_dtype(frame[c]) for c in features):
            raise ValueError('Features must be numeric')
        if np.isinf(frame[features].to_numpy(dtype=float)).any():
            raise ValueError('Infinite feature values')
    return train.reset_index(drop=True), test.reset_index(drop=True)


def model_factory(name, seed=42):
    """V0 hyperparameters; no test-set tuning and no deserializing model pickles."""
    if name == 'logit':
        return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                             StandardScaler(), LogisticRegression(max_iter=5000))
    if name == 'xgb':
        return XGBClassifier(n_estimators=400, max_depth=4, learning_rate=.05,
                             subsample=.8, colsample_bytree=.8, eval_metric='logloss',
                             random_state=seed, n_jobs=2)
    raise ValueError(f'Unknown model: {name}')


def importance_vector(model, name, reference_sd):
    if name == 'logit':
        # Convert from each refit's scaler to common units: one original-train SD.
        beta = model[-1].coef_[0] / model[-2].scale_
        return beta * reference_sd
    return model.feature_importances_


def cosine_distance(a, b):
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(1 - np.clip(np.dot(a, b) / norm, -1, 1)) if norm else np.nan


def auc_or_nan(y, p):
    return float(roc_auc_score(y, p)) if np.unique(y).size == 2 else np.nan


def interval(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {'mean': None, 'p025': None, 'p975': None, 'valid': 0}
    return dict(mean=float(values.mean()), p025=float(np.quantile(values, .025)),
                p975=float(np.quantile(values, .975)), valid=len(values))


def frozen_tabicl_predictions(test, predictions):
    """One-to-one key join; never rely on export row order or load a model."""
    keys = ['iid', 'pid', 'wave']
    required = keys + ['tabicl_proba']
    if set(required) - set(predictions):
        raise ValueError('TabICL prediction columns missing')
    if predictions[keys].isna().any().any() or predictions.duplicated(keys).any():
        raise ValueError('TabICL prediction keys must be non-null and unique')
    if test.duplicated(keys).any():
        raise ValueError('Test encounter keys are not unique')
    joined = test[keys].merge(predictions[required], on=keys, how='left', sort=False, validate='one_to_one')
    p = joined.tabicl_proba.to_numpy(dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Missing, non-finite or out-of-range TabICL test predictions')
    return p


def run(root, output, n_refits=200, n_eval=2000, seed=42, threshold=.5):
    root, output = Path(root), Path(output)
    if n_refits < 2 or n_eval < 2 or not 0 < threshold < 1:
        raise ValueError('At least two replicates and a threshold strictly between 0 and 1 required')
    contract = json.loads((root / 'data/features.json').read_text())
    split = json.loads((root / 'data/split.json').read_text())
    data = pd.read_parquet(root / 'data/clean.parquet')
    train, test = validate_contract(data, contract, split)
    features, target = contract['features'], contract['target']
    y, yt = train[target].to_numpy(), test[target].to_numpy()
    names = ['logit', 'xgb']
    features_by_model = {name: model_features(name, contract) for name in names}
    X = {name: train[features_by_model[name]] for name in names}
    Xt = {name: test[features_by_model[name]] for name in names}
    reference_sd = X['logit'].std(ddof=0).fillna(0).to_numpy()
    baselines, scores, vectors = {}, {}, {}
    for name in names:
        model = model_factory(name, seed).fit(X[name], y)
        baselines[name] = model
        scores[name] = model.predict_proba(Xt[name])[:, 1]
        vectors[name] = importance_vector(model, name, reference_sd)
        print(f'{name}: reference AUC {auc_or_nan(yt, scores[name]):.4f}', flush=True)

    # Paired draws: every model sees the same resampled sessions.
    rng = np.random.default_rng(seed)
    draws = [session_sample(train.wave, rng) for _ in range(n_refits)]
    refits, coefficients = [], []
    for b, indices in enumerate(draws):
        if np.unique(y[indices]).size < 2:
            raise ValueError('One-class training bootstrap; inspect the dataset')
        for name in names:
            model = model_factory(name, seed).fit(X[name].iloc[indices], y[indices])
            p = model.predict_proba(Xt[name])[:, 1]
            importance = importance_vector(model, name, reference_sd)
            refits.append(dict(model=name, replicate=b, auc=auc_or_nan(yt, p),
                               flip_rate=float(np.mean((p >= threshold) != (scores[name] >= threshold))),
                               probability_mae=float(np.mean(np.abs(p - scores[name]))),
                               vector_cosine_distance=cosine_distance(importance, vectors[name]),
                               vector_euclidean_distance=float(np.linalg.norm(importance - vectors[name]))))
            coefficients.append(importance if name == 'logit' else None)
        if (b + 1) % 20 == 0 or b + 1 == n_refits:
            print(f'{b + 1}/{n_refits} paired training-session refits', flush=True)
    refits = pd.DataFrame(refits)
    betas = np.stack([v for v in coefficients if v is not None])
    coef = pd.DataFrame({'feature': features_by_model['logit'], 'reference_beta': vectors['logit'],
                         'p025': np.quantile(betas, .025, axis=0),
                         'p975': np.quantile(betas, .975, axis=0),
                         'sign_agreement': np.mean(np.sign(betas) == np.sign(vectors['logit']), axis=0)})

    prediction_path = root / 'data/tabicl_predictions.parquet'
    if prediction_path.exists():
        scores['tabicl'] = frozen_tabicl_predictions(test, pd.read_parquet(prediction_path))
    evaluated_names = list(scores)

    # Separate test uncertainty: fixed fitted models, resampling only test sessions.
    rng_eval = np.random.default_rng(seed + 1)
    evaluations = []
    for b in range(n_eval):
        idx = session_sample(test.wave, rng_eval)
        row = {'replicate': b, **{f'{name}_auc': auc_or_nan(yt[idx], scores[name][idx]) for name in evaluated_names}}
        for i, left in enumerate(evaluated_names):
            for right in evaluated_names[i+1:]:
                row[f'{left}_minus_{right}'] = row[f'{left}_auc'] - row[f'{right}_auc']
        evaluations.append(row)
    evaluations = pd.DataFrame(evaluations)
    per_session = []
    for wave in sorted(test.wave.unique()):
        mask = test.wave.to_numpy() == wave
        for name in evaluated_names:
            per_session.append(dict(model=name, wave=int(wave), n=int(mask.sum()),
                                    positive_rate=float(yt[mask].mean()), auc=auc_or_nan(yt[mask], scores[name][mask])))
    summary = {
        'status': 'current_feature_contract', 'models_run': evaluated_names,
        'models_refitted': names, 'models_test_evaluated': evaluated_names,
        'tabicl': {'status': 'test_only' if 'tabicl' in scores else 'pending', 'reason': 'Frozen predictions only; no training resamples, decision-flip analysis or importance-vector distances for TabICL.'},
        'features_by_model': features_by_model,
        'logit_feature_source': 'features_logit' if 'features_logit' in contract else 'features (legacy fallback; reduced contract not supplied)',
        'train_rows': len(train), 'test_rows': len(test), 'features': len(features),
        'train_waves': sorted(split['train_waves']), 'test_waves': sorted(split['test_waves']),
        'seed': seed, 'threshold': threshold, 'n_refits': n_refits, 'n_eval': n_eval,
        'method': {
            'training': 'Paired session bootstrap; fixed hyperparameters and estimator seed; refit preprocessing on each training draw; fixed test set.',
            'evaluation': 'Paired session bootstrap of fixed-model test predictions; row-weighted pooled AUC.',
            'coefficient_units': 'Original training-set standard deviations; signed logit coefficients.',
            'xgb_vector': 'Normalized gain importance, compared only within XGBoost; not comparable to coefficient distances.',
            'threshold': '0.5 is a provisional display rule, not a tuned business threshold.' if threshold == .5 else 'Externally supplied fixed threshold; this script does not optimize it.',
        },
        'limitations': [f'Only {test.wave.nunique()} test sessions: uncertainty estimates are exploratory.',
                        'Refit percentile ranges describe sensitivity, not confidence intervals for the best model.',
                        'The v0 feature set includes imprace_A/B; the group must decide whether to retain them.',
                        'Results must be regenerated after any data, feature, model or threshold change.',
                        'TabICL is evaluated on frozen predictions only, not training sensitivity. No production-readiness conclusion.'],
        'models': {},
        'paired_auc_difference': interval(evaluations.logit_minus_xgb),
        'paired_auc_differences': {c: interval(evaluations[c]) for c in evaluations if '_minus_' in c},
        'inputs_sha256': {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                          for name in ['data/clean.parquet', 'data/features.json', 'data/split.json', '01_data_models_v0.py', 'src/stability.py'] + (['data/tabicl_predictions.parquet'] if prediction_path.exists() else [])},
        'versions': {name: importlib.metadata.version(name) for name in ['numpy','pandas','scikit-learn','xgboost']},
    }
    for name in evaluated_names:
        subset = refits[refits.model == name]
        summary['models'][name] = {
            'reference_auc': auc_or_nan(yt, scores[name]),
            'test_auc_interval': interval(evaluations[f'{name}_auc']),
            **{col: interval(subset[col]) for col in ['auc','flip_rate','probability_mae','vector_cosine_distance','vector_euclidean_distance'] if name in names},
            'training_stability_available': name in names,
        }
    output.mkdir(parents=True, exist_ok=True)
    refits.to_csv(output/'refits.csv', index=False)
    evaluations.to_csv(output/'test_bootstrap.csv', index=False)
    coef.to_csv(output/'logit_coefficients.csv', index=False)
    pd.DataFrame(per_session).to_csv(output/'by_session.csv', index=False)
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, default=Path('reports/stability'))
    parser.add_argument('--refits', type=int, default=200)
    parser.add_argument('--eval-bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threshold', type=float, default=.5)
    args = parser.parse_args()
    run(args.root, args.output, args.refits, args.eval_bootstrap, args.seed, args.threshold)
