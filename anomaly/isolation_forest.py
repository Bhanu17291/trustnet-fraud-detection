# ============================================================
# TrustNet — Anomaly Detection (Isolation Forest)
# Step 5 of Phase 2
# Catches NEW fraud patterns the supervised model never saw
# Run: python anomaly/isolation_forest.py --train
#      python anomaly/isolation_forest.py --score --amount 0.76 --time 8000
# ============================================================

import os, sys, pickle, json
import numpy as np
import pandas as pd
from datetime import datetime

_HERE         = os.path.abspath(__file__)
_ANOMALY_DIR  = os.path.dirname(_HERE)
_PROJECT_ROOT = os.path.dirname(_ANOMALY_DIR)
for _p in [_PROJECT_ROOT, os.getcwd()]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

MODELS_DIR  = os.path.join(_PROJECT_ROOT, 'models')
ANOMALY_DIR = os.path.join(_PROJECT_ROOT, 'anomaly')
MODEL_PATH  = os.path.join(MODELS_DIR, 'isolation_forest.pkl')

FEATURE_NAMES = [
    'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
    'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
    'V21','V22','V23','V24','V25','V26','V27','V28',
    'hour','log_amount','amount_zscore','is_small_amount',
    'is_round_amount','Amount_scaled','Time_scaled'
]
AMOUNT_MEAN = 88.35;  AMOUNT_STD = 250.12
TIME_MEAN   = 94813.0; TIME_STD  = 47488.0

# ── Train Isolation Forest on LEGITIMATE transactions only ────
def train():
    print("\nLoading processed data...")
    with open(os.path.join(MODELS_DIR, 'processed_data.pkl'), 'rb') as f:
        data = pickle.load(f)

    X_train = pd.DataFrame(data['X_train'], columns=FEATURE_NAMES)
    y_train = pd.Series(data['y_train'])

    # Train ONLY on legitimate transactions — anomalies = new fraud
    X_legit = X_train[y_train == 0]
    print(f"Training on {len(X_legit):,} legitimate transactions...")

    iso = IsolationForest(
        n_estimators=200,
        contamination=0.01,   # expect ~1% anomalies in production
        max_samples='auto',
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_legit)

    # Compute anomaly score threshold from training data
    scores = iso.decision_function(X_legit)
    threshold = float(np.percentile(scores, 1))  # bottom 1% = anomaly

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'model': iso, 'threshold': threshold,
                     'trained_at': datetime.utcnow().isoformat(),
                     'trained_on': len(X_legit)}, f)

    print(f"Isolation Forest trained → {MODEL_PATH}")
    print(f"Anomaly threshold: {threshold:.4f}")
    print(f"Trained on: {len(X_legit):,} legit transactions")
    return iso, threshold

# ── Load trained model ────────────────────────────────────────
_iso_cache = {}

def load_iso():
    if 'model' not in _iso_cache:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Isolation Forest not trained yet. Run: python anomaly/isolation_forest.py --train"
            )
        with open(MODEL_PATH, 'rb') as f:
            saved = pickle.load(f)
        _iso_cache['model']     = saved['model']
        _iso_cache['threshold'] = saved['threshold']
        _iso_cache['meta']      = {
            'trained_at': saved.get('trained_at'),
            'trained_on': saved.get('trained_on'),
        }
    return _iso_cache['model'], _iso_cache['threshold']

# ── Score a single transaction ────────────────────────────────
def anomaly_score(features: dict) -> dict:
    """
    features: dict with keys matching FEATURE_NAMES
    Returns anomaly score, flag, and severity.
    """
    iso, threshold = load_iso()
    X = pd.DataFrame([features])[FEATURE_NAMES]

    raw_score  = float(iso.decision_function(X)[0])
    prediction = int(iso.predict(X)[0])   # -1 = anomaly, 1 = normal

    is_anomaly = prediction == -1
    # Normalise: how far below threshold? More negative = more anomalous
    deviation  = threshold - raw_score
    severity   = _severity(raw_score, threshold)

    return {
        'is_anomaly':    is_anomaly,
        'anomaly_score': round(raw_score, 4),
        'threshold':     round(threshold, 4),
        'deviation':     round(deviation, 4),
        'severity':      severity,
    }

def _severity(score: float, threshold: float) -> str:
    if score > threshold + 0.05:
        return 'NORMAL'
    elif score > threshold:
        return 'BORDERLINE'
    elif score > threshold - 0.05:
        return 'ANOMALOUS'
    elif score > threshold - 0.15:
        return 'HIGHLY_ANOMALOUS'
    else:
        return 'EXTREME_ANOMALY'

def get_model_info() -> dict:
    if 'meta' not in _iso_cache:
        try:
            load_iso()
        except FileNotFoundError:
            return {'status': 'not_trained'}
    return {
        'status':     'ready',
        'trained_at': _iso_cache['meta']['trained_at'],
        'trained_on': _iso_cache['meta']['trained_on'],
        'threshold':  round(_iso_cache['threshold'], 4),
    }

# ── CLI ───────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='TrustNet Isolation Forest')
    parser.add_argument('--train',  action='store_true', help='Train the model')
    parser.add_argument('--score',  action='store_true', help='Score a transaction')
    parser.add_argument('--amount', type=float, default=150.0)
    parser.add_argument('--time',   type=float, default=50000.0)
    args = parser.parse_args()

    if args.train:
        train()
    elif args.score:
        hour    = int((args.time % 86400) // 3600)
        log_amt = float(np.log1p(args.amount))
        z       = float((args.amount - AMOUNT_MEAN) / AMOUNT_STD)
        features = {f'V{i}': 0.0 for i in range(1, 29)}
        features.update({
            'hour': hour, 'log_amount': log_amt, 'amount_zscore': z,
            'is_small_amount': int(args.amount < 10),
            'is_round_amount': int(args.amount % 10 == 0),
            'Amount_scaled': z, 'Time_scaled': (args.time - TIME_MEAN) / TIME_STD,
        })
        result = anomaly_score(features)
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()