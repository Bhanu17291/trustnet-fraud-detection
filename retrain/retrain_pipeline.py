# ============================================================
# TrustNet — Feedback Loop & Retraining Pipeline
# Step 4 of Phase 2
# Run manually: python retrain/retrain_pipeline.py
# Scheduled:    runs automatically every Sunday 2am
# ============================================================

import os, sys, pickle, json
import numpy as np
import pandas as pd
from datetime import datetime

_HERE         = os.path.abspath(__file__)
_RETRAIN_DIR  = os.path.dirname(_HERE)
_PROJECT_ROOT = os.path.dirname(_RETRAIN_DIR)
for _p in [_PROJECT_ROOT, os.getcwd()]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

MODELS_DIR  = os.path.join(_PROJECT_ROOT, 'models')
DATA_DIR    = os.path.join(_PROJECT_ROOT, 'data')
RETRAIN_DIR = os.path.join(_PROJECT_ROOT, 'retrain')
LOG_PATH    = os.path.join(RETRAIN_DIR, 'retrain_log.json')

FEATURE_NAMES = [
    'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
    'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
    'V21','V22','V23','V24','V25','V26','V27','V28',
    'hour','log_amount','amount_zscore','is_small_amount',
    'is_round_amount','Amount_scaled','Time_scaled'
]
AMOUNT_MEAN = 88.35;  AMOUNT_STD = 250.12
TIME_MEAN   = 94813.0; TIME_STD  = 47488.0

# ── Load existing training data ───────────────────────────────
def load_base_data() -> tuple[pd.DataFrame, pd.Series]:
    path = os.path.join(MODELS_DIR, 'processed_data.pkl')
    with open(path, 'rb') as f:
        data = pickle.load(f)
    X = pd.DataFrame(data['X_train'], columns=FEATURE_NAMES)
    y = pd.Series(data['y_train'], name='Class')
    print(f"Base training data: {len(X):,} rows  |  fraud rate: {y.mean()*100:.2f}%")
    return X, y

# ── Load analyst feedback from alert queue ────────────────────
def load_feedback() -> tuple[pd.DataFrame, pd.Series]:
    """
    Reads analyst-confirmed cases from the SQLite alert queue.
    Returns empty DataFrames if no feedback yet.
    """
    try:
        from alerts.alert_manager import get_feedback_for_retraining
        rows = get_feedback_for_retraining()
        if not rows:
            print("No analyst feedback found — skipping feedback augmentation.")
            return pd.DataFrame(), pd.Series()

        records = []
        for r in rows:
            rec = {'label': r['label'], 'amount': r['amount'], 'hour': r['hour']}
            records.append(rec)

        df = pd.DataFrame(records)
        print(f"Analyst feedback: {len(df)} cases  |  fraud: {df['label'].sum()}")

        # Build minimal feature rows from feedback (V features unknown, use 0)
        X_fb = pd.DataFrame(0.0, index=df.index, columns=FEATURE_NAMES)
        X_fb['hour']           = df['hour'].values
        X_fb['Amount_scaled']  = (df['amount'] - AMOUNT_MEAN) / AMOUNT_STD
        X_fb['amount_zscore']  = (df['amount'] - AMOUNT_MEAN) / AMOUNT_STD
        X_fb['log_amount']     = np.log1p(df['amount'])
        X_fb['is_small_amount']= (df['amount'] < 10).astype(int)
        X_fb['is_round_amount']= (df['amount'] % 10 == 0).astype(int)
        y_fb = df['label'].astype(int)
        return X_fb, y_fb

    except Exception as e:
        print(f"Could not load feedback: {e}")
        return pd.DataFrame(), pd.Series()

# ── Train new model ───────────────────────────────────────────
def train(X: pd.DataFrame, y: pd.Series) -> tuple:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE only on training split
    print(f"Applying SMOTE to {len(X_train):,} training rows...")
    smote = SMOTE(random_state=42, k_neighbors=min(5, y_train.sum() - 1))
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {len(X_res):,} rows  |  fraud rate: {y_res.mean()*100:.1f}%")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,
        use_label_encoder=False,
        eval_metric='aucpr',
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_res, y_res, eval_set=[(X_val, y_val)], verbose=False)

    probs = model.predict_proba(X_val)[:, 1]
    preds = (probs > 0.5).astype(int)
    metrics = {
        'roc_auc':   round(float(roc_auc_score(y_val, probs)), 4),
        'precision': round(float(precision_score(y_val, preds, zero_division=0)), 4),
        'recall':    round(float(recall_score(y_val, preds, zero_division=0)), 4),
        'f1':        round(float(f1_score(y_val, preds, zero_division=0)), 4),
        'val_size':  len(y_val),
        'fraud_in_val': int(y_val.sum()),
    }
    print(f"New model — ROC-AUC: {metrics['roc_auc']}  |  Recall: {metrics['recall']}  |  Precision: {metrics['precision']}")
    return model, metrics

# ── Compare and decide whether to deploy ─────────────────────
def load_current_metrics() -> dict:
    if not os.path.exists(LOG_PATH):
        return {'roc_auc': 0.0}
    with open(LOG_PATH) as f:
        log = json.load(f)
    return log[-1]['metrics'] if log else {'roc_auc': 0.0}

def save_model(model, metrics: dict, training_rows: int):
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Archive current model with timestamp
    current_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    if os.path.exists(current_path):
        ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        archive_path = os.path.join(MODELS_DIR, f'best_model_{ts}.pkl')
        os.rename(current_path, archive_path)
        print(f"Archived current model → {archive_path}")

    with open(current_path, 'wb') as f:
        pickle.dump({'model': model, 'name': 'XGBoost'}, f)
    print(f"New model saved → {current_path}")

    # Append to retrain log
    log = []
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            log = json.load(f)
    log.append({
        'retrained_at':  datetime.utcnow().isoformat(),
        'training_rows': training_rows,
        'metrics':       metrics,
        'deployed':      True,
    })
    with open(LOG_PATH, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"Retrain log updated → {LOG_PATH}")

# ── Main pipeline ─────────────────────────────────────────────
def run_retraining(force: bool = False) -> dict:
    print(f"\n{'='*55}")
    print(f"TrustNet Retraining Pipeline — {datetime.utcnow().isoformat()}")
    print(f"{'='*55}")

    # Load base data
    X_base, y_base = load_base_data()

    # Augment with analyst feedback
    X_fb, y_fb = load_feedback()
    if len(X_fb) > 0:
        X_all = pd.concat([X_base, X_fb], ignore_index=True)
        y_all = pd.concat([y_base, y_fb], ignore_index=True)
        print(f"Combined dataset: {len(X_all):,} rows")
    else:
        X_all, y_all = X_base, y_base

    # Train
    new_model, new_metrics = train(X_all, y_all)

    # Compare against current model
    current_metrics = load_current_metrics()
    improvement = new_metrics['roc_auc'] - current_metrics.get('roc_auc', 0)
    print(f"\nCurrent ROC-AUC: {current_metrics.get('roc_auc', 'N/A')}")
    print(f"New     ROC-AUC: {new_metrics['roc_auc']}")
    print(f"Improvement:     {improvement:+.4f}")

    if force or improvement >= -0.005:
        save_model(new_model, new_metrics, len(X_all))
        print("\nDeployment: SUCCESS — new model is live.")
        return {'deployed': True, 'metrics': new_metrics, 'improvement': improvement}
    else:
        print(f"\nDeployment: SKIPPED — new model is worse by {abs(improvement):.4f}")
        return {'deployed': False, 'metrics': new_metrics, 'improvement': improvement}

# ── Scheduler (runs weekly) ───────────────────────────────────
def start_scheduler():
    from apscheduler.schedulers.blocking import BlockingScheduler
    scheduler = BlockingScheduler()
    scheduler.add_job(
        run_retraining,
        trigger='cron',
        day_of_week='sun',
        hour=2,
        minute=0,
        id='weekly_retrain',
    )
    print("Retraining scheduler started — runs every Sunday at 2:00 AM UTC")
    print("Press Ctrl+C to stop.")
    try:
        scheduler.start()
    except KeyboardInterrupt:
        print("Scheduler stopped.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='TrustNet Retraining Pipeline')
    parser.add_argument('--run-now',   action='store_true', help='Retrain immediately')
    parser.add_argument('--force',     action='store_true', help='Deploy even if worse')
    parser.add_argument('--schedule',  action='store_true', help='Start weekly scheduler')
    parser.add_argument('--show-log',  action='store_true', help='Show retrain history')
    args = parser.parse_args()

    if args.show_log:
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH) as f:
                log = json.load(f)
            for entry in log:
                print(f"\n{entry['retrained_at']}  |  ROC-AUC: {entry['metrics']['roc_auc']}  |  Deployed: {entry['deployed']}")
        else:
            print("No retrain log found yet.")
    elif args.schedule:
        start_scheduler()
    else:
        run_retraining(force=args.force)