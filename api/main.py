# ============================================================
# TrustNet — Real-time Fraud Scoring API
# Phase 2 · Steps 1 + 2 + 3
# Run with: uvicorn api.main:app --reload --port 8000
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import pickle
import os
import sys
from datetime import datetime

# Ensure project root is on the path (works on Windows + uvicorn)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from rules.rule_engine import rule_engine
from alerts.alert_manager import (
    create_case, get_open_cases, get_case,
    update_case_status, get_queue_stats, get_feedback_for_retraining
)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

FEATURE_NAMES = [
    'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
    'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
    'V21','V22','V23','V24','V25','V26','V27','V28',
    'hour','log_amount','amount_zscore','is_small_amount',
    'is_round_amount','Amount_scaled','Time_scaled'
]
AMOUNT_MEAN = 88.35;  AMOUNT_STD = 250.12
TIME_MEAN   = 94813.0; TIME_STD  = 47488.0

def load_model():
    path = os.path.join(MODELS_DIR, 'best_model.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    with open(path, 'rb') as f:
        saved = pickle.load(f)
    return saved['model'], saved['name']

try:
    MODEL, MODEL_NAME = load_model()
    MODEL_LOADED = True
except FileNotFoundError as e:
    MODEL, MODEL_NAME = None, None
    MODEL_LOADED = False
    print(f"WARNING: {e}")

app = FastAPI(
    title="TrustNet Fraud Scoring API",
    description="Real-time fraud scoring · XGBoost + Rule Engine + Alert Queue",
    version="3.0.0"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Schemas ───────────────────────────────────────────────────
class TransactionRequest(BaseModel):
    V1:float=0.0;  V2:float=0.0;  V3:float=0.0;  V4:float=0.0
    V5:float=0.0;  V6:float=0.0;  V7:float=0.0;  V8:float=0.0
    V9:float=0.0;  V10:float=0.0; V11:float=0.0; V12:float=0.0
    V13:float=0.0; V14:float=0.0; V15:float=0.0; V16:float=0.0
    V17:float=0.0; V18:float=0.0; V19:float=0.0; V20:float=0.0
    V21:float=0.0; V22:float=0.0; V23:float=0.0; V24:float=0.0
    V25:float=0.0; V26:float=0.0; V27:float=0.0; V28:float=0.0
    Amount: float = Field(..., gt=0)
    Time:   float = Field(..., ge=0)
    tx_count_last_hour: int   = Field(None)
    amount_last_hour:   float = Field(None)

class ScoreResponse(BaseModel):
    transaction_id: str
    fraud_score:    float
    fraud_score_pct:str
    risk_level:     str
    decision:       str
    is_fraud:       bool
    rules_fired:    list[str]
    rule_override:  bool
    case_id:        int | None   # set if a case was created in the alert queue
    model_used:     str
    scored_at:      str
    engineered:     dict

class CaseUpdateRequest(BaseModel):
    status:       str = Field(..., description="APPROVED or REJECTED")
    analyst_note: str = Field("", description="Optional note from analyst")

# ── Feature engineering ───────────────────────────────────────
def engineer(tx: TransactionRequest):
    hour    = int((tx.Time % 86400) // 3600)
    log_amt = float(np.log1p(tx.Amount))
    z       = float((tx.Amount - AMOUNT_MEAN) / AMOUNT_STD)
    small   = int(tx.Amount < 10)
    rnd     = int(tx.Amount % 10 == 0)
    amt_sc  = float((tx.Amount - AMOUNT_MEAN) / AMOUNT_STD)
    time_sc = float((tx.Time - TIME_MEAN) / TIME_STD)
    row = {f'V{i}': getattr(tx, f'V{i}') for i in range(1, 29)}
    row.update({'hour':hour,'log_amount':log_amt,'amount_zscore':z,
                'is_small_amount':small,'is_round_amount':rnd,
                'Amount_scaled':amt_sc,'Time_scaled':time_sc})
    eng = {'hour':hour,'log_amount':round(log_amt,4),'amount_zscore':round(z,4),
           'is_small_amount':small,'is_round_amount':rnd,
           'Amount_scaled':round(amt_sc,4),'Time_scaled':round(time_sc,4)}
    return pd.DataFrame([row])[FEATURE_NAMES], eng, hour

# ── Core endpoints ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"service":"TrustNet","version":"3.0.0",
            "status":"online" if MODEL_LOADED else "model_not_loaded",
            "model":MODEL_NAME,"docs":"/docs"}

@app.get("/health")
def health():
    return {"status":"healthy" if MODEL_LOADED else "degraded",
            "model_loaded":MODEL_LOADED,"model_name":MODEL_NAME,
            "timestamp":datetime.utcnow().isoformat()}

@app.post("/score", response_model=ScoreResponse)
def score(tx: TransactionRequest):
    if not MODEL_LOADED:
        raise HTTPException(503, "Model not loaded.")
    X, eng, hour = engineer(tx)
    fraud_score = float(MODEL.predict_proba(X)[0][1])
    result = rule_engine.evaluate(fraud_score, tx.Amount, hour,
                                  tx.tx_count_last_hour, tx.amount_last_hour)
    tx_id  = f"TX-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"
    scored = datetime.utcnow().isoformat()

    # Auto-create alert case for REVIEW and BLOCK decisions
    case_id = create_case(
        transaction_id = tx_id,
        fraud_score    = fraud_score,
        risk_level     = result.risk_level,
        decision       = result.decision,
        amount         = tx.Amount,
        hour           = hour,
        rules_fired    = result.rules_fired,
        model_used     = MODEL_NAME,
        scored_at      = scored,
    )

    return ScoreResponse(
        transaction_id  = tx_id,
        fraud_score     = round(fraud_score, 6),
        fraud_score_pct = f"{fraud_score*100:.1f}%",
        risk_level      = result.risk_level,
        decision        = result.decision,
        is_fraud        = result.decision == "BLOCK",
        rules_fired     = result.rules_fired,
        rule_override   = result.override,
        case_id         = case_id,
        model_used      = MODEL_NAME,
        scored_at       = scored,
        engineered      = eng,
    )

@app.post("/score/batch")
def score_batch(transactions: list[TransactionRequest]):
    if not MODEL_LOADED:
        raise HTTPException(503, "Model not loaded.")
    if len(transactions) > 10000:
        raise HTTPException(400, "Batch limit is 10,000.")
    results = []
    for tx in transactions:
        X, _, hour = engineer(tx)
        s = float(MODEL.predict_proba(X)[0][1])
        r = rule_engine.evaluate(s, tx.Amount, hour,
                                 tx.tx_count_last_hour, tx.amount_last_hour)
        tx_id  = f"TX-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"
        case_id = create_case(tx_id, s, r.risk_level, r.decision,
                              tx.Amount, hour, r.rules_fired, MODEL_NAME,
                              datetime.utcnow().isoformat())
        results.append({"fraud_score":round(s,6),"risk_level":r.risk_level,
                         "decision":r.decision,"case_id":case_id})
    blocked = sum(1 for r in results if r['decision']=='BLOCK')
    review  = sum(1 for r in results if r['decision']=='REVIEW')
    return {"total":len(results),"blocked":blocked,"review":review,
            "allowed":len(results)-blocked-review,
            "block_rate":f"{blocked/len(results)*100:.2f}%","results":results}

# ── Alert queue endpoints ──────────────────────────────────────
@app.get("/alerts/queue")
def get_queue(risk_level: str = None, limit: int = 50, offset: int = 0):
    """Analyst review queue — open cases, highest risk first."""
    cases = get_open_cases(risk_level=risk_level, limit=limit, offset=offset)
    return {"count": len(cases), "cases": cases}

@app.get("/alerts/stats")
def queue_stats():
    """Dashboard summary: open cases, confirmed fraud, false alarms, precision."""
    return get_queue_stats()

@app.get("/alerts/case/{case_id}")
def get_single_case(case_id: int):
    case = get_case(case_id)
    if not case:
        raise HTTPException(404, f"Case {case_id} not found")
    return case

@app.patch("/alerts/case/{case_id}")
def review_case(case_id: int, body: CaseUpdateRequest):
    """Analyst approves (confirmed fraud) or rejects (false alarm) a case."""
    if body.status not in ("APPROVED", "REJECTED"):
        raise HTTPException(400, "status must be APPROVED or REJECTED")
    updated = update_case_status(case_id, body.status, body.analyst_note)
    if not updated:
        raise HTTPException(404, f"Case {case_id} not found")
    return {"case_id": case_id, "status": body.status, "note": body.analyst_note}

@app.get("/alerts/feedback")
def feedback_export():
    """Export analyst-reviewed cases for retraining pipeline (Step 4)."""
    rows = get_feedback_for_retraining()
    return {"count": len(rows), "feedback": rows}

# ── Rules endpoints ────────────────────────────────────────────
@app.get("/rules/config")
def get_rules():
    return rule_engine.config

@app.post("/rules/reload")
def reload_rules():
    rule_engine.reload()
    return {"status": "reloaded", "config": rule_engine.config}

# ── Retraining endpoints (Step 4) ─────────────────────────────
@app.get("/retrain/log")
def retrain_log():
    """Show history of all retraining runs."""
    import json
    log_path = os.path.join(_PROJECT_ROOT, 'retrain', 'retrain_log.json')
    if not os.path.exists(log_path):
        return {"runs": [], "message": "No retraining runs yet."}
    with open(log_path) as f:
        log = json.load(f)
    return {"runs": len(log), "history": log}

@app.post("/retrain/trigger")
def trigger_retrain(force: bool = False):
    """
    Manually trigger a retraining run.
    Set force=true to deploy even if new model is slightly worse.
    WARNING: This blocks the request for ~60 seconds while training.
    """
    import threading
    from retrain.retrain_pipeline import run_retraining

    result = {}
    def _run():
        result.update(run_retraining(force=force))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=300)

    return {
        "status":      "completed" if result else "timeout",
        "deployed":    result.get("deployed"),
        "improvement": result.get("improvement"),
        "metrics":     result.get("metrics"),
    }