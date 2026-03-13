# ============================================================
# TrustNet — Alert System & Case Management
# Step 3 of Phase 2
# SQLite-backed analyst review queue
# ============================================================

import sqlite3
import os
import json
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from typing import Optional

DB_PATH = os.path.join(os.getcwd(), 'alerts', 'cases.db')

class CaseStatus(str, Enum):
    OPEN     = "OPEN"
    REVIEWED = "REVIEWED"
    APPROVED = "APPROVED"   # analyst confirmed: real fraud
    REJECTED = "REJECTED"   # analyst dismissed: false alarm

class RiskLevel(str, Enum):
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class Case:
    transaction_id: str
    fraud_score:    float
    risk_level:     str
    decision:       str
    amount:         float
    hour:           int
    rules_fired:    str       # JSON string
    model_used:     str
    scored_at:      str
    status:         str = "OPEN"
    analyst_note:   str = ""
    reviewed_at:    str = ""
    id:             Optional[int] = None

# ── Database setup ────────────────────────────────────────────
def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT    NOT NULL UNIQUE,
            fraud_score    REAL    NOT NULL,
            risk_level     TEXT    NOT NULL,
            decision       TEXT    NOT NULL,
            amount         REAL    NOT NULL,
            hour           INTEGER NOT NULL,
            rules_fired    TEXT    NOT NULL,
            model_used     TEXT    NOT NULL,
            scored_at      TEXT    NOT NULL,
            status         TEXT    NOT NULL DEFAULT 'OPEN',
            analyst_note   TEXT    DEFAULT '',
            reviewed_at    TEXT    DEFAULT '',
            created_at     TEXT    NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_status    ON cases(status);
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_risk      ON cases(risk_level);
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_scored_at ON cases(scored_at);
    """)
    conn.commit()
    conn.close()
    print(f"Database initialised at {DB_PATH}")

# ── Core operations ───────────────────────────────────────────
def create_case(
    transaction_id: str,
    fraud_score:    float,
    risk_level:     str,
    decision:       str,
    amount:         float,
    hour:           int,
    rules_fired:    list[str],
    model_used:     str,
    scored_at:      str,
) -> Optional[int]:
    """
    Insert a new case into the queue.
    Returns the new case ID, or None if the transaction already exists.
    Only REVIEW and BLOCK decisions are stored — ALLOW transactions are skipped.
    """
    if decision == "ALLOW":
        return None

    conn = get_connection()
    try:
        cursor = conn.execute("""
            INSERT OR IGNORE INTO cases
            (transaction_id, fraud_score, risk_level, decision,
             amount, hour, rules_fired, model_used, scored_at)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            transaction_id, fraud_score, risk_level, decision,
            amount, hour, json.dumps(rules_fired), model_used, scored_at
        ))
        conn.commit()
        return cursor.lastrowid if cursor.rowcount else None
    finally:
        conn.close()

def get_open_cases(
    risk_level: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """Fetch open cases for the analyst queue, highest risk first."""
    conn = get_connection()
    risk_order = "CASE risk_level WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2 WHEN 'MEDIUM' THEN 3 ELSE 4 END"
    query = f"SELECT * FROM cases WHERE status = 'OPEN'"
    params = []
    if risk_level:
        query += " AND risk_level = ?"
        params.append(risk_level)
    query += f" ORDER BY {risk_order}, fraud_score DESC LIMIT ? OFFSET ?"
    params += [limit, offset]
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]

def get_case(case_id: int) -> Optional[dict]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
    conn.close()
    return _row_to_dict(row) if row else None

def update_case_status(
    case_id:      int,
    status:       str,
    analyst_note: str = "",
) -> bool:
    """Analyst approves (confirmed fraud) or rejects (false alarm) a case."""
    if status not in [s.value for s in CaseStatus]:
        raise ValueError(f"Invalid status: {status}")
    conn = get_connection()
    cursor = conn.execute("""
        UPDATE cases
        SET status = ?, analyst_note = ?, reviewed_at = ?
        WHERE id = ?
    """, (status, analyst_note, datetime.utcnow().isoformat(), case_id))
    conn.commit()
    conn.close()
    return cursor.rowcount > 0

def get_queue_stats() -> dict:
    """Dashboard summary stats for the analyst queue."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT
            COUNT(*)                                          AS total,
            SUM(CASE WHEN status='OPEN'     THEN 1 ELSE 0 END) AS open,
            SUM(CASE WHEN status='APPROVED' THEN 1 ELSE 0 END) AS confirmed_fraud,
            SUM(CASE WHEN status='REJECTED' THEN 1 ELSE 0 END) AS false_alarms,
            SUM(CASE WHEN risk_level='CRITICAL' AND status='OPEN' THEN 1 ELSE 0 END) AS critical_open,
            SUM(CASE WHEN risk_level='HIGH'     AND status='OPEN' THEN 1 ELSE 0 END) AS high_open,
            AVG(CASE WHEN status='OPEN' THEN fraud_score END)  AS avg_open_score
        FROM cases
    """).fetchone()
    conn.close()
    r = dict(rows)
    total_reviewed = (r['confirmed_fraud'] or 0) + (r['false_alarms'] or 0)
    r['precision'] = (
        f"{r['confirmed_fraud']/total_reviewed*100:.1f}%"
        if total_reviewed > 0 else "N/A"
    )
    r['avg_open_score'] = round(r['avg_open_score'] or 0, 4)
    return r

def get_feedback_for_retraining() -> list[dict]:
    """
    Export analyst-reviewed cases for the retraining pipeline (Step 4).
    Returns confirmed fraud (label=1) and confirmed legit (label=0).
    """
    conn = get_connection()
    rows = conn.execute("""
        SELECT transaction_id, fraud_score, amount, hour,
               rules_fired, reviewed_at,
               CASE WHEN status='APPROVED' THEN 1 ELSE 0 END AS label
        FROM cases
        WHERE status IN ('APPROVED', 'REJECTED')
        ORDER BY reviewed_at DESC
    """).fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]

def _row_to_dict(row) -> dict:
    d = dict(row)
    if 'rules_fired' in d and isinstance(d['rules_fired'], str):
        try:
            d['rules_fired'] = json.loads(d['rules_fired'])
        except Exception:
            pass
    return d

# ── Initialise DB on import ───────────────────────────────────
init_db()