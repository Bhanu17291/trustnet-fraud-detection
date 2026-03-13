# ============================================================
# TrustNet — Rule Engine with Dynamic Thresholds
# Step 2 of Phase 2
# Sits on top of the ML score: block · flag · allow
# ============================================================

from dataclasses import dataclass, field
from typing import Optional
import json
import os

# ── Rule config (edit thresholds here without touching code) ──
DEFAULT_CONFIG = {
    "thresholds": {
        "block":  0.80,   # score >= this → BLOCK immediately
        "review": 0.30,   # score >= this → send to analyst queue
        "allow":  0.00    # score <  review → ALLOW
    },
    "amount_rules": {
        "high_value_threshold": 5000.0,   # flag high-value tx regardless of score
        "micro_tx_threshold":   1.0,      # flag card-testing micro-transactions
        "high_value_review_score": 0.15   # lower review threshold for high-value tx
    },
    "velocity_rules": {
        # These are evaluated if you pass tx_count_last_hour from your system
        "max_tx_per_hour": 10,
        "max_amount_per_hour": 10000.0
    }
}

CONFIG_PATH = os.path.join(os.getcwd(), 'rules', 'config.json')

def load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return DEFAULT_CONFIG

def save_config(config: dict):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {CONFIG_PATH}")

@dataclass
class RuleResult:
    decision:      str            # ALLOW / REVIEW / BLOCK
    risk_level:    str            # LOW / MEDIUM / HIGH / CRITICAL
    fraud_score:   float
    rules_fired:   list[str] = field(default_factory=list)
    override:      bool = False   # True if a hard rule overrode the ML score

class RuleEngine:
    """
    Applies business rules on top of the ML fraud score.
    Rules are evaluated in priority order — first match wins for BLOCK/REVIEW.
    ALLOW only fires if no other rule matches.
    """

    def __init__(self):
        self.config = load_config()

    def reload(self):
        """Hot-reload config without restarting the API."""
        self.config = load_config()

    def evaluate(
        self,
        fraud_score: float,
        amount: float,
        hour: int,
        tx_count_last_hour: Optional[int] = None,
        amount_last_hour: Optional[float] = None,
    ) -> RuleResult:

        cfg   = self.config
        th    = cfg['thresholds']
        ar    = cfg['amount_rules']
        vr    = cfg['velocity_rules']
        rules = []

        # ── Hard block rules (override ML score) ──────────────
        if fraud_score >= th['block']:
            rules.append(f"ML score {fraud_score:.2f} >= block threshold {th['block']}")
            return RuleResult("BLOCK", "CRITICAL", fraud_score, rules, override=False)

        # Velocity: too many transactions in last hour
        if tx_count_last_hour is not None and tx_count_last_hour > vr['max_tx_per_hour']:
            rules.append(f"Velocity: {tx_count_last_hour} tx in last hour > limit {vr['max_tx_per_hour']}")
            return RuleResult("BLOCK", "CRITICAL", fraud_score, rules, override=True)

        # Velocity: too much spend in last hour
        if amount_last_hour is not None and amount_last_hour > vr['max_amount_per_hour']:
            rules.append(f"Velocity: ${amount_last_hour:.2f} in last hour > limit ${vr['max_amount_per_hour']:.2f}")
            return RuleResult("BLOCK", "CRITICAL", fraud_score, rules, override=True)

        # ── Review rules ───────────────────────────────────────
        if fraud_score >= th['review']:
            rules.append(f"ML score {fraud_score:.2f} >= review threshold {th['review']}")
            risk = "HIGH" if fraud_score >= 0.5 else "MEDIUM"
            return RuleResult("REVIEW", risk, fraud_score, rules)

        # High-value transaction with even modest fraud signal
        if amount >= ar['high_value_threshold'] and fraud_score >= ar['high_value_review_score']:
            rules.append(f"High-value tx ${amount:.2f} with score {fraud_score:.2f}")
            return RuleResult("REVIEW", "MEDIUM", fraud_score, rules, override=True)

        # Micro transaction (card testing pattern)
        if amount <= ar['micro_tx_threshold']:
            rules.append(f"Micro-transaction ${amount:.2f} flagged for card-testing pattern")
            return RuleResult("REVIEW", "MEDIUM", fraud_score, rules, override=True)

        # Unusual hour (2am–5am) with any fraud signal
        if hour in range(2, 6) and fraud_score > 0.10:
            rules.append(f"Unusual hour ({hour}:00) with score {fraud_score:.2f}")
            return RuleResult("REVIEW", "LOW", fraud_score, rules, override=True)

        # ── Allow ──────────────────────────────────────────────
        rules.append(f"ML score {fraud_score:.2f} below all thresholds — clean")
        return RuleResult("ALLOW", "LOW", fraud_score, rules)


# ── Singleton — import this in your API ───────────────────────
rule_engine = RuleEngine()


# ── CLI: update thresholds interactively ──────────────────────
if __name__ == "__main__":
    import sys
    config = load_config()
    print("\nCurrent thresholds:")
    print(json.dumps(config['thresholds'], indent=2))

    if "--set-block" in sys.argv:
        idx = sys.argv.index("--set-block")
        config['thresholds']['block'] = float(sys.argv[idx + 1])
    if "--set-review" in sys.argv:
        idx = sys.argv.index("--set-review")
        config['thresholds']['review'] = float(sys.argv[idx + 1])

    save_config(config)
    print("\nUpdated thresholds:")
    print(json.dumps(config['thresholds'], indent=2))