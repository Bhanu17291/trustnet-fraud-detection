"""
TrustNet API — quick smoke test
Run with: python test_api.py
(API must be running: uvicorn api.main:app --reload --port 8000)
"""

import httpx
import json

BASE = "http://localhost:8000"

def test_health():
    r = httpx.get(f"{BASE}/health")
    print("HEALTH:", r.json())

def test_single_legit():
    payload = {
        "V1": 1.19, "V2": 0.26, "V3": 0.16, "V4": 0.45,
        "V5": -0.02, "V6": -0.16, "V7": -0.27, "V8": -0.13,
        "V9": 0.05, "V10": 0.09, "V11": -0.05, "V12": 0.19,
        "V13": -0.05,"V14": -0.11,"V15": 0.12, "V16": -0.08,
        "V17": 0.06, "V18": -0.04,"V19": 0.04, "V20": 0.02,
        "V21": -0.02,"V22": -0.02,"V23": -0.01,"V24": -0.01,
        "V25": 0.03, "V26": 0.02, "V27": 0.00, "V28": 0.01,
        "Amount": 29.99, "Time": 50000
    }
    r = httpx.post(f"{BASE}/score", json=payload)
    print("\nLEGIT TX:", json.dumps(r.json(), indent=2))

def test_single_fraud():
    # High-risk feature values
    payload = {
        "V1": -3.04, "V2": -3.16, "V3": 1.09, "V4": 2.29,
        "V5": -1.35, "V6": -1.68, "V7": -2.94, "V8": 0.47,
        "V9": -0.71, "V10": -2.32,"V11": 1.94, "V12": -3.43,
        "V13": -0.22,"V14": -3.70,"V15": 0.40, "V16": -0.85,
        "V17": -4.00,"V18": -1.40,"V19": -0.30,"V20": 0.47,
        "V21": 0.41, "V22": -0.09,"V23": -0.04,"V24": -0.30,
        "V25": 0.60, "V26": 0.01, "V27": 0.40, "V28": 0.14,
        "Amount": 0.76, "Time": 8000
    }
    r = httpx.post(f"{BASE}/score", json=payload)
    print("\nFRAUD TX:", json.dumps(r.json(), indent=2))

def test_rules_config():
    r = httpx.get(f"{BASE}/rules/config")
    print("\nRULES CONFIG:", json.dumps(r.json(), indent=2))

if __name__ == "__main__":
    test_health()
    test_single_legit()
    test_single_fraud()
    test_rules_config()
    print("\nAll tests complete.")