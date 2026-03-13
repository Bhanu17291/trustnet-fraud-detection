# ============================================================
# TRUSTNET — Step 1: Data Loading & Setup
# ============================================================

import pandas as pd
import numpy as np
import os

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR  = os.getcwd()
DATA_PATH = os.path.join(BASE_DIR, 'data', 'creditcard.csv')

print("=" * 55)
print("  TRUSTNET — Fraud Detection Engine")
print("  Step 1: Loading Dataset")
print("=" * 55)

# ── Check file exists ─────────────────────────────────────────
if not os.path.exists(DATA_PATH):
    print(f"\n❌ ERROR: creditcard.csv not found at:\n   {DATA_PATH}")
    print("\n👉 Download it from:")
    print("   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print("   Then place it inside the 'data/' folder.")
    exit()

# ── Load ──────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"\n✅ Dataset loaded successfully!")

# ── Overview ──────────────────────────────────────────────────
print(f"\n── Shape ──")
print(f"   Rows    : {df.shape[0]:,}")
print(f"   Columns : {df.shape[1]}")

print(f"\n── Columns ──")
print(f"   {df.columns.tolist()}")

print(f"\n── Missing Values ──")
missing = df.isnull().sum().sum()
print(f"   Total missing: {missing} {'✅ None!' if missing == 0 else '⚠️ Found!'}")

print(f"\n── Class Distribution ──")
counts = df['Class'].value_counts()
print(f"   Legitimate (0) : {counts[0]:,}")
print(f"   Fraud      (1) : {counts[1]:,}")
print(f"   Fraud rate     : {df['Class'].mean()*100:.4f}%")

print(f"\n── Amount Statistics ──")
print(df.groupby('Class')['Amount'].describe().round(2).to_string())

print(f"\n── Time Range ──")
print(f"   Min : {df['Time'].min():.0f}s")
print(f"   Max : {df['Time'].max():.0f}s  (~{df['Time'].max()/3600:.1f} hours of data)")

print("\n" + "=" * 55)
print("  Step 1 Complete ✅  —  Run step2_eda.py next")
print("=" * 55)