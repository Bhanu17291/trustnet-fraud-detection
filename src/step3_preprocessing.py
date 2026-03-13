# ============================================================
# TRUSTNET — Step 3: Feature Engineering & Preprocessing
# ============================================================

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR    = os.getcwd()
DATA_PATH   = os.path.join(BASE_DIR, 'data',   'creditcard.csv')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 55)
print("  TRUSTNET — Fraud Detection Engine")
print("  Step 3: Feature Engineering & Preprocessing")
print("=" * 55)

# ── Load ──────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"\n✅ Data loaded — {len(df):,} transactions")

# ── Feature Engineering ───────────────────────────────────────
print("\n🔧 Engineering features...")

# 1. Hour of day — fraud spikes at odd hours
df['hour'] = (df['Time'] % (3600 * 24)) // 3600

# 2. Log-transform amount — reduces skew
df['log_amount'] = np.log1p(df['Amount'])

# 3. Amount z-score — flags unusually large/small amounts
df['amount_zscore'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()

# 4. Small amount flag — fraudsters test cards with tiny charges
df['is_small_amount'] = (df['Amount'] < 10).astype(int)

# 5. Round amount flag — common in scripted fraud attacks
df['is_round_amount'] = (df['Amount'] % 10 == 0).astype(int)

print(f"   ✅ 5 new features created")
print(f"   Total features: {df.shape[1]}")

# ── Scale Amount & Time ───────────────────────────────────────
print("\n🔧 Scaling Amount and Time...")
scaler_amount = StandardScaler()
scaler_time   = StandardScaler()
df['Amount_scaled'] = scaler_amount.fit_transform(df[['Amount']])
df['Time_scaled']   = scaler_time.fit_transform(df[['Time']])

# ── Define X and y ────────────────────────────────────────────
drop_cols = ['Class', 'Amount', 'Time']
X = df.drop(drop_cols, axis=1)
y = df['Class']

print(f"\n── Feature matrix: {X.shape}")
print(f"── Target: {y.value_counts().to_dict()}")

# ── Train / Test Split ────────────────────────────────────────
print("\n🔧 Splitting data (80% train / 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"   Train set : {X_train.shape[0]:,} rows  ({y_train.sum()} fraud)")
print(f"   Test set  : {X_test.shape[0]:,} rows  ({y_test.sum()} fraud)")

# ── SMOTE — only on training data (prevent data leakage) ──────
print("\n🔧 Applying SMOTE to balance training data...")
print("   (This may take ~30 seconds...)")
smote = SMOTE(random_state=42, sampling_strategy=0.1)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"\n   Before SMOTE — Fraud: {y_train.sum():,}  |  Legit: {(y_train==0).sum():,}")
print(f"   After  SMOTE — Fraud: {(y_train_sm==1).sum():,}  |  Legit: {(y_train_sm==0).sum():,}")
print(f"   New fraud rate: {y_train_sm.mean()*100:.1f}%")

# ── Save processed data ───────────────────────────────────────
save_path = os.path.join(MODELS_DIR, 'processed_data.pkl')
with open(save_path, 'wb') as f:
    pickle.dump({
        'X_train':       X_train_sm,
        'X_test':        X_test,
        'y_train':       y_train_sm,
        'y_test':        y_test,
        'feature_names': X.columns.tolist(),
        'scaler_amount': scaler_amount,
        'scaler_time':   scaler_time,
    }, f)

print(f"\n   ✅ Saved: models/processed_data.pkl")

print("\n" + "=" * 55)
print("  Step 3 Complete ✅  —  Run step4_train_models.py next")
print("=" * 55)