# ============================================================
# TRUSTNET — Step 5: Model Explainability (SHAP)
# ============================================================

import pickle
import os
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR   = os.getcwd()
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 55)
print("  TRUSTNET — Fraud Detection Engine")
print("  Step 5: SHAP Explainability")
print("=" * 55)

# ── Load model & data ─────────────────────────────────────────
with open(os.path.join(MODELS_DIR, 'best_model.pkl'), 'rb') as f:
    saved = pickle.load(f)
with open(os.path.join(MODELS_DIR, 'processed_data.pkl'), 'rb') as f:
    data = pickle.load(f)

model        = saved['model']
model_name   = saved['name']
X_test       = data['X_test']
y_test       = data['y_test']
feature_names = data['feature_names']

X_test_df = pd.DataFrame(X_test, columns=feature_names)

print(f"\n✅ Model loaded: {model_name}")
print(f"   Test set: {X_test_df.shape[0]:,} transactions")

# ── SHAP explainer ────────────────────────────────────────────
print(f"\n🔧 Computing SHAP values...")
print("   (This takes 1–3 minutes — please wait...)")

explainer   = shap.TreeExplainer(model)
sample_size = min(1000, len(X_test_df))
X_sample    = X_test_df.iloc[:sample_size]
shap_values = explainer.shap_values(X_sample)

# For binary classifiers shap_values is a list → take class 1 (fraud)
sv = shap_values[1] if isinstance(shap_values, list) else shap_values
print(f"   ✅ SHAP values computed for {sample_size} transactions")

# ── Chart 1: Feature importance bar ──────────────────────────
print("\n📊 Generating Chart 1: Feature Importance...")
plt.figure(figsize=(10, 7))
shap.summary_plot(sv, X_sample, plot_type='bar',
                  feature_names=feature_names, show=False)
plt.title(f'TrustNet — Top Features Driving Fraud\n({model_name})',
          fontweight='bold')
plt.tight_layout()
out1 = os.path.join(OUTPUT_DIR, 'shap_importance.png')
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.show()
print(f"   ✅ Saved: outputs/shap_importance.png")

# ── Chart 2: SHAP beeswarm (direction of impact) ─────────────
print("\n📊 Generating Chart 2: SHAP Beeswarm...")
plt.figure(figsize=(10, 8))
shap.summary_plot(sv, X_sample, feature_names=feature_names, show=False)
plt.title(f'TrustNet — How Features Affect Fraud Score\n({model_name})',
          fontweight='bold')
plt.tight_layout()
out2 = os.path.join(OUTPUT_DIR, 'shap_beeswarm.png')
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.show()
print(f"   ✅ Saved: outputs/shap_beeswarm.png")

# ── Chart 3: Single transaction explanation ───────────────────
print("\n📊 Generating Chart 3: Single Transaction Explanation...")
y_pred       = model.predict(X_test_df)
fraud_idx    = np.where((y_test.values == 1) & (y_pred == 1))[0]

if len(fraud_idx) > 0:
    idx = fraud_idx[0]
    base_val = (explainer.expected_value[1]
                if isinstance(explainer.expected_value, list)
                else explainer.expected_value)

    plt.figure(figsize=(12, 4))
    shap.waterfall_plot(
        shap.Explanation(
            values=sv[idx],
            base_values=base_val,
            data=X_sample.iloc[idx].values,
            feature_names=feature_names
        ),
        show=False
    )
    plt.title(f'TrustNet — Why Transaction #{idx} Was Flagged as FRAUD',
              fontweight='bold')
    plt.tight_layout()
    out3 = os.path.join(OUTPUT_DIR, 'shap_single_transaction.png')
    plt.savefig(out3, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"   ✅ Saved: outputs/shap_single_transaction.png")
else:
    print("   ⚠️  No correctly-predicted fraud found in sample — skipping waterfall chart")

# ── Summary ───────────────────────────────────────────────────
print("\n── What SHAP tells you ──")
mean_abs = np.abs(sv).mean(axis=0)
top_features = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)
print("\n   Top 10 most important features:")
print(top_features.head(10).to_string())

print("\n" + "=" * 55)
print("  Step 5 Complete ✅  —  Run step6_app.py to launch dashboard")
print("  Command: streamlit run src/step6_app.py")
print("=" * 55)