# ============================================================
# TRUSTNET — Step 4: Train & Compare Models
# ============================================================

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)
from xgboost import XGBClassifier

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR   = os.getcwd()
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 55)
print("  TRUSTNET — Fraud Detection Engine")
print("  Step 4: Model Training & Comparison")
print("=" * 55)

# ── Load processed data ───────────────────────────────────────
data_path = os.path.join(MODELS_DIR, 'processed_data.pkl')
with open(data_path, 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
X_test  = data['X_test']
y_train = data['y_train']
y_test  = data['y_test']

print(f"\n✅ Data loaded")
print(f"   Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

# ── Define models ─────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=100,
        scale_pos_weight=10,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
}

# ── Train & evaluate all models ───────────────────────────────
results = {}
print("\n── Training Models ──")

for name, model in models.items():
    print(f"\n🔧 Training {name}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc  = average_precision_score(y_test, y_prob)

    results[name] = {
        'model':   model,
        'y_pred':  y_pred,
        'y_prob':  y_prob,
        'roc_auc': roc_auc,
        'pr_auc':  pr_auc,
    }

    print(f"   ROC-AUC : {roc_auc:.4f}")
    print(f"   PR-AUC  : {pr_auc:.4f}  ← key metric for imbalanced data")
    print(classification_report(y_test, y_pred,
                                 target_names=['Legit', 'Fraud'],
                                 digits=3))

# ── Best model ────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['pr_auc'])
best      = results[best_name]
print(f"\n🏆 Best model: {best_name}")
print(f"   PR-AUC  : {best['pr_auc']:.4f}")
print(f"   ROC-AUC : {best['roc_auc']:.4f}")

# ── Chart 1: ROC + PR curves ──────────────────────────────────
print("\n📊 Generating comparison charts...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('TrustNet — Model Comparison', fontsize=14, fontweight='bold')
colors = ['#3498db', '#2ecc71', '#e74c3c']

for (name, res), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    axes[0].plot(fpr, tpr, color=color,
                 label=f"{name}  (AUC={res['roc_auc']:.3f})", linewidth=2)

    prec, rec, _ = precision_recall_curve(y_test, res['y_prob'])
    axes[1].plot(rec, prec, color=color,
                 label=f"{name}  (AUC={res['pr_auc']:.3f})", linewidth=2)

axes[0].plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Random')
axes[0].set_title('ROC Curve', fontweight='bold')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend(loc='lower right', fontsize=9)
axes[0].grid(alpha=0.3)

axes[1].axhline(y=y_test.mean(), color='k', linestyle='--',
                linewidth=0.8, label='Random baseline')
axes[1].set_title('Precision-Recall Curve', fontweight='bold')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].legend(loc='upper right', fontsize=9)
axes[1].grid(alpha=0.3)

plt.tight_layout()
out1 = os.path.join(OUTPUT_DIR, 'model_comparison.png')
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.show()
print(f"   ✅ Saved: outputs/model_comparison.png")

# ── Chart 2: Confusion matrix ─────────────────────────────────
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, best['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Legit', 'Predicted Fraud'],
            yticklabels=['Actual Legit', 'Actual Fraud'],
            linewidths=0.5)
plt.title(f'Confusion Matrix — {best_name}', fontweight='bold')
plt.tight_layout()
out2 = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.show()
print(f"   ✅ Saved: outputs/confusion_matrix.png")

# ── Save best model ───────────────────────────────────────────
save_path = os.path.join(MODELS_DIR, 'best_model.pkl')
with open(save_path, 'wb') as f:
    pickle.dump({'name': best_name, 'model': best['model']}, f)
print(f"\n   ✅ Saved: models/best_model.pkl  ({best_name})")

print("\n" + "=" * 55)
print("  Step 4 Complete ✅  —  Run step5_explainability.py next")
print("=" * 55)