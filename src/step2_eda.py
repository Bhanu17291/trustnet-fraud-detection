# ============================================================
# TRUSTNET — Step 2: Exploratory Data Analysis
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR    = os.getcwd()
DATA_PATH   = os.path.join(BASE_DIR, 'data', 'creditcard.csv')
OUTPUT_DIR  = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 55)
print("  TRUSTNET — Fraud Detection Engine")
print("  Step 2: Exploratory Data Analysis")
print("=" * 55)

# ── Load ──────────────────────────────────────────────────────
df    = pd.read_csv(DATA_PATH)
fraud = df[df['Class'] == 1]
legit = df[df['Class'] == 0]

print(f"\n✅ Data loaded — {len(df):,} transactions")
print(f"   Fraud: {len(fraud):,}  |  Legit: {len(legit):,}")

# ── Chart 1: Overview (3 panels) ─────────────────────────────
print("\n📊 Generating Chart 1: Overview...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('TrustNet — EDA Overview', fontsize=15, fontweight='bold', y=1.02)

# Panel 1: Class imbalance
axes[0].bar(['Legitimate', 'Fraud'],
            [len(legit), len(fraud)],
            color=['#2ecc71', '#e74c3c'],
            edgecolor='white', linewidth=0.5)
axes[0].set_title('Class Distribution', fontweight='bold')
axes[0].set_ylabel('Count')
for i, v in enumerate([len(legit), len(fraud)]):
    axes[0].text(i, v + 500, f'{v:,}', ha='center', fontweight='bold', fontsize=10)

# Panel 2: Amount distribution
axes[1].hist(legit['Amount'], bins=60, alpha=0.6,
             color='#2ecc71', label='Legitimate', density=True)
axes[1].hist(fraud['Amount'], bins=40, alpha=0.7,
             color='#e74c3c', label='Fraud', density=True)
axes[1].set_title('Transaction Amount Distribution', fontweight='bold')
axes[1].set_xlabel('Amount ($)')
axes[1].set_ylabel('Density')
axes[1].set_xlim(0, 1000)
axes[1].legend()

# Panel 3: Timing scatter
axes[2].scatter(legit['Time'] / 3600, legit['Amount'],
                alpha=0.01, color='#2ecc71', s=1, label='Legitimate')
axes[2].scatter(fraud['Time'] / 3600, fraud['Amount'],
                alpha=0.5, color='#e74c3c', s=8, label='Fraud')
axes[2].set_title('Fraud Timing Pattern', fontweight='bold')
axes[2].set_xlabel('Hours since first transaction')
axes[2].set_ylabel('Amount ($)')
axes[2].legend()

plt.tight_layout()
out1 = os.path.join(OUTPUT_DIR, 'eda_overview.png')
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.show()
print(f"   ✅ Saved: outputs/eda_overview.png")

# ── Chart 2: Feature correlations ────────────────────────────
print("\n📊 Generating Chart 2: Feature Correlations...")
plt.figure(figsize=(16, 5))
correlations = df.corr()['Class'].drop('Class').sort_values()
colors = ['#e74c3c' if x > 0 else '#3498db' for x in correlations]
correlations.plot(kind='bar', color=colors, edgecolor='white', linewidth=0.3)
plt.title('TrustNet — Feature Correlation with Fraud', fontsize=13, fontweight='bold')
plt.xlabel('Feature')
plt.ylabel('Correlation with Fraud (Class=1)')
plt.axhline(y=0, color='black', linewidth=0.8)
plt.tight_layout()
out2 = os.path.join(OUTPUT_DIR, 'feature_correlations.png')
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.show()
print(f"   ✅ Saved: outputs/feature_correlations.png")

# ── Chart 3: Fraud amount boxplot ────────────────────────────
print("\n📊 Generating Chart 3: Amount Boxplot...")
fig, ax = plt.subplots(figsize=(8, 5))
df.boxplot(column='Amount', by='Class', ax=ax,
           flierprops=dict(marker='.', markersize=2, alpha=0.3))
ax.set_title('Transaction Amount by Class', fontweight='bold')
ax.set_xlabel('Class  (0 = Legitimate, 1 = Fraud)')
ax.set_ylabel('Amount ($)')
plt.suptitle('')
plt.tight_layout()
out3 = os.path.join(OUTPUT_DIR, 'amount_boxplot.png')
plt.savefig(out3, dpi=150, bbox_inches='tight')
plt.show()
print(f"   ✅ Saved: outputs/amount_boxplot.png")

# ── Key patterns summary ──────────────────────────────────────
print("\n── KEY FRAUD PATTERNS DISCOVERED ──")
print(f"   Avg fraud amount      : ${fraud['Amount'].mean():.2f}")
print(f"   Avg legit amount      : ${legit['Amount'].mean():.2f}")
print(f"   Max fraud amount      : ${fraud['Amount'].max():.2f}")
print(f"\n   Top 5 features correlated WITH fraud:")
print(correlations.tail(5).to_string())
print(f"\n   Top 5 features correlated AGAINST fraud:")
print(correlations.head(5).to_string())

print("\n" + "=" * 55)
print("  Step 2 Complete ✅  —  Run step3_preprocessing.py next")
print("=" * 55)