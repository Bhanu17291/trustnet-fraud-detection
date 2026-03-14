<div align="center">

# 🔐 TrustNet
**AI-Powered Fraud Detection Engine**

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?style=for-the-badge)](https://trustnet-fraud-detection-ah5wz3dvxjjjmmc5xxkmhm.streamlit.app)
[![Live API](https://img.shields.io/badge/Live_API-Render-00D4AA?style=for-the-badge)](https://trustnet-api-lbfp.onrender.com/docs)

**ROC-AUC 0.98 · 92% Fraud Recall · <100ms Decision · Weekly Auto-Retraining**

</div>

---

## What It Does

Real-time credit card fraud detection with 4 layers of defence:
```
Transaction → XGBoost (0.98 AUC) → Rule Engine → Isolation Forest → Alert Queue
```

Every decision is **explainable via SHAP**. The model **retrains automatically** every week using analyst feedback.

---

## Live Links

| | |
|--|--|
| 🌐 Dashboard | https://trustnet-fraud-detection-ah5wz3dvxjjjmmc5xxkmhm.streamlit.app |
| ⚡ API | https://trustnet-api-lbfp.onrender.com/docs |
| 💻 Code | https://github.com/Bhanu17291/trustnet-fraud-detection |

---

## Performance

| Metric | TrustNet | Industry Avg |
|--------|----------|-------------|
| ROC-AUC | **0.98** | 0.85–0.92 |
| Fraud Recall | **92%** | 75–80% |
| False Alarms | **~5%** | 10–15% |

---

## Features

- **Batch & real-time scoring** via dashboard and REST API
- **SHAP explainability** — why was this flagged?
- **Fraud network graph** — visualise connected fraud rings
- **Isolation Forest** — catches new fraud patterns the model hasn't seen
- **Auto-retraining** — weekly pipeline using confirmed analyst decisions

---

## Quick Start
```bash
git clone https://github.com/Bhanu17291/trustnet-fraud-detection.git
cd trustnet-fraud-detection
pip install -r requirements.txt
python src/step4_train_models.py
streamlit run src/step6_app.py
```

---

## Stack

`Python` `XGBoost` `SHAP` `Isolation Forest` `FastAPI` `Streamlit` `SQLite` `NetworkX` `PyVis` `SMOTE`
