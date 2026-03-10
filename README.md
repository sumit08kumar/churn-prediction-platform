# 🔮 ChurnIQ — Customer Churn Prediction Platform

> End-to-end ML platform: from raw data to a live React dashboard.  
> XGBoost · MLflow · FastAPI · Docker · React · SHAP Explainability

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)
![React](https://img.shields.io/badge/React-18-61dafb?logo=react)
![MLflow](https://img.shields.io/badge/MLflow-2.13-orange?logo=mlflow)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 Problem Statement

A telecom company loses significant revenue when customers cancel subscriptions (churn). This platform predicts **which customers will churn**, so retention teams can intervene proactively.

**Dataset:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM) — 7,043 customers, 21 features.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         ChurnIQ Platform                        │
├──────────────┬─────────────────────────────┬────────────────────┤
│  ML Pipeline │      FastAPI Backend        │  React Frontend    │
│              │                             │                    │
│  train.py    │  POST /predict              │  Dashboard         │
│  ├─ Load CSV │  POST /predict/batch/csv    │  Predictor Form    │
│  ├─ EDA      │  POST /predict/batch/json   │  SHAP Explanations │
│  ├─ FeatEng  │  POST /explain              │  Batch Upload      │
│  ├─ SMOTE    │  GET  /model/info           │  Model Registry UI │
│  ├─ Optuna   │  GET  /health               │                    │
│  ├─ XGBoost  │                             │                    │
│  └─ MLflow   │  Model: sklearn Pipeline    │  Charts: Recharts  │
│              │  Serving: uvicorn           │  State: React      │
└──────────────┴─────────────────────────────┴────────────────────┘
         ↕                    ↕                        ↕
    MLflow Server         Docker                  Render/Railway
    (Experiments +        Compose                 (Live Deploy)
     Model Registry)
```

---

## 🚀 Quick Start

### Option A — Docker Compose (Recommended)

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/churn-prediction-platform.git
cd churn-prediction-platform

# 2. Download the dataset
# From Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Place it at: backend/data/WA_Fn-UseC_-Telco-Customer-Churn.csv

# 3. Train the model first
cd backend
pip install -r requirements.txt
python -m ml.train --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv

# 4. Start everything
cd ..
docker-compose up --build
```

**Services:**
| Service | URL |
|---|---|
| React Frontend | http://localhost:3000 |
| FastAPI Backend | http://localhost:8000 |
| FastAPI Swagger Docs | http://localhost:8000/docs |
| MLflow Dashboard | http://localhost:5000 |

---

### Option B — Local Development

```bash
# Backend
cd backend
pip install -r requirements.txt
python -m ml.train --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv
uvicorn app.main:app --reload --port 8000

# Frontend (new terminal)
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| AUC-ROC | **0.891** |
| F1 Score | **0.637** |
| Precision | **0.681** |
| Recall | **0.598** |
| Accuracy | **0.812** |

### Key findings:
- **Month-to-month contracts** have 42.7% churn rate vs 2.8% for 2-year contracts
- **Fiber optic customers** churn at 2x the rate of DSL customers
- **Senior citizens** with no tech support are the highest-risk segment
- **Tenure < 12 months** is the single strongest predictor of churn

---

## 🧠 ML Pipeline Details

```
Raw CSV → Feature Engineering → SMOTE → Preprocessing → XGBoost → MLflow → FastAPI
```

### 1. Feature Engineering
- `charge_per_tenure` — Monthly charges normalized by tenure
- `no_support_mtm` — Month-to-month + no tech support (high risk combo)
- `senior_no_security` — Senior citizen with no online security
- `total_services` — Total number of subscribed services
- `charge_tier` — MonthlyCharges bucketed into 4 tiers

### 2. Preprocessing Pipeline (sklearn)
- **Numeric:** Median imputation → StandardScaler
- **Categorical:** Mode imputation → OneHotEncoder
- Built as a `ColumnTransformer` inside a full `Pipeline` object

### 3. Class Imbalance
- ~26% churn rate → imbalanced dataset
- **SMOTE** applied on training data only (no data leakage)
- Used `imbalanced-learn` Pipeline to prevent leakage

### 4. Model Selection
| Model | AUC-ROC |
|---|---|
| Logistic Regression (baseline) | 0.842 |
| Random Forest | 0.858 |
| Gradient Boosting | 0.865 |
| **XGBoost (tuned)** | **0.891** |

### 5. Hyperparameter Tuning (Optuna)
- 50 trials with TPE sampler
- Objective: maximize CV AUC-ROC (5-fold StratifiedKFold)
- All trials logged to MLflow

### 6. Explainability (SHAP)
- `shap.TreeExplainer` on the XGBoost model
- Top 5 factors returned for every prediction
- Full SHAP waterfall available via `/explain` endpoint

---

## 🔌 API Reference

### `POST /predict`
Single customer churn prediction.

```json
// Request
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "tenure": 12,
  "Contract": "Month-to-month",
  "MonthlyCharges": 70.35,
  ...
}

// Response
{
  "churn_probability": 0.7234,
  "churn_prediction": true,
  "risk_level": "High",
  "risk_score": 72,
  "confidence": "High",
  "top_factors": [
    {"feature": "Month To Month Contract", "impact": 35.1, "direction": "increases"},
    ...
  ],
  "recommendation": "Intervene now. Offer a contract upgrade or personalized discount."
}
```

### `POST /predict/batch/csv`
Upload a CSV, get predictions back as a downloadable CSV.

### `POST /explain`
Full SHAP explanation — top 15 features with values.

### `GET /model/info`
Model version, metrics, and registry status.

---

## 🌐 Deployment (Render — Free Tier)

```bash
# 1. Push to GitHub

# 2. Create a Render Web Service for the backend
#    - Build Command: pip install -r requirements.txt
#    - Start Command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
#    - Set env: MODEL_PATH=artifacts/churn_pipeline.pkl

# 3. Create a Render Static Site for the frontend
#    - Build Command: npm install && npm run build
#    - Publish Directory: dist
#    - Set env: VITE_API_URL=https://your-backend.onrender.com

# 4. Add Render deploy hooks to GitHub secrets:
#    RENDER_BACKEND_DEPLOY_HOOK
#    RENDER_FRONTEND_DEPLOY_HOOK
```

---

## 🧪 Running Tests

```bash
cd backend
pip install pytest httpx
pytest tests/ -v
```

---

## 📁 Project Structure

```
churn-prediction-platform/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI app + all routes
│   │   ├── model.py         # Model loading, prediction, SHAP
│   │   └── schemas.py       # Pydantic request/response schemas
│   ├── ml/
│   │   ├── preprocess.py    # Feature engineering + sklearn pipeline
│   │   └── train.py         # Full training pipeline with MLflow
│   ├── tests/
│   │   └── test_api.py      # pytest API tests
│   ├── artifacts/           # Saved model .pkl + metadata (gitignored)
│   ├── data/                # Dataset CSV (gitignored)
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx      # Analytics + KPI charts
│   │   │   ├── Predictor.jsx      # Prediction form + SHAP panel
│   │   │   ├── BatchUpload.jsx    # CSV drag & drop
│   │   │   └── ModelInfo.jsx      # Model registry + metrics
│   │   ├── services/
│   │   │   └── api.js             # Axios API client
│   │   ├── App.jsx                # Router + sidebar layout
│   │   └── index.css              # Design system + CSS variables
│   ├── Dockerfile
│   └── package.json
├── .github/
│   └── workflows/ci-cd.yml        # GitHub Actions CI/CD
├── docker-compose.yml
└── README.md
```

---

## 💼 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11, JavaScript (ES2023) |
| ML Framework | scikit-learn, XGBoost, imbalanced-learn |
| Tuning | Optuna |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| API | FastAPI + Pydantic + uvicorn |
| Frontend | React 18 + Recharts + React Router |
| Styling | CSS Variables (custom design system) |
| Containers | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Deployment | Render (free tier) |

---

## 📝 License

MIT — feel free to fork and build on this.

---

*Built to demonstrate end-to-end ML engineering: from raw data to a live, deployed product.*
