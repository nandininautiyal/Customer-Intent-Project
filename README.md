# 🛒 Customer Purchase Intent Engine

> A production-grade machine learning system that predicts whether an online shopper will make a purchase — and recommends the optimal marketing intervention to convert them.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)

---

## 📌 Problem Statement

E-commerce platforms lose revenue every day because they treat all visitors the same. A visitor who has spent 15 minutes on product pages with high page values is fundamentally different from someone who bounced after 30 seconds — yet most platforms serve them identical experiences.

This system solves that by:
1. **Predicting** purchase intent in real time (will this session convert?)
2. **Segmenting** visitors into Cold / Warm / Hot / Convert buckets
3. **Recommending** the optimal intervention for each segment (discount, urgency nudge, social proof, checkout prompt)

---

## 🏗️ System Architecture

```
Raw Session Data (UCI)
        │
        ▼
┌───────────────────┐
│  Feature Engineer  │  ← 7 domain-specific features added
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   Preprocessing    │  ← StandardScaler + SMOTE (train only)
└───────────────────┘
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Logistic    │  │   XGBoost    │  │  Neural Net  │
│  Regression  │  │ (Optuna-tuned│  │  (PyTorch)   │
│  (Baseline)  │  │  50 trials)  │  │  BatchNorm   │
└──────────────┘  └──────────────┘  └──────────────┘
        │                │                  │
        └────────────────┴──────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Stacking Ensemble  │  ← OOF meta-features (5-fold)
              │  (Meta-Learner:     │  ← Threshold tuning on val set
              │  Logistic Reg.)     │
              └─────────────────────┘
                         │
              ┌──────────┴───────────┐
              ▼                      ▼
    ┌──────────────────┐   ┌──────────────────────┐
    │  SHAP Explainer  │   │  Intent Recommender  │
    │  (Why did the    │   │  Cold/Warm/Hot/Convert│
    │  model predict?) │   │  + Intervention logic │
    └──────────────────┘   └──────────────────────┘
                                     │
                                     ▼
                           ┌──────────────────┐
                           │   FastAPI Server  │
                           │  POST /predict    │
                           │  GET  /health     │
                           └──────────────────┘
```

---

## 📊 Model Performance (Out-of-Sample Test Set)

| Rank | Model | Accuracy | F1 Score | Precision | Recall | ROC-AUC | PR-AUC | Lift@Top20% |
|------|-------|----------|----------|-----------|--------|---------|--------|-------------|
| 1 | **XGBoost** | 0.8913 | 0.6716 | 0.6313 | 0.7173 | **0.9282** | 0.7244 | 0.7775 |
| 2 | **Ensemble** | **0.8946** | 0.6597 | 0.6597 | 0.6597 | 0.9204 | **0.7249** | **0.7723** |
| 3 | NeuralNet | 0.8869 | 0.6391 | 0.6317 | 0.6466 | 0.9019 | 0.6881 | 0.7435 |
| 4 | Logistic | 0.8755 | 0.6279 | 0.5847 | 0.6780 | 0.8966 | 0.6272 | 0.7304 |

### Key Findings

- **XGBoost achieves ROC-AUC of 0.928** — at the top end of published results on this dataset (academic papers report 0.88–0.93)
- **Ensemble has perfectly balanced precision = recall = 0.6597** — a sign of well-calibrated predictions from OOF stacking
- **Lift@Top20% of 0.777** — targeting the top 20% of predicted buyers captures **77.7% of all actual purchases**, directly translating to marketing ROI
- **SMOTE + threshold tuning** resolved the class imbalance problem (15.5% buyers vs 84.5% non-buyers)

---

## 🔬 What Makes This Different from a Notebook

| Standard ML Project | This Project |
|---|---|
| Single model | 4 models compared + stacking ensemble |
| Fixed 0.5 threshold | F1-optimal threshold tuning per model |
| No class imbalance handling | SMOTE on training set only (no leakage) |
| Accuracy as metric | ROC-AUC, PR-AUC, F1, Lift@20% |
| No explainability | SHAP beeswarm + bar plots |
| Jupyter notebook | Modular pipeline, Docker, FastAPI |
| Predict only | Predict + segment + recommend |

---

## 📁 Repository Structure

```
customer-intent-engine/
│
├── docker-compose.yml          ← Spins up app + postgres
├── Dockerfile
├── requirements.txt
├── .env.example
├── orchestrator.py             ← Master run script (one command runs everything)
│
├── pipeline/
│   ├── ingest.py               ← Loads + validates UCI dataset
│   ├── preprocess.py           ← Encoding, scaling, SMOTE
│   └── features.py             ← 7 engineered features
│
├── models/
│   ├── base_model.py           ← Abstract interface for all models
│   ├── logistic_model.py       ← L2 logistic baseline
│   ├── xgboost_model.py        ← Optuna-tuned XGBoost
│   ├── neural_model.py         ← PyTorch net (BatchNorm + Dropout)
│   ├── ensemble_model.py       ← OOF stacking meta-learner
│   └── recommender.py          ← Rule-based intervention engine
│
├── evaluation/
│   ├── metrics.py              ← All scoring logic + business lift metric
│   ├── shap_explainer.py       ← SHAP summary + beeswarm plots
│   └── reporter.py             ← ROC, PR, confusion matrix, bar chart
│
├── api/
│   ├── main.py                 ← FastAPI app with /predict endpoint
│   └── schemas.py              ← Pydantic input/output schemas
│
├── data/
│   └── raw/                    ← Place UCI CSV here (see Dataset section)
│
└── reports/                    ← Auto-generated outputs (gitignored)
    ├── artifacts/              ← Saved model weights + scalers
    ├── roc_curves.png
    ├── pr_curves.png
    ├── confusion_matrices.png
    ├── metrics_comparison.png
    ├── shap_summary_bar.png
    ├── shap_summary_beeswarm.png
    ├── model_comparison_metrics.csv
    └── model_comparison_metrics.json
```

---



### Option 1: Local (Recommended for development)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/customer-intent-engine.git
cd customer-intent-engine

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the dataset
# Download from: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
# Save as: data/raw/online_shoppers_intention.csv

# 5. Run the full pipeline
python orchestrator.py

# 6. Launch the inference API
uvicorn api.main:app --reload
# → Open http://localhost:8000/docs
```

### Option 2: Docker

```bash
# Copy and fill in environment variables
cp .env.example .env

# Build and run all services
docker-compose up -d --build

# Run the pipeline
docker exec -it intent-app python orchestrator.py

# API is live at http://localhost:8001/docs
```

---

## 📦 Dataset

**UCI Online Shoppers Purchasing Intention Dataset**

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository |
| Link | https://archive.ics.uci.edu/dataset/468 |
| Sessions | 12,330 |
| Features | 18 raw + 7 engineered = 24 total |
| Target | `Revenue` (True = purchase made) |
| Class Balance | 15.5% positive (buyers), 84.5% negative |

### Engineered Features

| Feature | Formula | Business Meaning |
|---|---|---|
| `TotalPages` | Admin + Info + ProductRelated | Overall browsing depth |
| `TotalDuration` | Sum of all durations | Total time investment |
| `ProductPageRatio` | ProductRelated / TotalPages | Focus on product pages |
| `AvgTimePerPage` | TotalDuration / TotalPages | Engagement depth per page |
| `HighPageValue` | PageValues ≥ 75th percentile | Strong purchase signal flag |
| `NearSpecialDay` | SpecialDay > 0 | Seasonal urgency context |
| `ExitBounceRisk` | BounceRates + ExitRates | Combined abandonment risk |

---

## 🤖 Model Details

### Logistic Regression (Baseline)
- L2 regularization, `class_weight=balanced`
- F1-optimal threshold tuning on validation set
- Purpose: interpretable lower bound

### XGBoost (Best Single Model)
- **Optuna hyperparameter search**: 40 trials optimizing F1
- Tuned params: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda, gamma
- F1-optimal threshold tuning post-training
- ROC-AUC: **0.928**

### Neural Network (PyTorch)
- Architecture: 256 → 128 → 64 → 1
- BatchNorm + Dropout at every layer
- BCEWithLogitsLoss with positive class weighting
- CosineAnnealingLR scheduler
- Best checkpoint saved and restored
- F1-optimal threshold tuning post-training

### Stacking Ensemble
- **Out-of-fold (OOF) meta-feature generation** across 5 stratified folds
- Meta-learner: Logistic Regression trained on OOF probability outputs
- Threshold tuned on clean validation set (zero data leakage)
- Result: perfectly balanced precision = recall = 0.6597

---

## 🎯 Recommendation Engine

The system doesn't just predict — it tells you what to do:

```json
{
  "will_purchase": false,
  "purchase_probability": 0.43,
  "confidence": "Medium",
  "recommendation": {
    "segment": "Warm",
    "recommended_action": "show_exit_intent_offer",
    "reason": "High exit risk detected — trigger exit-intent popup with discount.",
    "urgency": "medium",
    "message": "Offer a time-limited discount or free shipping nudge."
  }
}
```

### Segments and Actions

| Segment | Probability Range | Default Action | Override Triggers |
|---|---|---|---|
| Cold | 0.00 – 0.30 | Show social proof | New visitor → trust badges |
| Warm | 0.30 – 0.55 | Show discount | High exit risk → exit-intent popup |
| Hot | 0.55 – 0.75 | Show urgency | Near special day → seasonal offer |
| Convert | 0.75 – 1.00 | Checkout prompt | Returning + high page value → loyalty reward |

---

## 🔍 SHAP Explainability

Top drivers of purchase intent (from SHAP analysis):

1. **PageValues** — strongest signal by far. High page value sessions are serious buyers
2. **Month** — seasonality matters. November/December sessions behave differently
3. **ProductPageRatio** — engineered feature. High ratio = focused product browsing
4. **ExitRates** — high exit rate pushes prediction strongly toward "won't buy"
5. **Administrative** — time spent on account/info pages signals engaged users

---

## 🌐 API Reference

### `POST /predict`
Predicts purchase intent and returns a recommendation.

**Request:**
```json
{
  "Administrative": 2,
  "Administrative_Duration": 80.0,
  "Informational": 0,
  "Informational_Duration": 0.0,
  "ProductRelated": 12,
  "ProductRelated_Duration": 720.5,
  "BounceRates": 0.02,
  "ExitRates": 0.04,
  "PageValues": 25.3,
  "SpecialDay": 0.0,
  "Month": 11,
  "OperatingSystems": 2,
  "Browser": 2,
  "Region": 1,
  "TrafficType": 2,
  "VisitorType": 2,
  "Weekend": 0,
  "ProductPageRatio": 0.8,
  "AvgTimePerPage": 60.0,
  "ExitBounceRisk": 0.06,
  "NearSpecialDay": 0,
  "HighPageValue": 1,
  "TotalPages": 14,
  "TotalDuration": 800.5
}
```

**Response:**
```json
{
  "will_purchase": true,
  "purchase_probability": 0.7823,
  "confidence": "High",
  "model_used": "ensemble",
  "recommendation": {
    "segment": "Convert",
    "recommended_action": "show_returning_visitor_offer",
    "reason": "Returning visitor with high page value — offer loyalty reward or saved cart reminder.",
    "urgency": "critical",
    "message": "Surface a direct CTA — streamline path to checkout immediately."
  }
}
```

### Other Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Model load status |
| `GET` | `/model-info` | Architecture summary |
| `GET` | `/segment-stats` | Segment distribution |
| `GET` | `/docs` | Swagger UI |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Models | scikit-learn, XGBoost, PyTorch |
| Hyperparameter Tuning | Optuna |
| Class Imbalance | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| API | FastAPI + Pydantic |
| Visualization | Matplotlib, Seaborn |
| Database | PostgreSQL (Docker) |
| Containerization | Docker + Docker Compose |
| Experiment Logging | Loguru |

---

## 📈 Business Impact

> "Targeting the top 20% of sessions by predicted purchase probability captures **77.7% of all actual purchases**."

This means a marketing team using this system can:
- Send discount emails to **20% of visitors** and reach **77.7% of buyers**
- Reduce wasted ad spend by **~80%** compared to blanket campaigns
- Personalize on-site experience in real time via the `/predict` API



---

## 📄 Dataset Citation

Sakar, C.O., Polat, S.O., Katircioglu, M. et al.  
*Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks.*  
Neural Comput & Applic 31, 6893–6908 (2019).
