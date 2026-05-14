# Customer Purchase Intent Engine

A machine learning system that predicts whether an online shopper will make a purchase, segments them by intent level, and recommends the optimal marketing intervention — served through a REST API and an interactive React dashboard.

---

## What This Project Does

Most e-commerce platforms treat all visitors the same. This system doesn't. Given a single browsing session, it answers three questions:

1. Will this visitor buy? (purchase probability + binary prediction)
2. Which intent bucket do they fall into? (Cold / Warm / Hot / Convert)
3. What should we show them right now? (intervention recommendation via a learned bandit model)

---

## System Architecture

```
UCI Raw Session Data
        |
        v
Feature Engineering        7 domain-specific features derived from raw columns
        |
        v
Preprocessing              StandardScaler + SMOTE (applied to training set only)
        |
   _____|_____
  |     |     |
  v     v     v
Logistic  XGBoost   Neural Net
Regression (Optuna)  (PyTorch)
  |     |     |
  |_____|_____|
        |
        v
Stacking Ensemble          Out-of-fold meta-features across 5 stratified folds
        |
   _____|_____
  |           |
  v           v
SHAP          LinUCB Contextual Bandit
Explainer     Recommender
              |
              v
          FastAPI Server  →  React Dashboard
```

---

## Model Performance

Results on the held-out test set (2,466 sessions, never seen during training):

| Model | ROC-AUC | F1 | Precision | Recall | Lift @ Top 20% |
|---|---|---|---|---|---|
| XGBoost | 0.928 | 0.672 | 0.631 | 0.717 | 0.777 |
| Ensemble | 0.920 | 0.660 | 0.660 | 0.660 | 0.772 |
| Neural Net | 0.902 | 0.639 | 0.632 | 0.647 | 0.744 |
| Logistic | 0.897 | 0.628 | 0.585 | 0.678 | 0.730 |

**XGBoost** achieves the highest AUC at 0.928, which sits at the top end of published academic results on this dataset (typical range: 0.88 to 0.93).

**Ensemble** is deployed in the API. Its precision equals its recall at 0.660 — a sign of well-calibrated predictions, which is more important than raw accuracy when the goal is actionable scoring.

**Lift @ Top 20%** of 0.777 means: if you target only the top 20% of sessions by predicted probability, you capture 77.7% of all actual buyers. That is the core business value of the system.

---

## Dataset

**UCI Online Shoppers Purchasing Intention Dataset**

- Source: UCI Machine Learning Repository
- Link: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
- 12,330 browsing sessions collected from an online retail platform
- 18 raw features covering page categories, durations, bounce/exit rates, traffic type, visitor type, and temporal signals
- Binary target: Revenue (True = purchase completed)
- Class distribution: 15.5% positive (buyers), 84.5% negative — handled via SMOTE

### Engineered Features

Seven additional features are derived before training. All are interpretable and business-meaningful:

| Feature | Definition |
|---|---|
| TotalPages | Sum of administrative, informational, and product-related pages visited |
| TotalDuration | Total time spent across all page categories |
| ProductPageRatio | Product pages as a fraction of total pages |
| AvgTimePerPage | Total duration divided by total pages (engagement depth) |
| HighPageValue | Binary flag: PageValues in the top 25th percentile |
| NearSpecialDay | Binary flag: SpecialDay > 0 |
| ExitBounceRisk | BounceRates + ExitRates combined abandonment score |

---

## Models

### Logistic Regression
Standard L2-regularised logistic regression with class_weight=balanced. Serves as an interpretable baseline. F1-optimal threshold tuned on the validation set post-training.

### XGBoost
Gradient-boosted decision trees tuned with Optuna over 40 trials, optimising F1 score on the validation set. Tuned hyperparameters: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda, gamma, min_child_weight. F1-optimal threshold tuned post-training.

### Neural Network
Three-block feedforward network built in PyTorch: 24 → 256 → 128 → 64 → 1. Each block uses BatchNorm1d and Dropout. Trained with BCEWithLogitsLoss and a positive class weight to handle imbalance. CosineAnnealingLR scheduler. Best checkpoint restored after training. F1-optimal threshold tuned on validation set.

### Stacking Ensemble
A two-layer stacking architecture. In the first layer, each base model generates out-of-fold predictions across 5 stratified folds — this ensures the meta-learner never trains on predictions from data the base models themselves saw. In the second layer, a logistic regression meta-learner is trained on these OOF probability outputs. Threshold tuned on the clean validation set. This is the model deployed in the API.

### LinUCB Contextual Bandit Recommender
Replaces a static rule-based system. For each of 8 possible marketing actions, the bandit maintains a weight matrix updated with every prediction. At inference it selects the action with the highest Upper Confidence Bound — balancing exploitation of known-good actions with exploration of uncertain ones. Uses 6 session features as context: purchase probability, exit/bounce risk, page value, product page ratio, visitor type, and proximity to special day. Warm-started with 2,000 simulated interactions before first deployment.

---

## Repository Structure

```
customer-intent-engine/
|
|-- orchestrator.py              Master pipeline script
|-- docker-compose.yml
|-- Dockerfile
|-- requirements.txt
|-- .env.example
|
|-- data/
|   |-- raw/
|       |-- online_shoppers_intention.csv
|
|-- pipeline/
|   |-- ingest.py                Schema validation and loading
|   |-- features.py              Feature engineering
|   |-- preprocess.py            Encoding, scaling, SMOTE
|   |-- segments.py              Cold/Warm/Hot/Convert segmentation
|
|-- models/
|   |-- base_model.py            Abstract interface
|   |-- logistic_model.py
|   |-- xgboost_model.py
|   |-- neural_model.py
|   |-- ensemble_model.py
|   |-- bandit_recommender.py    LinUCB contextual bandit
|   |-- recommender.py           Bandit wrapper with action labels
|
|-- evaluation/
|   |-- metrics.py               Scoring + business lift metric
|   |-- shap_explainer.py        SHAP summary and beeswarm plots
|   |-- reporter.py              ROC, PR, confusion matrix, bar charts
|
|-- api/
|   |-- main.py                  FastAPI application
|   |-- schemas.py               Pydantic request/response schemas
|
|-- frontend/
|   |-- index.html
|   |-- vite.config.js
|   |-- package.json
|   |-- src/
|       |-- App.jsx
|       |-- main.jsx
|       |-- index.css
|       |-- components/
|           |-- PredictForm.jsx
|           |-- ResultCard.jsx
|           |-- MetricsDashboard.jsx
|
|-- reports/                     Auto-generated on pipeline run
    |-- artifacts/               Saved model weights and scalers
    |-- roc_curves.png
    |-- pr_curves.png
    |-- confusion_matrices.png
    |-- metrics_comparison.png
    |-- shap_summary_bar.png
    |-- shap_summary_beeswarm.png
    |-- model_comparison_metrics.csv
    |-- model_comparison_metrics.json
```

---

## Quickstart

### Local setup

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/customer-intent-engine.git
cd customer-intent-engine

# 2. Python environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place dataset
# Download from the UCI link above
# Save as: data/raw/online_shoppers_intention.csv

# 5. Run the full training pipeline
python orchestrator.py
# Trains all models, evaluates on test set, saves artifacts, generates reports

# 6. Start the API (Terminal 1)
uvicorn api.main:app --reload

# 7. Start the frontend (Terminal 2)
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

### Docker

```bash
cp .env.example .env
docker-compose up -d --build
docker exec -it intent-app python orchestrator.py
# API at http://localhost:8001/docs
```

---

## API Reference

Base URL: `http://localhost:8000`

### POST /predict

Accepts a session feature object. Returns purchase probability, segment, and recommended intervention.

Request body (key fields):

```json
{
  "ProductRelated": 12,
  "ProductRelated_Duration": 720.0,
  "PageValues": 25.3,
  "BounceRates": 0.02,
  "ExitRates": 0.04,
  "Month": 11,
  "VisitorType": 2,
  "Weekend": 0,
  "Administrative": 2,
  "Administrative_Duration": 80.0,
  "Informational": 0,
  "Informational_Duration": 0.0,
  "SpecialDay": 0.0,
  "OperatingSystems": 2,
  "Browser": 2,
  "Region": 1,
  "TrafficType": 2,
  "ProductPageRatio": 0.8,
  "ExitBounceRisk": 0.06,
  "NearSpecialDay": 0,
  "HighPageValue": 1,
  "TotalPages": 14,
  "TotalDuration": 800.0,
  "AvgTimePerPage": 57.1
}
```

Response:

```json
{
  "will_purchase": true,
  "purchase_probability": 0.7823,
  "confidence": "High",
  "model_used": "ensemble",
  "recommendation": {
    "segment": "Convert",
    "recommended_action": "show_returning_visitor_offer",
    "action_label": "Returning Visitor Loyalty Reward",
    "reason": "Returning visitor with high page value — offer loyalty reward or saved cart reminder.",
    "urgency": "critical",
    "message": "Surface a direct CTA — streamline path to checkout immediately.",
    "confidence_score": 2.341
  }
}
```

### Other endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | API and model status |
| GET | /model-info | Architecture summary |
| GET | /segment-stats | Live segment distribution |
| GET | /bandit-stats | Bandit action counts and average rewards |
| GET | /recent-predictions | Last 20 predictions logged in memory |
| GET | /docs | Swagger UI |

---

## Recommendation System

The bandit recommender maps each session to one of 8 actions:

| Action | Trigger Condition |
|---|---|
| show_social_proof | Cold segment, new visitor |
| show_discount | Warm segment baseline |
| show_urgency | Hot segment baseline |
| show_checkout_prompt | Convert segment baseline |
| show_exit_intent_offer | High exit/bounce risk on Warm or Hot |
| show_seasonal_urgency | Near a special day |
| show_returning_visitor_offer | Returning visitor with high PageValues |
| show_product_recommendation | High engagement but low product focus |

Unlike a static rule system, the bandit updates its action weights after every prediction. Actions that correlate with higher purchase probability receive stronger reinforcement over time.

---

## SHAP Analysis

The SHAP beeswarm plot identifies the top drivers of purchase prediction:

- **PageValues** is the strongest single feature. Sessions with high page values (high-priced product pages) push predictions strongly toward purchase.
- **Month** captures strong seasonality. November and December sessions behave differently from the rest of the year.
- **ProductPageRatio** is the most impactful engineered feature. A high ratio of product pages to total pages signals focused buying intent.
- **ExitRates** and **ExitBounceRisk** push predictions toward non-purchase when elevated.
- **Administrative** pages signal engaged, returning users when combined with other high-intent features.

SHAP plots are saved to `reports/` automatically after every pipeline run.

---

## Tech Stack

| Component | Technology |
|---|---|
| ML training | scikit-learn, XGBoost, PyTorch |
| Hyperparameter tuning | Optuna |
| Class imbalance | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Recommender | Custom LinUCB implementation |
| API | FastAPI, Pydantic v2, Uvicorn |
| Frontend | React 18, Vite, Recharts |
| Visualisation | Matplotlib, Seaborn |
| Containerisation | Docker, Docker Compose |
| Logging | Loguru |

---

## Business Interpretation

The Lift @ Top 20% metric is the most direct measure of business value. It answers: if you act on only the top-scored 20% of sessions, what fraction of actual buyers do you reach?

At 0.777, the system captures 77.7% of buyers by targeting 20% of traffic. A random targeting strategy would capture 20%. This represents a 3.9x improvement over baseline, which directly translates to reduced marketing spend per conversion.


---

