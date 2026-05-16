---
title: Customer Intent Engine
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Customer Purchase Intent Engine

**Live Demo:** https://nandininautiyal-customer-intent-engine.hf.space
**API Docs:** https://nandininautiyal-customer-intent-engine.hf.space/docs
**GitHub:** https://github.com/nandininautiyal/Customer-Intent-Project

---

## What This Project Does

When someone visits an online store, most platforms have no idea whether that person is going to buy something or just browse and leave. This system solves that.

Given a single browsing session — how many product pages someone visited, how long they stayed, whether they're a returning visitor, what month it is — the system answers three questions:

1. Will this person buy? (probability + yes/no prediction)
2. How serious are they? (Cold / Warm / Hot / Convert segment)
3. What should we show them right now? (marketing intervention recommendation)

Everything runs in real time through a live API and an interactive dashboard.

---

## Results

Tested on 2,466 sessions that the models never saw during training:

| Model | Accuracy | F1 | ROC-AUC | Captures of Top 20% |
|---|---|---|---|---|
| XGBoost | 89.9% | 0.674 | 0.929 | 78.0% |
| Ensemble | 90.0% | 0.668 | 0.923 | 75.9% |
| Neural Net | 88.7% | 0.648 | 0.903 | 73.6% |
| Logistic | 87.6% | 0.631 | 0.897 | 73.0% |

The most important number is the last column. If you take the top 20% of visitors ranked by predicted purchase probability and target only them, you capture 78% of all actual buyers. A random targeting strategy would capture 20%. That is a 3.9x improvement — which directly means less wasted marketing spend.

The Ensemble is deployed in the live API because it has the highest accuracy and best precision. XGBoost has the best raw AUC and is the strongest individual model.

---

## Dataset

UCI Online Shoppers Purchasing Intention Dataset — 12,330 real browsing sessions collected from an online retail platform.

- 18 raw features: page visit counts, time spent per category, bounce rate, exit rate, page value, traffic type, visitor type, month, weekend flag
- Target: did the session result in a purchase (15.5% yes, 84.5% no)
- Download: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

Seven additional features were engineered on top of the raw ones:

| Feature | What it measures |
|---|---|
| TotalPages | How many pages were visited in total |
| TotalDuration | Total time spent on the site |
| ProductPageRatio | What fraction of pages visited were product pages |
| AvgTimePerPage | How deeply engaged the visitor was per page |
| HighPageValue | Whether the session had unusually valuable pages |
| NearSpecialDay | Whether the session was close to a special day |
| ExitBounceRisk | Combined abandonment signal |

---

## How It Works

```
Raw session data
      |
      v
Feature engineering (7 new features added)
      |
      v
Preprocessing (StandardScaler + SMOTETomek to fix class imbalance)
      |
      |-----> Logistic Regression (baseline)
      |-----> XGBoost (tuned with Optuna, 80 trials)
      |-----> Neural Network (PyTorch, 3 layers, BatchNorm + Dropout)
      |
      v
Stacking Ensemble
  - Each base model generates predictions on data it never trained on (5-fold OOF)
  - XGBoost meta-learner learns when to trust each model
  - Threshold tuned to balance precision and recall
      |
      v
LinUCB Contextual Bandit Recommender
  - Chooses from 8 possible marketing interventions
  - Learns from every prediction which action works best for which visitor type
  - Updates its weights continuously — not hardcoded rules
      |
      v
FastAPI backend + React dashboard
```

---

## Models in Detail

**Logistic Regression** — the simplest model, used as a baseline. Tells us what minimum performance looks like and gives interpretable coefficients.

**XGBoost** — gradient boosted decision trees. Each tree corrects the mistakes of the previous one. Hyperparameters were tuned automatically using Optuna over 80 trials, optimising F1 score. This is the best single model with AUC 0.929.

**Neural Network** — a 3-layer PyTorch network (24 → 256 → 128 → 64 → 1) with BatchNorm and Dropout at each layer to prevent overfitting. Trained with a weighted loss function that penalises missing actual buyers more than false alarms.

**Stacking Ensemble** — combines all three models. Instead of voting, it generates out-of-fold probability predictions and trains a meta-learner on top of those. The meta-learner gets 14 features including pairwise products of model probabilities and the top 5 raw features identified by SHAP. This way it learns when XGBoost should be trusted more than the neural network and vice versa.

**LinUCB Bandit** — for each of 8 marketing actions, the bandit maintains a weight matrix. At inference it selects the action with the highest Upper Confidence Bound — balancing exploitation of known-good actions with exploration of less-tried ones. It updates after every prediction. Over time it learns, for example, that returning visitors with high page values respond better to loyalty rewards than generic discounts.

---

## Recommendation System

The system recommends one of 8 actions per visitor:

| Action | When it triggers |
|---|---|
| Show social proof | Cold segment, new visitor |
| Show discount | Warm segment |
| Show urgency nudge | Hot segment |
| Direct checkout CTA | Convert segment |
| Exit-intent popup | High abandonment risk detected |
| Seasonal offer | Session near a special day |
| Loyalty reward | Returning visitor with high page value |
| Product recommendation | Engaged but not focused on products |

Example API response:
```json
{
  "will_purchase": false,
  "purchase_probability": 0.43,
  "confidence": "Medium",
  "recommendation": {
    "segment": "Warm",
    "recommended_action": "show_exit_intent_offer",
    "action_label": "Exit-Intent Popup with Offer",
    "reason": "High exit risk detected — intercept with an exit-intent offer.",
    "urgency": "medium"
  }
}
```

---

## What SHAP Tells Us

SHAP (SHapley Additive exPlanations) explains why the model made each prediction. The top drivers identified:

- **PageValues** — the single strongest signal. Visitors on high-value product pages are serious buyers.
- **Month** — November and December sessions convert at much higher rates.
- **ProductPageRatio** — the engineered feature that measures how focused a visitor is on products. High ratio = buying intent.
- **ExitRates** — high exit rate strongly predicts the visitor will leave without buying.
- **VisitorType** — returning visitors convert at roughly twice the rate of new visitors.

---

## Project Structure

```
customer-intent-engine/
|
|-- orchestrator.py          Runs the full training pipeline in one command
|-- Dockerfile               Container setup for Hugging Face deployment
|-- start_hf.sh              Startup script (trains if needed, then serves)
|-- requirements.txt
|
|-- pipeline/
|   |-- ingest.py            Loads and validates the UCI dataset
|   |-- features.py          Engineers 7 new features
|   |-- preprocess.py        Scaling and SMOTETomek resampling
|   |-- segments.py          Cold/Warm/Hot/Convert segmentation logic
|
|-- models/
|   |-- base_model.py        Abstract interface all models follow
|   |-- logistic_model.py
|   |-- xgboost_model.py
|   |-- neural_model.py
|   |-- ensemble_model.py    OOF stacking with XGBoost meta-learner
|   |-- bandit_recommender.py  LinUCB contextual bandit
|   |-- recommender.py       Wraps the bandit with human-readable labels
|
|-- evaluation/
|   |-- metrics.py           Accuracy, F1, AUC, MCC, Lift@20%
|   |-- shap_explainer.py    SHAP summary and beeswarm plots
|   |-- reporter.py          ROC curves, PR curves, confusion matrices
|
|-- api/
|   |-- main.py              FastAPI app, serves both API and React frontend
|   |-- schemas.py           Input/output data validation
|
|-- frontend/
|   |-- src/
|   |   |-- App.jsx          Main app with Predict and Dashboard tabs
|   |   |-- components/
|   |   |   |-- PredictForm.jsx      Session input sliders
|   |   |   |-- ResultCard.jsx       Probability ring and recommendation
|   |   |   |-- MetricsDashboard.jsx Model comparison charts
|
|-- notebooks/
|   |-- exploratory_analysis.ipynb   EDA with 7 analysis sections
|
|-- reports/                 Auto-generated after training
    |-- artifacts/           Saved model weights and scalers
    |-- roc_curves.png
    |-- shap_summary_beeswarm.png
    |-- model_comparison_metrics.csv
```

---

## Running Locally

```bash
# Clone
git clone https://github.com/nandininautiyal/Customer-Intent-Project.git
cd Customer-Intent-Project

# Set up Python environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Place dataset
# Download from https://archive.ics.uci.edu/dataset/468
# Save as data/raw/online_shoppers_intention.csv

# Train all models (takes ~20 minutes)
python orchestrator.py

# Start API (Terminal 1)
uvicorn api.main:app --reload

# Start frontend (Terminal 2)
cd frontend
npm install
npm run dev

# Open http://localhost:5173
```

---

## API Endpoints

| Method | Endpoint | What it does |
|---|---|---|
| POST | /predict | Returns purchase probability + recommendation |
| GET | /health | Checks if models are loaded |
| GET | /model-info | Architecture summary |
| GET | /bandit-stats | What the bandit has learned so far |
| GET | /segment-stats | Distribution of visitor segments |
| GET | /docs | Interactive Swagger UI |

---

## Tech Stack

| What | Tool |
|---|---|
| ML models | scikit-learn, XGBoost, PyTorch |
| Hyperparameter tuning | Optuna |
| Class imbalance | imbalanced-learn (SMOTETomek) |
| Explainability | SHAP |
| API | FastAPI + Pydantic |
| Frontend | React 18 + Vite + Recharts |
| Deployment | Hugging Face Spaces (Docker) |
| Logging | Loguru |


---

