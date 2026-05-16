import numpy as np
import joblib
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from loguru import logger
from pathlib import Path

from api.schemas import SessionInput, PredictionResponse, BanditStatsResponse
from models.logistic_model import LogisticIntentModel
from models.xgboost_model import XGBoostIntentModel
from models.neural_model import NeuralIntentModel
from models.ensemble_model import EnsembleIntentModel
from models.recommender import IntentRecommender
from models.bandit_recommender import ContextualBandit
from pipeline.segments import get_segment_distribution

app = FastAPI(
    title="Customer Purchase Intent + Recommendation API",
    description="Predicts purchase intent and recommends the optimal marketing intervention.",
    version="3.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACTS    = Path("reports/artifacts")
FRONTEND_DIR = Path("frontend/dist")
models_loaded = {}
prediction_log = []


@app.on_event("startup")
def load_models():
    try:
        scaler = joblib.load(ARTIFACTS / "scaler.pkl")
        models_loaded["scaler"] = scaler

        logistic = LogisticIntentModel()
        logistic.load(str(ARTIFACTS / "logistic.pkl"))

        xgb = XGBoostIntentModel(tune=False)
        xgb.load(str(ARTIFACTS / "xgboost.pkl"))

        input_dim = scaler.n_features_in_
        neural = NeuralIntentModel(input_dim=input_dim)
        neural.load(str(ARTIFACTS / "neural.pth"))

        ensemble = EnsembleIntentModel([logistic, xgb, neural])
        ensemble.load(str(ARTIFACTS / "ensemble_meta.pkl"))

        bandit_path = ARTIFACTS / "bandit.pkl"
        bandit = ContextualBandit(n_features=6, alpha=1.0)
        if bandit_path.exists():
            bandit.load(str(bandit_path))
            logger.info("[Bandit] Loaded existing bandit state.")
        else:
            bandit.warm_start(n_samples=2000)
            bandit.save(str(bandit_path))
            logger.info("[Bandit] Warm-started and saved.")

        recommender = IntentRecommender(bandit=bandit)

        models_loaded["ensemble"]    = ensemble
        models_loaded["bandit"]      = bandit
        models_loaded["recommender"] = recommender

        logger.info("All models loaded successfully.")

    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise




@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "models_ready": "ensemble" in models_loaded,
        "bandit_ready": "bandit" in models_loaded,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(session: SessionInput):
    if "ensemble" not in models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    feature_array = np.array([[
        session.Administrative, session.Administrative_Duration,
        session.Informational, session.Informational_Duration,
        session.ProductRelated, session.ProductRelated_Duration,
        session.BounceRates, session.ExitRates, session.PageValues,
        session.SpecialDay, session.Month, session.OperatingSystems,
        session.Browser, session.Region, session.TrafficType,
        session.VisitorType, session.Weekend,
        session.TotalPages, session.TotalDuration,
        session.ProductPageRatio, session.AvgTimePerPage,
        session.HighPageValue, session.NearSpecialDay,
        session.ExitBounceRisk
    ]])

    scaler   = models_loaded["scaler"]
    X_scaled = scaler.transform(feature_array)

    ensemble = models_loaded["ensemble"]
    proba    = float(ensemble.predict_proba(X_scaled)[0])
    pred     = proba >= ensemble.optimal_threshold

    confidence = (
        "High"   if proba > 0.75 or proba < 0.25 else
        "Medium" if proba > 0.60 or proba < 0.40 else
        "Low"
    )

    session_context = {
        "ExitBounceRisk":   session.ExitBounceRisk,
        "PageValues":       session.PageValues,
        "ProductPageRatio": session.ProductPageRatio,
        "VisitorType":      session.VisitorType,
        "AvgTimePerPage":   session.AvgTimePerPage,
        "NearSpecialDay":   session.NearSpecialDay,
    }

    recommender = models_loaded["recommender"]
    rec = recommender.recommend(proba, session_context)

    bandit = models_loaded["bandit"]
    bandit.save(str(ARTIFACTS / "bandit.pkl"))

    prediction_log.append({
        "probability":   proba,
        "will_purchase": bool(pred),
        "segment":       rec["segment"],
        "action":        rec["recommended_action"],
    })

    return PredictionResponse(
        will_purchase=bool(pred),
        purchase_probability=round(proba, 4),
        confidence=confidence,
        recommendation=rec
    )


@app.get("/bandit-stats", response_model=BanditStatsResponse)
def bandit_stats():
    if "bandit" not in models_loaded:
        raise HTTPException(status_code=503, detail="Bandit not loaded.")
    bandit = models_loaded["bandit"]
    return BanditStatsResponse(
        action_stats=bandit.get_action_stats(),
        total_predictions=len(prediction_log)
    )


@app.get("/segment-stats")
def segment_stats():
    if not prediction_log:
        sample = np.random.beta(2, 8, 500)
    else:
        sample = np.array([p["probability"] for p in prediction_log])
    return {"segment_distribution": get_segment_distribution(sample)}


@app.get("/recent-predictions")
def recent_predictions(limit: int = 20):
    return {"predictions": prediction_log[-limit:]}


@app.get("/model-info")
def model_info():
    return {
        "base_models":  ["Logistic Regression", "XGBoost (Optuna)", "Neural Network (PyTorch)"],
        "ensemble":     "OOF Stacking with XGBoost meta-learner",
        "recommender":  "LinUCB Contextual Bandit (6 context features, 8 actions)",
        "segments":     ["Cold", "Warm", "Hot", "Convert"],
        "dataset":      "UCI Online Shoppers Intention (12,330 sessions)",
        "features":     24,
    }




if FRONTEND_DIR.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=str(FRONTEND_DIR / "assets")),
        name="assets"
    )

    @app.get("/")
    def serve_frontend():
        return FileResponse(str(FRONTEND_DIR / "index.html"))

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        """
        Catch-all route for React Router.
        Any path that isn't an API route returns index.html
        so React handles the routing client-side.
        """
        file_path = FRONTEND_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(FRONTEND_DIR / "index.html"))