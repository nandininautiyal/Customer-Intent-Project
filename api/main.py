import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from loguru import logger
from pathlib import Path

from api.schemas import SessionInput, PredictionResponse, RecommendationDetail
from models.logistic_model import LogisticIntentModel
from models.xgboost_model import XGBoostIntentModel
from models.neural_model import NeuralIntentModel
from models.ensemble_model import EnsembleIntentModel
from models.recommender import IntentRecommender
from pipeline.segments import get_segment_distribution

app = FastAPI(
    title="Customer Purchase Intent + Recommendation API",
    description="Predicts purchase intent and recommends the optimal marketing intervention.",
    version="2.0.0"
)

ARTIFACTS = Path("reports/artifacts")
models_loaded = {}
recommender = IntentRecommender()


@app.on_event("startup")
def load_models():
    try:
        scaler = joblib.load(ARTIFACTS / "scaler.pkl")
        models_loaded["scaler"] = scaler

        logistic = LogisticIntentModel()
        logistic.load(ARTIFACTS / "logistic.pkl")

        xgb = XGBoostIntentModel(tune=False)
        xgb.load(ARTIFACTS / "xgboost.pkl")

        input_dim = scaler.n_features_in_
        neural = NeuralIntentModel(input_dim=input_dim)
        neural.load(ARTIFACTS / "neural.pth")

        ensemble = EnsembleIntentModel([logistic, xgb, neural])
        ensemble.load(ARTIFACTS / "ensemble_meta.pkl")

        models_loaded["ensemble"] = ensemble
        logger.info("All models loaded successfully.")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")


@app.get("/health")
def health_check():
    return {"status": "ok", "models_ready": "ensemble" in models_loaded}


@app.post("/predict", response_model=PredictionResponse)
def predict(session: SessionInput):
    if "ensemble" not in models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

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

    scaler = models_loaded["scaler"]
    X_scaled = scaler.transform(feature_array)

    ensemble = models_loaded["ensemble"]
    proba = float(ensemble.predict_proba(X_scaled)[0])
    pred = proba >= 0.5

    confidence = (
        "High" if proba > 0.75 or proba < 0.25 else
        "Medium" if proba > 0.6 or proba < 0.4 else
        "Low"
    )

    # Build session context dict for recommender
    session_context = {
        "ExitBounceRisk": session.ExitBounceRisk,
        "PageValues": session.PageValues,
        "ProductPageRatio": session.ProductPageRatio,
        "VisitorType": session.VisitorType,
        "AvgTimePerPage": session.AvgTimePerPage,
        "NearSpecialDay": session.NearSpecialDay,
    }

    rec = recommender.recommend(proba, session_context)

    return PredictionResponse(
        will_purchase=bool(pred),
        purchase_probability=round(proba, 4),
        confidence=confidence,
        recommendation=RecommendationDetail(**rec)
    )


@app.get("/segment-stats")
def segment_stats():
    """
    Returns segment distribution across a sample of test probabilities.
    In production this would query stored predictions from the database.
    """
    sample_probas = np.random.beta(2, 5, 500)  # approximates real distribution
    dist = get_segment_distribution(sample_probas)
    return {"segment_distribution": dist}


@app.get("/model-info")
def model_info():
    return {
        "models": ["Logistic Regression", "XGBoost", "Neural Network"],
        "ensemble": "Stacking with Logistic meta-learner",
        "recommender": "Rule-based with 6 contextual override rules",
        "segments": ["Cold", "Warm", "Hot", "Convert"],
        "dataset": "UCI Online Shoppers Intention",
        "features": 24
    }