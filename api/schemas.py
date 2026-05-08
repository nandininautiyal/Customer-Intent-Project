from pydantic import BaseModel, Field
from typing import Optional


class SessionInput(BaseModel):
    Administrative: int = Field(..., ge=0)
    Administrative_Duration: float = Field(..., ge=0)
    Informational: int = Field(..., ge=0)
    Informational_Duration: float = Field(..., ge=0)
    ProductRelated: int = Field(..., ge=0)
    ProductRelated_Duration: float = Field(..., ge=0)
    BounceRates: float = Field(..., ge=0, le=1)
    ExitRates: float = Field(..., ge=0, le=1)
    PageValues: float = Field(..., ge=0)
    SpecialDay: float = Field(..., ge=0, le=1)
    Month: int = Field(..., ge=1, le=12)
    OperatingSystems: int = Field(..., ge=1)
    Browser: int = Field(..., ge=1)
    Region: int = Field(..., ge=1)
    TrafficType: int = Field(..., ge=1)
    VisitorType: int = Field(..., ge=0, le=2)
    Weekend: int = Field(..., ge=0, le=1)
    # Engineered features
    TotalPages: float = 0
    TotalDuration: float = 0
    ProductPageRatio: float = 0
    AvgTimePerPage: float = 0
    HighPageValue: int = 0
    NearSpecialDay: int = 0
    ExitBounceRisk: float = 0


class RecommendationDetail(BaseModel):
    segment: str                  # Cold / Warm / Hot / Convert
    recommended_action: str       # machine-readable action key
    reason: str                   # human-readable explanation
    urgency: str                  # low / medium / high / critical
    message: str                  # business-facing instruction


class PredictionResponse(BaseModel):
    will_purchase: bool
    purchase_probability: float
    confidence: str               # High / Medium / Low
    model_used: str = "ensemble"
    recommendation: RecommendationDetail