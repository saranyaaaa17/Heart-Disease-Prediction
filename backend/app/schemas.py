from pydantic import BaseModel, Field


class PatientMetrics(BaseModel):
    """
    Input clinical metrics. Bounds are loose physiological sanity limits,
    not diagnostic thresholds -- they exist to reject obviously malformed
    requests (e.g. negative height) before they hit the model.
    """

    age: float = Field(..., ge=1, le=120, description="Age in years")
    gender: float = Field(
        ..., ge=1, le=2,
        description="1 = female, 2 = male (matches the source dataset's encoding)",
    )
    height: float = Field(..., ge=50, le=250, description="Height in cm")
    weight: float = Field(..., ge=2, le=400, description="Weight in kg")
    ap_hi: float = Field(..., ge=50, le=300, description="Systolic blood pressure")
    ap_lo: float = Field(..., ge=30, le=200, description="Diastolic blood pressure")
    cholesterol: float = Field(..., ge=1, le=3, description="1=normal, 2=above normal, 3=well above normal")
    gluc: float = Field(..., ge=1, le=3, description="1=normal, 2=above normal, 3=well above normal")
    smoke: float = Field(..., ge=0, le=1)
    alco: float = Field(..., ge=0, le=1)
    active: float = Field(..., ge=0, le=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 54,
                "gender": 2,
                "height": 175,
                "weight": 82,
                "ap_hi": 130,
                "ap_lo": 85,
                "cholesterol": 2,
                "gluc": 1,
                "smoke": 0,
                "alco": 0,
                "active": 1,
            }
        }
    }


class PredictionResponse(BaseModel):
    is_high_risk: bool
    risk_label: str
    risk_probability_pct: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str | None = None


class FeatureImportanceResponse(BaseModel):
    features: list[str]
    coefficients: list[float] | None = None
    note: str


class FeatureContribution(BaseModel):
    feature: str
    contribution: float


class ExplanationResponse(BaseModel):
    base_risk_probability: float
    feature_contributions: list[FeatureContribution]
    note: str


class LLMSummaryResponse(BaseModel):
    summary: str
