import logging

from fastapi import APIRouter, HTTPException

from app.llm import LLMNotConfiguredError, LLMRequestError, generate_plain_language_summary
from app.ml.predictor import ModelNotLoadedError, predictor
from app.schemas import (
    ExplanationResponse,
    FeatureImportanceResponse,
    LLMSummaryResponse,
    PatientMetrics,
    PredictionResponse,
)

logger = logging.getLogger("heart_disease_api")

router = APIRouter(prefix="/api/v1", tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse)
def predict(metrics: PatientMetrics):
    try:
        result = predictor.predict(metrics.model_dump())
        return PredictionResponse(**result)
    except ModelNotLoadedError as e:
        logger.error("Prediction attempted with no model loaded: %s", e)
        raise HTTPException(status_code=503, detail="Model is not available. Try again shortly.")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        logger.exception("Unexpected error during prediction")
        raise HTTPException(status_code=500, detail="Prediction failed unexpectedly.")


@router.get("/features", response_model=FeatureImportanceResponse)
def get_features():
    """
    Returns model coefficients when available (linear models only).
    For non-linear models (e.g. RandomForest) coefficients don't exist in
    the same sense, so this is explicit about that instead of crashing or
    silently returning nonsense -- SHAP values are the more honest answer
    for tree-based models and are a Phase 2 item.
    """
    if not predictor.is_ready:
        raise HTTPException(status_code=503, detail="Model is not available.")

    if predictor.linear_coefficients is not None:
        return FeatureImportanceResponse(
            features=predictor.feature_names,
            coefficients=predictor.linear_coefficients,
            note="Coefficients are for a linear model; magnitude and sign are directly interpretable.",
        )

    model_type = type(predictor.model).__name__
    return FeatureImportanceResponse(
        features=predictor.feature_names,
        coefficients=None,
        note=(
            f"The selected model ({model_type}) is not linear, "
            "so raw coefficients aren't meaningful. See /api/v1/explain for "
            "per-patient SHAP-based explanations, or /api/v1/model-info for "
            "cross-validated metrics."
        ),
    )


@router.post("/explain-llm", response_model=LLMSummaryResponse)
def explain_llm(metrics: PatientMetrics):
    """
    Optional AI feature: turns the already-computed prediction + SHAP
    explanation into a plain-language paragraph. Requires ANTHROPIC_API_KEY
    to be set; returns 503 (not a crash) if it isn't, since the rest of the
    app is fully functional without this endpoint.
    """
    patient = metrics.model_dump()
    try:
        prediction = predictor.predict(patient)
        explanation = predictor.explain(patient)
        summary = generate_plain_language_summary(patient, prediction, explanation)
        return LLMSummaryResponse(summary=summary)
    except ModelNotLoadedError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except LLMNotConfiguredError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except LLMRequestError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        logger.exception("Unexpected error generating AI summary")
        raise HTTPException(status_code=500, detail="Failed to generate AI summary.")


@router.post("/explain", response_model=ExplanationResponse)
def explain(metrics: PatientMetrics):
    """
    Per-patient explanation: how much did each input push THIS prediction's
    risk up or down, via SHAP. This is distinct from /features, which gives
    a global (dataset-wide) importance ranking.
    """
    try:
        result = predictor.explain(metrics.model_dump())
        return ExplanationResponse(**result)
    except ModelNotLoadedError as e:
        logger.error("Explanation attempted with no model/background sample loaded: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        logger.exception("Unexpected error during explanation")
        raise HTTPException(status_code=500, detail="Explanation failed unexpectedly.")


@router.get("/model-info")
def model_info():
    if not predictor.metrics:
        raise HTTPException(status_code=503, detail="No metrics available. Train the model first.")
    return predictor.metrics
