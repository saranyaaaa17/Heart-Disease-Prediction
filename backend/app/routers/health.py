from fastapi import APIRouter

from app.ml.predictor import predictor
from app.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health():
    """Liveness/readiness check. A load balancer or uptime monitor hits this."""
    return HealthResponse(
        status="ok" if predictor.is_ready else "degraded",
        model_loaded=predictor.is_ready,
        model_type=type(predictor.model).__name__ if predictor.is_ready else None,
    )
