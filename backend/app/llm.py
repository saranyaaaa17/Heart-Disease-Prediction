"""
Turns a prediction + its SHAP explanation into a plain-language paragraph
via the Anthropic API. This is deliberately the ONE "AI feature" added to
this project -- not a medical chatbot, not RAG, not voice -- because it's
the one that's actually low-risk and high-value for a demo: it explains an
already-computed, already-validated model output in plain English, rather
than generating new medical claims of its own.

Requires ANTHROPIC_API_KEY to be set. If it isn't, callers get a clear
LLMNotConfiguredError rather than the app crashing -- this feature is
optional, the core prediction API works fine without it.
"""
import logging
from typing import Dict, List

import httpx

from app.config import get_settings

logger = logging.getLogger("heart_disease_api")

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


class LLMNotConfiguredError(RuntimeError):
    pass


class LLMRequestError(RuntimeError):
    pass


def _build_prompt(patient: Dict[str, float], prediction: dict, top_factors: List[dict]) -> str:
    factors_text = "\n".join(
        f"- {f['feature']}: {'pushes risk up' if f['contribution'] >= 0 else 'pushes risk down'}"
        for f in top_factors
    )

    return f"""You are helping explain a cardiovascular disease risk model's output to a
patient in plain, calm, non-alarming language.

Model output:
- Risk classification: {prediction['risk_label']}
- Risk probability: {prediction['risk_probability_pct']}%

Top contributing factors for this specific patient (from a SHAP explanation):
{factors_text}

Patient's raw inputs: {patient}

Write a short (3-5 sentence) plain-language explanation of this result for
the patient. Rules:
- Do NOT diagnose anything or state medical certainty.
- Explicitly note this is a statistical estimate from a demo model, not a
  medical diagnosis, and that they should discuss any concerns with a
  real doctor.
- Mention 1-2 of the top contributing factors in plain terms (e.g. "your
  blood pressure reading" rather than "ap_hi").
- Keep the tone calm and factual, not alarming, regardless of the result.
"""


def generate_plain_language_summary(patient: Dict[str, float], prediction: dict, explanation: dict) -> str:
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise LLMNotConfiguredError(
            "ANTHROPIC_API_KEY is not set. This feature is optional -- "
            "set the environment variable to enable it."
        )

    top_factors = explanation["feature_contributions"][:5]
    prompt = _build_prompt(patient, prediction, top_factors)

    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.post(
                ANTHROPIC_API_URL,
                headers={
                    "x-api-key": settings.anthropic_api_key,
                    "anthropic-version": ANTHROPIC_VERSION,
                    "content-type": "application/json",
                },
                json={
                    "model": settings.anthropic_model,
                    "max_tokens": 400,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as e:
        logger.error("Anthropic API returned an error: %s", e.response.text)
        raise LLMRequestError(f"AI summary service returned an error: {e.response.status_code}")
    except httpx.HTTPError as e:
        logger.error("Anthropic API request failed: %s", e)
        raise LLMRequestError("AI summary service is unreachable.")

    text_blocks = [block["text"] for block in data.get("content", []) if block.get("type") == "text"]
    if not text_blocks:
        raise LLMRequestError("AI summary service returned an empty response.")
    return "\n".join(text_blocks).strip()
