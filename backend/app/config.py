"""
Centralized settings, loaded from environment variables (with sane
defaults for local dev). This replaces hardcoded `allow_origins=["*"]`
and similar values scattered through the old main.py.
"""
import os
from functools import lru_cache


class Settings:
    app_name: str = "Heart Disease Prediction API"
    api_version: str = "v1"
    environment: str = os.getenv("ENVIRONMENT", "development")

    # Comma-separated list, e.g. "http://localhost:5173,https://myapp.com"
    cors_origins: list[str] = os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")

    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # Optional: enables /api/v1/explain-llm. If unset, that endpoint returns
    # a clear 503 instead of crashing -- the rest of the app works fine
    # without it, this is deliberately an optional add-on, not a dependency.
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    # Model strings change over time -- check https://docs.claude.com for
    # the current list before relying on this default in production.
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")


@lru_cache
def get_settings() -> Settings:
    return Settings()
