"""
Loads the trained model artifacts once at startup and exposes a single
`predict()` function.

The important fix vs. the original code: feature order is never hardcoded
here. It's read from `feature_names.json`, which was written by train.py
from the *actual* training dataframe's column order. If training ever
changes column order, this code adapts automatically instead of silently
feeding the model mislabeled values.
"""
import json
import logging
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import shap

logger = logging.getLogger("heart_disease_api")

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"


class ModelNotLoadedError(RuntimeError):
    pass


class Predictor:
    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR):
        self.artifacts_dir = artifacts_dir
        self.model = None
        self.scaler = None
        self.feature_names: list[str] = []
        self.metrics: dict = {}
        self.background_sample: np.ndarray | None = None
        self._explainer = None
        self._load()

    def _load(self):
        try:
            with open(self.artifacts_dir / "model.pkl", "rb") as f:
                self.model = pickle.load(f)
            with open(self.artifacts_dir / "scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
            with open(self.artifacts_dir / "feature_names.json") as f:
                self.feature_names = json.load(f)
            metrics_path = self.artifacts_dir / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    self.metrics = json.load(f)
            background_path = self.artifacts_dir / "background_sample.json"
            if background_path.exists():
                with open(background_path) as f:
                    self.background_sample = np.array(json.load(f))
            logger.info(
                "Model artifacts loaded: model=%s features=%s",
                type(self.model).__name__,
                self.feature_names,
            )
        except FileNotFoundError as e:
            logger.error("Model artifacts missing: %s", e)
            self.model = None
            self.scaler = None

    @property
    def linear_coefficients(self) -> list[float] | None:
        """
        Returns coefficients if the underlying model is linear, unwrapping
        CalibratedClassifierCV if that's what's deployed. Returns None for
        genuinely non-linear models (e.g. RandomForest) -- callers should
        fall back to SHAP (/explain) rather than pretend a coefficient exists.
        """
        if self.model is None:
            return None
        if hasattr(self.model, "coef_"):
            return self.model.coef_[0].tolist()
        if hasattr(self.model, "calibrated_classifiers_"):
            inner = self.model.calibrated_classifiers_[0].estimator
            if hasattr(inner, "coef_"):
                return inner.coef_[0].tolist()
        return None

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.scaler is not None

    def predict(self, patient: Dict[str, float]) -> dict:
        if not self.is_ready:
            raise ModelNotLoadedError(
                "Model artifacts are not loaded. Run `python -m app.ml.train` first."
            )

        missing = [name for name in self.feature_names if name not in patient]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        # Build the feature vector in the EXACT order the model was trained on.
        ordered_values = [patient[name] for name in self.feature_names]
        features = np.array([ordered_values])

        scaled = self.scaler.transform(features)
        prediction = int(self.model.predict(scaled)[0])
        probability = float(self.model.predict_proba(scaled)[0][1])

        return {
            "is_high_risk": bool(prediction == 1),
            "risk_label": "High Risk of Heart Disease" if prediction == 1 else "Low Risk of Heart Disease",
            "risk_probability_pct": round(probability * 100, 2),
        }

    def _get_explainer(self):
        """
        Built lazily (not at import time) since constructing the SHAP
        explainer has real cost, and many processes -- like tests -- never
        call /explain at all.

        Uses a model-agnostic Permutation explainer over predict_proba
        rather than TreeExplainer/LinearExplainer specifically, because
        the deployed model is wrapped in CalibratedClassifierCV -- an
        explainer tied to one model family would break the moment the
        winning candidate model type changes.
        """
        if self._explainer is None:
            if self.background_sample is None:
                raise ModelNotLoadedError(
                    "No background_sample.json found. Re-run training to enable explanations."
                )
            self._explainer = shap.Explainer(
                self.model.predict_proba,
                self.background_sample,
                feature_names=self.feature_names,
            )
        return self._explainer

    def explain(self, patient: Dict[str, float]) -> dict:
        """
        Returns per-feature SHAP values for the HIGH-RISK class on this one
        prediction -- i.e. "how much did each input push this specific
        patient's risk up or down," not a global importance ranking.
        """
        if not self.is_ready:
            raise ModelNotLoadedError(
                "Model artifacts are not loaded. Run `python -m app.ml.train` first."
            )
        missing = [name for name in self.feature_names if name not in patient]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        ordered_values = [patient[name] for name in self.feature_names]
        features = np.array([ordered_values])
        scaled = self.scaler.transform(features)

        explainer = self._get_explainer()
        shap_values = explainer(scaled)

        # shap_values.values shape is (1, n_features, n_classes) for
        # predict_proba-based explainers; index 1 = the "high risk" class.
        contributions = shap_values.values[0, :, 1]
        base_value = float(shap_values.base_values[0, 1])

        per_feature = sorted(
            (
                {"feature": name, "contribution": float(value)}
                for name, value in zip(self.feature_names, contributions)
            ),
            key=lambda item: abs(item["contribution"]),
            reverse=True,
        )

        return {
            "base_risk_probability": round(base_value * 100, 2),
            "feature_contributions": per_feature,
            "note": (
                "Positive contribution = pushed risk higher for this patient; "
                "negative = pushed it lower. Computed relative to a background "
                "sample of real training data, not a fixed 'average patient'."
            ),
        }


# Singleton used by the FastAPI app
predictor = Predictor()
