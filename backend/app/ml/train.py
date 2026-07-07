"""
Training pipeline for the Heart Disease risk model.

What this does, and why:
- Pulls the real OpenML cardiovascular dataset (id 45547, ~70k patient records).
- Compares a few reasonable model families with cross-validation instead of
  fitting one LogisticRegression and calling it done.
- Calibrates the winning model's probabilities (Platt scaling via
  CalibratedClassifierCV) -- the app surfaces a risk PERCENTAGE to the user,
  so an uncalibrated "70%" that doesn't actually correspond to a 70% empirical
  rate is a real, user-facing correctness bug, not a cosmetic one.
- Persists the WINNING model, its scaler, its feature name order, a small
  background sample for SHAP explainability, and its evaluation + calibration
  metrics as separate artifacts -- so the API and any future retraining never
  have to guess at feature order or "trust" a number that isn't written down
  anywhere.

Run:
    python -m app.ml.train
"""
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    """Load the real OpenML cardiovascular dataset. Requires internet access."""
    print("Fetching OpenML Cardiovascular Disease dataset (id=45547)...")
    data = fetch_openml(data_id=45547, parser="auto", as_frame=True)
    df = data.frame.dropna().copy()
    df["age"] = (df["age"] / 365).astype(int)
    return df


CANDIDATE_MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "random_forest": RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1
    ),
}

SCORING = {
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
}


def compare_models(X_train, y_train) -> dict:
    """5-fold CV across candidate models. Returns per-model mean/std scores."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, model in CANDIDATE_MODELS.items():
        print(f"Cross-validating {name}...")
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=SCORING)
        results[name] = {
            metric: {
                "mean": float(np.mean(scores[f"test_{metric}"])),
                "std": float(np.std(scores[f"test_{metric}"])),
            }
            for metric in SCORING
        }
    return results


def select_best_model(cv_results: dict, metric: str = "roc_auc") -> str:
    return max(cv_results, key=lambda name: cv_results[name][metric]["mean"])


def evaluate_on_holdout(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred).tolist()
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": {
            "labels": ["low_risk", "high_risk"],
            "matrix": cm,  # [[TN, FP], [FN, TP]]
        },
    }


def evaluate_calibration(model, X_test, y_test, n_bins: int = 10) -> dict:
    """
    Checks whether predicted probabilities match observed frequencies.
    Brier score: mean squared error between predicted probability and
    actual outcome (0 or 1) -- lower is better, 0 is perfect.
    Calibration curve: for patients the model said were e.g. "70% risk",
    were roughly 70% of them actually positive? If not, the risk % shown
    to users is misleading regardless of how good ROC-AUC looks.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    brier = float(brier_score_loss(y_test, y_proba))
    true_freq, predicted_freq = calibration_curve(y_test, y_proba, n_bins=n_bins, strategy="quantile")
    return {
        "brier_score": brier,
        "brier_score_note": "0 = perfect, 0.25 = uninformative baseline for a 50/50 class split",
        "calibration_curve": {
            "predicted_probability": predicted_freq.tolist(),
            "observed_frequency": true_freq.tolist(),
        },
    }


def train_and_save_model():
    df = load_data()

    X = df.drop("cardio", axis=1)
    y = df["cardio"].astype(int)
    feature_names = list(X.columns)  # <-- the single source of truth for order

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    cv_results = compare_models(X_train, y_train)
    best_name = select_best_model(cv_results)
    print(f"Best model by mean CV ROC-AUC: {best_name}")

    # Calibrate the winning model's probabilities via 5-fold Platt scaling.
    # This is what actually gets fit and shipped -- not the raw candidate --
    # because the API surfaces a risk PERCENTAGE, and an uncalibrated
    # probability from e.g. a Random Forest can look confident without being
    # empirically accurate.
    base_model = CANDIDATE_MODELS[best_name]
    calibrated_model = CalibratedClassifierCV(base_model, method="sigmoid", cv=5)
    calibrated_model.fit(X_train, y_train)

    holdout_metrics = evaluate_on_holdout(calibrated_model, X_test, y_test)
    calibration_metrics = evaluate_calibration(calibrated_model, X_test, y_test)

    with open(ARTIFACTS_DIR / "model.pkl", "wb") as f:
        pickle.dump(calibrated_model, f)
    with open(ARTIFACTS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(ARTIFACTS_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    # A small random sample of real (scaled) training rows, needed as the
    # background reference distribution for SHAP at inference time. Not
    # sensitive on its own (no labels, no identifiers) -- just feature
    # vectors -- but kept small deliberately.
    rng = np.random.default_rng(RANDOM_STATE)
    background_idx = rng.choice(X_train.shape[0], size=min(100, X_train.shape[0]), replace=False)
    background_sample = X_train[background_idx].tolist()
    with open(ARTIFACTS_DIR / "background_sample.json", "w") as f:
        json.dump(background_sample, f)

    metrics = {
        "trained_at_unix": int(time.time()),
        "selected_model": best_name,
        "calibration_method": "sigmoid (Platt scaling), 5-fold",
        "dataset": {"source": "openml:45547", "n_rows": int(len(df))},
        "cross_validation": cv_results,
        "holdout_evaluation": holdout_metrics,
        "calibration": calibration_metrics,
        "feature_names": feature_names,
    }
    with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    write_model_card(metrics)

    print(f"Saved model, scaler, feature_names.json, metrics.json, background_sample.json to {ARTIFACTS_DIR}")
    print(json.dumps(holdout_metrics, indent=2))
    print(json.dumps(calibration_metrics, indent=2))


def write_model_card(metrics: dict):
    """
    A model card is a short, standard document describing what a model does,
    how it was evaluated, and where it should NOT be used -- standard
    practice for any model touching health data, and a good thing to be able
    to point to in an interview.
    """
    holdout = metrics["holdout_evaluation"]
    calib = metrics["calibration"]
    card = f"""# Model Card: Heart Disease Risk Classifier

## Summary
Predicts cardiovascular disease risk from 11 clinical/lifestyle metrics
(age, gender, height, weight, blood pressure, cholesterol, glucose,
smoking, alcohol use, physical activity).

- **Model type:** {metrics['selected_model']} (calibrated via {metrics['calibration_method']})
- **Trained on:** {metrics['dataset']['n_rows']:,} rows from the OpenML
  Cardiovascular Disease dataset (id 45547)
- **Trained at (unix time):** {metrics['trained_at_unix']}

## Intended use
A portfolio/demo application illustrating an end-to-end ML pipeline: data
ingestion, model comparison, calibration, explainability, and a served API.

## NOT intended for
- Actual clinical decision-making or diagnosis of any kind.
- Use on patient populations that differ meaningfully from the training
  data (e.g. different countries, age ranges, or measurement units)
  without re-evaluation.
- Any deployment context where a human doesn't remain responsible for the
  final decision.

## Performance (held-out test set)
| Metric | Value |
|---|---|
| Accuracy | {holdout['accuracy']:.3f} |
| Precision | {holdout['precision']:.3f} |
| Recall | {holdout['recall']:.3f} |
| ROC-AUC | {holdout['roc_auc']:.3f} |

Confusion matrix (rows = actual, columns = predicted):
```
                 Predicted Low   Predicted High
Actual Low       {holdout['confusion_matrix']['matrix'][0][0]:>14} {holdout['confusion_matrix']['matrix'][0][1]:>15}
Actual High      {holdout['confusion_matrix']['matrix'][1][0]:>14} {holdout['confusion_matrix']['matrix'][1][1]:>15}
```

## Calibration
Brier score: **{calib['brier_score']:.4f}** ({calib['brier_score_note']}).

The risk percentage shown in the UI is only meaningful if it's calibrated
-- i.e. among patients told "70% risk," roughly 70% should actually have
the condition. This is checked via `calibration_curve` at training time
and is why the deployed model is wrapped in `CalibratedClassifierCV`
rather than shipping raw `predict_proba` output.

## Known limitations
- Demo-quality baseline: no hyperparameter search beyond default settings
  for the compared model families.
- No subgroup/fairness evaluation (e.g. performance broken out by age
  band or gender) -- a real deployment would need this.
- Feature set is limited to what's in the source dataset; known risk
  factors not present in the data (family history, genetic markers,
  detailed lab panels) are not modeled.

## Reproducing this card
Run `python -m app.ml.train`. It regenerates this file, `metrics.json`,
and the model artifacts from the same dataset and pipeline.
"""
    with open(ARTIFACTS_DIR / "MODEL_CARD.md", "w") as f:
        f.write(card)


if __name__ == "__main__":
    train_and_save_model()
