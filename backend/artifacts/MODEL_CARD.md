# Model Card: Heart Disease Risk Classifier

## Summary
Predicts cardiovascular disease risk from 11 clinical/lifestyle metrics
(age, gender, height, weight, blood pressure, cholesterol, glucose,
smoking, alcohol use, physical activity).

- **Model type:** random_forest (calibrated via sigmoid (Platt scaling), 5-fold)
- **Trained on:** 70,000 rows from the OpenML
  Cardiovascular Disease dataset (id 45547)
- **Trained at (unix time):** 1783397272

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
| Accuracy | 0.731 |
| Precision | 0.758 |
| Recall | 0.678 |
| ROC-AUC | 0.797 |

Confusion matrix (rows = actual, columns = predicted):
```
                 Predicted Low   Predicted High
Actual Low                 5491            1513
Actual High                2254            4742
```

## Calibration
Brier score: **0.1825** (0 = perfect, 0.25 = uninformative baseline for a 50/50 class split).

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
