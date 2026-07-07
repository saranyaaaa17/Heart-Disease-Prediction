# Legacy code

These files are the pre-Phase-1 versions of the app, kept for reference and
git history rather than deleted outright.

- `app_flask_legacy.py` — the original Flask implementation. The project
  had switched to FastAPI (`main.py`) but this was left in the repo,
  unused and undocumented, alongside it. Duplicate, dead entry points
  are a common thing reviewers flag, so it's been removed from the active
  codebase. It is **not** wired into `docker-compose.yml` or CI.
- `main_original_fastapi.py` — the original single-file FastAPI app.
  Superseded by `backend/app/` (see the Phase 1 notes in the root README
  for what changed and why: feature-order safety, validation, logging,
  versioning, tests).
- `train_original.py` — the original training script. Superseded by
  `backend/app/ml/train.py`, which adds cross-validation, model
  comparison, and persists feature names/metrics as artifacts instead of
  only a model.pkl.
- `templates/` — unused Flask templates from `app_flask_legacy.py`.

If you're happy with the Phase 1 rewrite after reviewing it, this whole
folder is safe to delete before you push.
