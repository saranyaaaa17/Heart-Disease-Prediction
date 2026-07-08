#!/bin/sh
set -e

# Start script for the backend container.
# If model artifacts are missing, run training at startup (requires internet).

ARTIFACTS_DIR=/app/artifacts

if [ ! -f "$ARTIFACTS_DIR/model.pkl" ] || [ ! -f "$ARTIFACTS_DIR/scaler.pkl" ]; then
  echo "Model artifacts not found in $ARTIFACTS_DIR — running training..."
  python -m app.ml.train
else
  echo "Model artifacts present — skipping training."
fi

# Start the ASGI server using the PORT provided by the host, default 8000
PORT=${PORT:-8000}
exec uvicorn app.main:app --host 0.0.0.0 --port $PORT
