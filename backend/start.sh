#!/bin/sh
set -e

# Start script for the backend container.
# If model artifacts are missing, run training at startup (requires internet).

ARTIFACTS_DIR=/app/artifacts

# If MODEL_URL is provided, try to download artifacts archive before training.
if [ ! -f "$ARTIFACTS_DIR/model.pkl" ] || [ ! -f "$ARTIFACTS_DIR/scaler.pkl" ]; then
  if [ -n "${MODEL_URL:-}" ]; then
    echo "MODEL_URL provided — attempting to download artifacts from $MODEL_URL"
    TMP=/tmp/artifacts_download
    mkdir -p "$TMP"
    set -o pipefail
    if echo "$MODEL_URL" | grep -E "\.tar\.gz$|\.tgz$" >/dev/null 2>&1; then
      curl -fsSL "$MODEL_URL" -o "$TMP/artifacts.tar.gz" && tar -xzf "$TMP/artifacts.tar.gz" -C "$TMP"
    elif echo "$MODEL_URL" | grep -E "\.zip$" >/dev/null 2>&1; then
      curl -fsSL "$MODEL_URL" -o "$TMP/artifacts.zip" && unzip -q "$TMP/artifacts.zip" -d "$TMP"
    else
      # Try to download a single file (e.g. model.pkl) into artifacts
      mkdir -p "$ARTIFACTS_DIR"
      echo "Unknown archive extension; attempting to download directly to $ARTIFACTS_DIR"
      curl -fsSL "$MODEL_URL" -o "$ARTIFACTS_DIR/$(basename "$MODEL_URL")" || true
    fi

    # If archive extracted, move files into ARTIFACTS_DIR
    if [ -d "$TMP/artifacts" ]; then
      cp -R "$TMP/artifacts/." "$ARTIFACTS_DIR/" || true
    else
      cp -R "$TMP/." "$ARTIFACTS_DIR/" || true
    fi

    if [ -f "$ARTIFACTS_DIR/model.pkl" ] && [ -f "$ARTIFACTS_DIR/scaler.pkl" ]; then
      echo "Artifacts downloaded successfully."
    else
      echo "Artifacts not found after download; falling back to training."
      python -m app.ml.train
    fi
  else
    echo "Model artifacts not found in $ARTIFACTS_DIR — running training..."
    python -m app.ml.train
  fi
else
  echo "Model artifacts present — skipping training."
fi

# Start the ASGI server using the PORT provided by the host, default 8000
PORT=${PORT:-8000}
exec uvicorn app.main:app --host 0.0.0.0 --port $PORT
