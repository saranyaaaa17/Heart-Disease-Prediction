FROM python:3.12-slim

WORKDIR /app

# System deps for scikit-learn/pandas wheels build faster with these present
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY artifacts ./artifacts

ENV ENVIRONMENT=production
ENV CORS_ORIGINS=http://localhost:5173

EXPOSE 8000

# NOTE: artifacts/ must contain model.pkl, scaler.pkl, feature_names.json
# before the container will serve real predictions. Run `python -m app.ml.train`
# and mount/copy the artifacts/ directory, or run training as a separate
# init step in your deployment pipeline.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
