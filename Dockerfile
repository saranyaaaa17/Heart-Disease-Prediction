FROM python:3.12-slim

WORKDIR /app

# System deps for scikit-learn/pandas wheels build faster with these present
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app
COPY backend/artifacts ./artifacts
COPY backend/start.sh ./start.sh
RUN chmod +x ./start.sh

ENV ENVIRONMENT=production
ENV CORS_ORIGINS=http://localhost:5173

EXPOSE 8000

# Use the PORT env var injected by hosts like Render; fallback to 8000.
CMD ["sh", "-c", "./start.sh"]
