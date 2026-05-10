# ----- Build stage -----
FROM python:3.9-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/base.txt requirements/api.txt ./requirements/
RUN pip install --no-cache-dir -r requirements/api.txt

# ----- Production stage -----
FROM python:3.9-slim

WORKDIR /app

# Run as non-root for security
RUN useradd --create-home appuser

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/requirements ./requirements

# Copy application source
COPY src/ ./src/
COPY configs/ ./configs/

# Model artifacts are NOT baked into the image — mount them at runtime:
#   docker run -v $(pwd)/models:/app/models fraud_detection_api:latest
# This keeps the image reproducible and allows model updates without rebuilds.
RUN mkdir -p /app/models && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

ENV WORKERS=4

# Use $WORKERS for horizontal concurrency; override at runtime with -e WORKERS=N
CMD uvicorn src.deployment.api.main:app --host 0.0.0.0 --port 8000 --workers $WORKERS
