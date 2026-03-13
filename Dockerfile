# Python Backend Dockerfile
FROM python:3.9-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies necessary for compiling C-extensions (like numpy/torch)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies strictly
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Production Stage ---
FROM python:3.9-slim

WORKDIR /app

# Copy installed dependencies from the builder image
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy Application logic
COPY src/ ./src/
COPY api/ ./api/
COPY configs/ ./configs/

# Expose FastAPI default port
EXPOSE 8000

# Specify environment vars
ENV WORKERS=4
ENV MODULE_NAME=api.main
ENV VARIABLE_NAME=app

# Start the uvicorn server serving the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
