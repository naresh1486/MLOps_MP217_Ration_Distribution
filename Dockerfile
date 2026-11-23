# Dockerfile for MP217 Ration Shop Distribution Prediction API (Kochi)
# SCSE3040 MLOps - Group G230

FROM python:3.12-slim

# Prevent python from generating pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app

# Set container working directory
WORKDIR /app

# Copy requirements.txt and install dependencies first for caching efficiency
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code, registered models, and configuration
COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/
COPY mlflow.db ./mlflow.db
COPY params.yaml ./params.yaml

# Expose container service port 8000
EXPOSE 8000

# Start FastAPI service with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
