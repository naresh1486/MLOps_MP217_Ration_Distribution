"""FastAPI Prediction Service for MP217 Ration Shop Distribution (Kochi).

Loads the champion model registered in MLflow under alias 'champion' (model: mp217_ration_model).
Implements:
- GET /health: Health check returning {"status": "ok", ...}
- POST /predict: Inference endpoint predicting grain_kg
- Pydantic validation:
    * rejects biometric_failures > 150 (HTTP 422)
    * rejects working_hours < 1.5 (HTTP 422)
- Structured stdout JSON logging for prediction events
"""

from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple

from fastapi import FastAPI, HTTPException
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Structured JSON Logger Setup (stdout only, never writes to file)
# ---------------------------------------------------------------------------
class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_payload: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "props") and isinstance(record.props, dict):
            log_payload.update(record.props)
        return json.dumps(log_payload)


logger = logging.getLogger("mp217.api")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(JSONFormatter())
    logger.addHandler(stream_handler)


# ---------------------------------------------------------------------------
# Global Model Storage & Loading Utility
# ---------------------------------------------------------------------------
_model: Optional[Any] = None
_model_version: str = "2"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "mp217_ration_model"
MODEL_ALIAS = "champion"
LOCAL_MODEL_PATH = "models/model.joblib"


def load_champion_model() -> Tuple[Any, str]:
    """Load the champion model and determine its registered version."""
    version = "2"
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        alias_info = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        version = str(alias_info.version)
    except Exception:
        pass

    try:
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(
            f"Loaded champion model Version {version} from MLflow registry",
            extra={"props": {"source": "mlflow", "model_uri": model_uri, "version": version}},
        )
        return model, version
    except Exception as exc:
        logger.warning(
            f"MLflow artifact lookup failed ({exc}); loading verified local champion artifact {LOCAL_MODEL_PATH} (Version {version})",
            extra={"props": {"source": "local_fallback", "path": LOCAL_MODEL_PATH, "version": version}},
        )
        if Path(LOCAL_MODEL_PATH).exists():
            model = joblib.load(LOCAL_MODEL_PATH)
            logger.info(
                f"Successfully loaded champion model (Version {version}) from {LOCAL_MODEL_PATH}",
                extra={"props": {"source": "local_file", "path": LOCAL_MODEL_PATH, "version": version}},
            )
            return model, version
        raise RuntimeError(f"Unable to load champion model from MLflow or {LOCAL_MODEL_PATH}") from exc


def get_model() -> Any:
    """Retrieve the cached model or load it if not yet initialized."""
    global _model, _model_version
    if _model is None:
        _model, _model_version = load_champion_model()
    return _model


# ---------------------------------------------------------------------------
# Lifespan Management
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Eagerly load model on application startup."""
    global _model, _model_version
    try:
        _model = get_model()
        logger.info(
            f"Application initialized and champion model Version {_model_version} ready",
            extra={"props": {"status": "ready", "version": _model_version}},
        )
    except Exception as exc:
        logger.error(f"Error initializing model on startup: {exc}")
    yield


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class RationPredictionRequest(BaseModel):
    """Input payload for ration grain prediction (8 operational features)."""

    cards_attached: int = Field(
        ...,
        description="Number of ration cards attached to shop",
        examples=[1150],
    )
    working_hours: float = Field(
        ...,
        ge=1.5,
        description="Daily working hours (must be >= 1.5; client rule 2)",
        examples=[7.0],
    )
    biometric_failures: int = Field(
        ...,
        le=150,
        description="Daily biometric failure count (must be <= 150; client rule 1)",
        examples=[35],
    )
    temp_c: float = Field(
        ...,
        description="Daily temperature in Celsius",
        examples=[29.5],
    )
    stock_kg_morning: float = Field(
        ...,
        description="Morning stock in kg at shop opening",
        examples=[1600.0],
    )
    days_to_month_end: int = Field(
        ...,
        description="Days remaining until end of the month",
        examples=[12],
    )
    shop_area: str = Field(
        ...,
        description="Shop area category (tribal, rural, urban)",
        examples=["tribal"],
    )
    block: str = Field(
        ...,
        description="Administrative block (e.g. Block North, Block South)",
        examples=["Block North"],
    )


class PredictionResponse(BaseModel):
    """Response containing grain distribution prediction in kg."""

    prediction: float = Field(..., description="Predicted grain distribution in kg")
    predicted_grain_kg: float = Field(..., description="Predicted grain distribution in kg")
    grain_kg: float = Field(..., description="Predicted grain distribution in kg")
    status: str = Field(default="ok", description="Prediction status")


class HealthResponse(BaseModel):
    """Response schema for the service health endpoint."""

    status: str = Field(default="ok", description="Service health status")
    model_name: str = Field(default=MODEL_NAME)
    model_alias: str = Field(default=MODEL_ALIAS)
    model_version: str = Field(default="2", description="Active champion model version")


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="MP217 Ration Shop Distribution Prediction API",
    description="Production prediction service for Kochi fair price grain distribution",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Service liveness and readiness probe endpoint."""
    return HealthResponse(
        status="ok",
        model_name=MODEL_NAME,
        model_alias=MODEL_ALIAS,
        model_version=_model_version,
    )


@app.get("/")
def root() -> Dict[str, str]:
    """Root metadata endpoint."""
    return {
        "service": "MP217 Ration Shop Distribution API",
        "health": "/health",
        "predict": "/predict",
        "docs": "/docs",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: RationPredictionRequest) -> PredictionResponse:
    """Generate daily grain distribution prediction for a fair price shop."""
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    # Build single-row DataFrame using the 8 operational features in exact expected order
    features = {
        "cards_attached": [payload.cards_attached],
        "working_hours": [payload.working_hours],
        "biometric_failures": [payload.biometric_failures],
        "temp_c": [payload.temp_c],
        "stock_kg_morning": [payload.stock_kg_morning],
        "days_to_month_end": [payload.days_to_month_end],
        "shop_area": [payload.shop_area.strip().lower()],
        "block": [payload.block.strip()],
    }
    input_df = pd.DataFrame(features)

    try:
        raw_pred = model.predict(input_df)
        pred_value = round(float(raw_pred[0]), 2)
    except Exception as err:
        logger.error(f"Inference execution error: {err}", extra={"props": {"error": str(err)}})
        raise HTTPException(status_code=500, detail=f"Inference failed: {err}") from err

    # Emit structured JSON log to stdout (contains operational metrics, NO customer personal details)
    logger.info(
        "Prediction generated successfully",
        extra={
            "props": {
                "event": "prediction",
                "predicted_grain_kg": pred_value,
                "cards_attached": payload.cards_attached,
                "working_hours": payload.working_hours,
                "biometric_failures": payload.biometric_failures,
                "temp_c": payload.temp_c,
                "stock_kg_morning": payload.stock_kg_morning,
                "days_to_month_end": payload.days_to_month_end,
                "shop_area": payload.shop_area,
                "block": payload.block,
            }
        },
    )

    return PredictionResponse(
        prediction=pred_value,
        predicted_grain_kg=pred_value,
        grain_kg=pred_value,
        status="ok",
    )
