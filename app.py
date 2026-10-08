"""app.py — Stage 3: FastAPI inference service.

Implement /health (liveness + loaded model info) and POST /predict (multipart image upload
→ {label, prob_defect, confidence}). Load the model once at startup; log every prediction
to artifacts/predictions.log.   Run: uvicorn app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import io, json, time
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import torch, torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError

import config
from src import data_prep
from src.model import load_model

_state = {"model": None, "tf": None, "meta": {}}

logger = logging.getLogger("casting_api")
logger.setLevel(logging.INFO)

PREDICTION_LOG = Path(config.ARTIFACT_DIR) / "predictions.log"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


def _read_json(path) -> dict:
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        logger.warning("Could not read %s: %s", path, error)
        return {}


def _load():
    """Load model + eval transforms + model_meta.json once, if a checkpoint exists.

    Never raises: without a checkpoint (e.g. in CI) the API still starts, /health
    reports model_loaded=false and /predict answers 503.
    """
    _state.update({"model": None, "tf": None, "meta": {}})
    model_path = Path(config.MODEL_PATH)

    if not model_path.is_file():
        logger.warning("No model checkpoint at %s; /predict will return 503.", model_path)
        return

    try:
        model = load_model(model_path)
        model.eval()
        _state["tf"] = data_prep.get_transforms(train=False)
        _state["meta"] = _read_json(config.MODEL_META_PATH)
        _state["model"] = model
        logger.info(
            "Model loaded from %s (version %s).",
            model_path, _state["meta"].get("model_version", "unregistered"),
        )
    except Exception:
        logger.exception("Failed to load the model; /predict will return 503.")
        _state.update({"model": None, "tf": None})


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load(); yield


app = FastAPI(title="Casting Defect Detection API", version="1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Liveness + information about the loaded model."""
    meta = _state["meta"]
    metrics = _read_json(config.METRICS_PATH)
    test_metrics = {
        key: metrics[key]
        for key in ("recall_defect", "precision_defect", "f1_defect", "roc_auc", "accuracy")
        if key in metrics
    } or None

    return {
        "status": "ok",
        "model_loaded": _state["model"] is not None,
        "classes": config.CLASS_TO_IDX,
        "positive_class": config.POSITIVE_CLASS,
        "model_version": meta.get("model_version"),
        "registered_model": meta.get("registered_model"),
        "mlflow_run_id": meta.get("mlflow_run_id"),
        "test_metrics": test_metrics,
    }


def _log_prediction(record: dict) -> None:
    """Append one JSON line to artifacts/predictions.log and to the app logger."""
    logger.info("prediction %s", json.dumps(record))
    try:
        PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)
        with PREDICTION_LOG.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record) + "\n")
    except OSError as error:
        logger.warning("Could not write prediction log: %s", error)


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    """Classify one uploaded casting image."""
    started = time.perf_counter()
    model, transform = _state["model"], _state["tf"]

    # 503: the service is up but has no model to serve.
    if model is None or transform is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train a model first.")

    # 400: empty, oversized or undecodable uploads.
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File too large (limit 10 MB).")
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()  # fully decode: catches truncated/corrupt files
            image = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        logger.info("Rejected upload %r: not a valid image.", file.filename)
        raise HTTPException(status_code=400, detail="Upload must be a valid image file.")

    # Same preprocessing as validation/test: grayscale x3, resize, ImageNet normalisation.
    device = next(model.parameters()).device
    batch = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = F.softmax(model(batch), dim=1)[0].tolist()

    predicted_idx = max(range(len(probabilities)), key=probabilities.__getitem__)
    label = config.IDX_TO_CLASS[predicted_idx]
    prob_defect = float(probabilities[config.POSITIVE_IDX])
    result = {
        "label": label,
        "is_defective": predicted_idx == config.POSITIVE_IDX,
        "prob_defect": round(prob_defect, 6),
        "confidence": round(float(probabilities[predicted_idx]), 6),
    }

    _log_prediction({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "filename": file.filename,
        "model_version": _state["meta"].get("model_version"),
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        **result,
    })
    return result
