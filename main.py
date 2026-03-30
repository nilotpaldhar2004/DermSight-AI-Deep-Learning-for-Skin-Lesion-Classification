"""
DermSight PRO — Skin Lesion Classification API
================================================
FastAPI backend serving a fine-tuned ResNet-50 model for real-time
dermoscopic image classification across 7 ISIC diagnostic categories.

Model     : ResNet-50 (fine-tuned on HAM10000)
Dataset   : HAM10000 — 10,015 dermoscopic images
Precision : FP16 (Half Precision for Render free-tier memory efficiency)

Author    : Nilotpal Dhar
Version   : 2.0.0
"""

import os
import io
import logging
import time
from contextlib import asynccontextmanager

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

# ---------------------------------------------------------------------------
# Memory optimisation — critical for Render free tier (512MB RAM limit)
# Limits PyTorch to 1 CPU thread to prevent OOM kills
# ---------------------------------------------------------------------------
torch.set_num_threads(1)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dermsight-api")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH   = os.getenv("MODEL_PATH",   "skin_lesion_resnet50_best.pth")
CLASSES_PATH = os.getenv("CLASSES_PATH", "classes.npy")
PORT         = int(os.getenv("PORT", 8000))

# Shared state — populated at startup via lifespan
ml = {}

# ---------------------------------------------------------------------------
# ImageNet normalisation constants (required for ResNet-50 inference)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
def build_model() -> nn.Module:
    """
    Reconstruct the ResNet-50 classifier head and load the saved weights.
    The model is converted to FP16 (half precision) after loading to match
    the compressed .pth checkpoint and stay within Render's memory limit.
    """
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 7),
    )
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.half()   # FP16 — matches compressed checkpoint
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Lifespan — load model and class labels once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading class labels from '%s' ...", CLASSES_PATH)
    try:
        ml["classes"] = np.load(CLASSES_PATH, allow_pickle=True)
        logger.info("Classes loaded: %s", list(ml["classes"]))
    except FileNotFoundError:
        logger.error("Class file '%s' not found.", CLASSES_PATH)
        ml["classes"] = None

    logger.info("Loading ResNet-50 model from '%s' ...", MODEL_PATH)
    try:
        ml["model"] = build_model()
        logger.info("Model loaded successfully (FP16, CPU).")
    except FileNotFoundError:
        logger.error("Model file '%s' not found.", MODEL_PATH)
        ml["model"] = None
    except Exception as exc:
        logger.exception("Failed to load model: %s", exc)
        ml["model"] = None

    yield  # server is live

    ml.clear()
    logger.info("Server shutting down — resources released.")


# ---------------------------------------------------------------------------
# App & middleware
# ---------------------------------------------------------------------------
app = FastAPI(
    title="DermSight PRO API",
    description=(
        "Real-time skin lesion classification using a fine-tuned ResNet-50 "
        "model trained on the HAM10000 dataset. Classifies dermoscopic images "
        "into 7 ISIC diagnostic categories."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    # "*" is appropriate for a portfolio/demo project.
    # For production, replace with your GitHub Pages URL.
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request timing middleware
@app.middleware("http")
async def timing_middleware(request, call_next):
    start    = time.perf_counter()
    response = await call_next(request)
    elapsed  = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed:.2f}"
    return response


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the dashboard HTML when the server URL is opened in a browser."""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "DermSight PRO API v2.0.0 — POST image to /predict"}


@app.get("/ping", include_in_schema=False)
async def ping():
    """
    Lightweight keep-alive endpoint.
    Point a free uptime monitor (UptimeRobot) here every 10 minutes
    to prevent Render free-tier cold-start delays.
    """
    return {"pong": True}


@app.get("/health", tags=["System"])
async def health():
    """Liveness and readiness probe. Returns 503 if model failed to load."""
    model_ready   = ml.get("model") is not None
    classes_ready = ml.get("classes") is not None
    ready = model_ready and classes_ready
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status":        "ok" if ready else "degraded",
            "model_loaded":  model_ready,
            "classes_loaded": classes_ready,
            "version":       "2.0.0",
        },
    )


@app.post("/predict", tags=["Inference"])
async def predict(file: UploadFile = File(...)):
    """
    Accept a dermoscopic image and return classification results.

    Returns:
    - **prediction** — top predicted class code (e.g. 'mel', 'nv')
    - **confidence** — confidence score for the top class (0–100)
    - **all_probabilities** — probability for every class as a percentage
    - **latency_ms** — server-side inference time
    """
    model   = ml.get("model")
    classes = ml.get("classes")

    if model is None or classes is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not ready. Check server logs.",
        )

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Upload a JPG, PNG, or TIFF image.",
        )

    try:
        t0        = time.perf_counter()
        img_bytes = await file.read()
        image     = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Preprocess: resize, normalise, convert to FP16 to match the model
        tensor = transform(image).unsqueeze(0).half()

        with torch.no_grad():
            outputs      = model(tensor)
            probabilities = torch.nn.functional.softmax(outputs[0].float(), dim=0)
            confidence, predicted = torch.max(probabilities, 0)

        top_class   = str(classes[predicted.item()])
        top_conf    = confidence.item() * 100
        all_probs   = {
            str(classes[i]): round(float(probabilities[i]) * 100, 4)
            for i in range(len(classes))
        }
        latency_ms  = round((time.perf_counter() - t0) * 1000, 2)

        logger.info(
            "Prediction | class=%s | confidence=%.2f%% | latency=%.2fms",
            top_class, top_conf, latency_ms,
        )

        return {
            "prediction":       top_class,
            "confidence":       f"{top_conf:.2f}",
            "all_probabilities": all_probs,
            "latency_ms":       latency_ms,
        }

    except Exception as exc:
        logger.exception("Inference failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting DermSight PRO API on port %d", PORT)
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
