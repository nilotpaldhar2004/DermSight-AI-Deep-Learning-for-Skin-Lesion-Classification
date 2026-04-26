
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

torch.set_num_threads(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dermsight-api")


MODEL_PATH   = os.getenv("MODEL_PATH",   "skin_lesion_resnet50_best.pth")
CLASSES_PATH = os.getenv("CLASSES_PATH", "classes.npy")
PORT         = int(os.getenv("PORT", 7860))   # 7860 for HuggingFace Spaces

ml = {}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def build_model() -> nn.Module:
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 2048),
        nn.BatchNorm1d(2048),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 7)
    )
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model = model.half()   # FP16
    model.eval()
    return model


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

    yield

    ml.clear()
    logger.info("Server shutting down — resources released.")


app = FastAPI(
    title="DermSight PRO API",
    description=(
        "Real-time skin lesion classification using a fine-tuned ResNet-50 "
        "model trained on HAM10000 + ISIC 2019 combined dataset (25,331 images). "
        "Classifies dermoscopic images into 7 ISIC diagnostic categories."
    ),
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def timing_middleware(request, call_next):
    start    = time.perf_counter()
    response = await call_next(request)
    elapsed  = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed:.2f}"
    return response


@app.get("/", include_in_schema=False)
async def serve_frontend():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "DermSight PRO API v3.0.0 — POST image to /predict"}


@app.get("/ping", include_in_schema=False)
async def ping():
    return {"pong": True}


@app.get("/health", tags=["System"])
async def health():
    model_ready   = ml.get("model") is not None
    classes_ready = ml.get("classes") is not None
    ready = model_ready and classes_ready
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status":         "ok" if ready else "degraded",
            "model_loaded":   model_ready,
            "classes_loaded": classes_ready,
            "version":        "3.0.0",
        },
    )


@app.post("/predict", tags=["Inference"])
async def predict(file: UploadFile = File(...)):
    
    model   = ml.get("model")
    classes = ml.get("classes")

    if model is None or classes is None:
        raise HTTPException(status_code=503, detail="Model not ready. Check server logs.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Upload JPG, PNG, or TIFF.",
        )

    try:
        t0        = time.perf_counter()
        img_bytes = await file.read()
        image     = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tensor    = transform(image).unsqueeze(0).half()

        with torch.no_grad():
            outputs       = model(tensor)
            probabilities = torch.nn.functional.softmax(outputs[0].float(), dim=0)
            confidence, predicted = torch.max(probabilities, 0)

        top_class  = str(classes[predicted.item()])
        top_conf   = confidence.item() * 100
        all_probs  = {
            str(classes[i]): round(float(probabilities[i]) * 100, 4)
            for i in range(len(classes))
        }
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        logger.info(
            "Prediction | class=%s | confidence=%.2f%% | latency=%.2fms",
            top_class, top_conf, latency_ms,
        )

        return {
            "prediction":        top_class,
            "confidence":        f"{top_conf:.2f}",
            "all_probabilities": all_probs,
            "latency_ms":        latency_ms,
        }

    except Exception as exc:
        logger.exception("Inference failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(exc)}")


if __name__ == "__main__":
    logger.info("Starting DermSight PRO API on port %d", PORT)
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
