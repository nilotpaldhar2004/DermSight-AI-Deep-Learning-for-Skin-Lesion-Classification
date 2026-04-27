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

# ── CONFIGURATION ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("dermsight-api")

# FIXED: Aligned with your actual saved filename
MODEL_PATH   = os.getenv("MODEL_PATH",   "best_resnet50_skin.pth") 
CLASSES_PATH = os.getenv("CLASSES_PATH", "classes.npy")
PORT         = int(os.getenv("PORT", 7860))

ml = {}

# Standard ImageNet Stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# FIXED: Ensure this matches your training resolution (448 or 224)
transform = transforms.Compose([
    transforms.Resize((448, 448)), 
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ── MODEL BUILDER ────────────────────────────────────────────────────────────
def build_model() -> nn.Module:
    # Use weights=None because we are loading local weights
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    
    # EXACT match of your custom training head
    model.fc = nn.Sequential(
        nn.Linear(num_features, 2048),
        nn.BatchNorm1d(2048),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        # ... (rest of your architecture)
        nn.Dropout(0.3),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 7)
    )
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model weight file not found at {MODEL_PATH}")

    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    
    # FIXED: Removed .half() to ensure CPU compatibility on HuggingFace
    model.eval()
    return model

# ── LIFESPAN (Resource Management) ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: Loading model and labels...")
    
    # Load Classes
    try:
        ml["classes"] = np.load(CLASSES_PATH, allow_pickle=True)
        logger.info("Successfully loaded %d classes", len(ml["classes"]))
    except Exception as e:
        logger.error("Failed to load classes: %s", e)
        ml["classes"] = None

    # Load Model
    try:
        ml["model"] = build_model()
        logger.info("Successfully loaded model from %s", MODEL_PATH)
    except Exception as e:
        logger.error("Model loading failed: %s", e)
        ml["model"] = None

    yield
    ml.clear()

# ── API SETUP ────────────────────────────────────────────────────────────────
app = FastAPI(title="DermSight PRO API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    model = ml.get("model")
    classes = ml.get("classes")

    if not model or classes is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server.")

    try:
        t0 = time.perf_counter()
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # FIXED: Removed .half() for CPU stability
        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(tensor)
            # Ensure probabilities are calculated in float32
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)

        top_class = str(classes[predicted.item()])
        top_conf = confidence.item() * 100
        
        return {
            "prediction": top_class,
            "confidence": f"{top_conf:.2f}%",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2)
        }
    except Exception as e:
        logger.error("Inference Error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT)
