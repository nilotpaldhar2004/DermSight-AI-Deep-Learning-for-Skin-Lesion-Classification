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

# ── 1. LOGGING & CONFIGURATION ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("dermsight-api")

# Environment Variables
MODEL_PATH   = os.getenv("MODEL_PATH",   "best_resnet50_skin.pth") 
CLASSES_PATH = os.getenv("CLASSES_PATH", "classes.npy")
PORT         = int(os.getenv("PORT", 7860))

ml = {}

# Standard ImageNet Stats for Normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Using 448x448 as per your highest accuracy training run
transform = transforms.Compose([
    transforms.Resize((448, 448)), 
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ── 2. MODEL ARCHITECTURE ───────────────────────────────────────────────────
def build_model() -> nn.Module:
    """Reconstructs the ResNet-50 model with the custom classifier head."""
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    
    # This must match your training code exactly
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
        nn.Linear(128, 7) # 7 diagnostic classes
    )
    
    if not os.path.exists(MODEL_PATH):
        logger.error("Weight file not found: %s", MODEL_PATH)
        raise FileNotFoundError(f"Model file {MODEL_PATH} missing.")

    # Load weights to CPU (HuggingFace default)
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ── 3. LIFESPAN MANAGEMENT ──────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown logic."""
    logger.info("Initializing DermSight PRO Resources...")
    
    # Load Class Labels
    try:
        ml["classes"] = np.load(CLASSES_PATH, allow_pickle=True)
        logger.info("Labels loaded: %s", list(ml["classes"]))
    except Exception as e:
        logger.error("CRITICAL: Failed to load classes.npy: %s", e)
        ml["classes"] = None

    # Load Model
    try:
        ml["model"] = build_model()
        logger.info("ResNet-50 Model loaded successfully.")
    except Exception as e:
        logger.error("CRITICAL: Model loading failed: %s", e)
        ml["model"] = None

    yield
    ml.clear()
    logger.info("Resources released.")

# ── 4. API & ROUTING ────────────────────────────────────────────────────────
app = FastAPI(
    title="DermSight PRO API",
    version="3.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serves the UI from index.html."""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return JSONResponse(
        status_code=404, 
        content={"error": "Frontend 'index.html' not found in root directory."}
    )

@app.get("/health", tags=["System"])
async def health():
    """Checks if model and labels are loaded."""
    is_ready = ml.get("model") is not None and ml.get("classes") is not None
    return {
        "status": "ready" if is_ready else "error",
        "model_loaded": ml.get("model") is not None,
        "classes_loaded": ml.get("classes") is not None
    }

@app.post("/predict", tags=["AI Inference"])
async def predict(file: UploadFile = File(...)):
    """Accepts image and returns classification result."""
    model = ml.get("model")
    classes = ml.get("classes")

    if not model or classes is None:
        raise HTTPException(status_code=503, detail="AI Engine not initialized.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    try:
        t0 = time.perf_counter()
        
        # Read and preprocess image
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tensor = transform(image).unsqueeze(0)

        # Inference
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)

        # Build Response
        class_name = str(classes[predicted_idx.item()])
        conf_score = confidence.item() * 100
        latency = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "prediction": class_name,
            "confidence": f"{conf_score:.2f}%",
            "latency_ms": latency,
            "status": "success"
        }

    except Exception as e:
        logger.error("Prediction Error: %s", e)
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")

# ── 5. EXECUTION ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
