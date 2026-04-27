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

# ── 1. CONFIGURATION ─────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("dermsight-api")

MODEL_PATH   = os.getenv("MODEL_PATH",   "best_resnet50_skin.pth") 
CLASSES_PATH = os.getenv("CLASSES_PATH", "classes.npy")
PORT         = int(os.getenv("PORT", 7860))

ml = {}

# Standard ImageNet Stats
transform = transforms.Compose([
    transforms.Resize((448, 448)), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── 2. MODEL ARCHITECTURE ───────────────────────────────────────────────────
def build_model() -> nn.Module:
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    
    # Ensure these Dropout values (0.5, 0.4) match your LATEST training run!
    model.fc = nn.Sequential(
        nn.Linear(num_features, 2048),
        nn.BatchNorm1d(2048),
        nn.ReLU(),
        nn.Dropout(0.5), 
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Dropout(0.4), 
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.3), 
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 7) 
    )
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing {MODEL_PATH}")

    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ── 3. LIFESPAN ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading AI Resources...")
    try:
        ml["classes"] = np.load(CLASSES_PATH, allow_pickle=True)
        ml["model"] = build_model()
        logger.info("Resources loaded successfully.")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
    yield
    ml.clear()

# ── 4. API ──────────────────────────────────────────────────────────────────
app = FastAPI(title="DermSight PRO", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
async def serve_frontend():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "DermSight PRO API is live."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    model = ml.get("model")
    classes = ml.get("classes")

    if not model or classes is None:
        raise HTTPException(status_code=503, detail="Model not ready.")

    try:
        t0 = time.perf_counter()
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            conf, idx = torch.max(probs, 0)

        # FORCE keys to lowercase so JavaScript 'mel' always matches
        all_probabilities = {
            str(classes[i]).lower(): round(float(probs[i]) * 100, 2)
            for i in range(len(classes))
        }

        return {
            "prediction": str(classes[idx.item()]), # Keeping display name as is
            "confidence": f"{conf.item()*100:.2f}%",
            "all_probabilities": all_probabilities,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2)
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT)
