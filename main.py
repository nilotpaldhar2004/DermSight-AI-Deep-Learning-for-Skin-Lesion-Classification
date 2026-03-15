import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from torchvision import models, transforms
from PIL import Image
import io
import numpy as np
import uvicorn
import os

app = FastAPI()

# 1. ENABLE CORS - Essential so your GitHub Pages site can talk to Render
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows any origin to access the API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load your labels (Ensure classes.npy is in the same folder)
class_names = np.load("classes.npy", allow_pickle=True)

# 3. Rebuild your specific ResNet50 architecture
def get_model():
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
        nn.Linear(128, 7)
    )
    # Load your best weights (Ensure the .pth file is in the same folder)
    # Using map_location='cpu' so it works on Render's free tier (no GPU)
    model.load_state_dict(torch.load("skin_lesion_resnet50_best.pth", map_location='cpu'))
    model.eval()
    return model

model = get_model()

# 4. Image Preprocessing (Standard ResNet normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)
        
    # Build a dictionary of all class probabilities for the UI chart
    all_probs = {class_names[i]: float(probabilities[i] * 100) for i in range(len(class_names))}
        
    return {
        "prediction": str(class_names[predicted.item()]),
        "confidence": f"{confidence.item() * 100:.2f}",
        "all_probabilities": all_probs
    }

@app.get("/", response_class=HTMLResponse)
async def home():
    # If index.html is in the same folder, it will serve it
    if os.path.exists("index.html"):
        with open("index.html", "r") as f:
            return f.read()
    return "DermSight API is Running. Use the /predict endpoint for inference."

# 5. START COMMAND FOR RENDER
if __name__ == "__main__":
    # Render provides the port via environment variables
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
