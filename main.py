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

# --- RENDER MEMORY OPTIMIZATION ---
# Limits PyTorch to 1 thread to stay under the 512MB RAM limit
torch.set_num_threads(1)

app = FastAPI()

# 1. ENABLE CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load labels
class_names = np.load("classes.npy", allow_pickle=True)

# 3. Rebuild Architecture with FP16 Support
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
    
    # Load compressed weights (52MB version)
    state_dict = torch.load("skin_lesion_resnet50_best.pth", map_location='cpu')
    model.load_state_dict(state_dict)
    
    # IMPORTANT: Convert model to Half Precision to match your compressed file
    model = model.half() 
    model.eval()
    return model

model = get_model()

# 4. Preprocessing
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
    
    # 5. CONVERT INPUT TO HALF PRECISION
    # We must call .half() here so the image matches the model's data type
    image = transform(image).unsqueeze(0).half() 
    
    with torch.no_grad():
        outputs = model(image)
        # Convert outputs back to float for stable Softmax calculation
        probabilities = torch.nn.functional.softmax(outputs[0].float(), dim=0)
        confidence, predicted = torch.max(probabilities, 0)
        
    all_probs = {class_names[i]: float(probabilities[i] * 100) for i in range(len(class_names))}
        
    return {
        "prediction": str(class_names[predicted.item()]),
        "confidence": f"{confidence.item() * 100:.2f}",
        "all_probabilities": all_probs
    }

@app.get("/", response_class=HTMLResponse)
async def home():
    if os.path.exists("index.html"):
        with open("index.html", "r") as f:
            return f.read()
    return "DermSight API is Online (Compressed Mode)."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
