import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from torchvision import models, transforms
from PIL import Image
import io
import numpy as np

app = FastAPI()

# 1. Load your labels
class_names = np.load("classes.npy", allow_pickle=True)

# 2. Rebuild your specific architecture
def get_model():
    model = models.resnet50(weights=None) 
    num_features = model.fc.in_features # 2048
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
    # Load your best weights
    model.load_state_dict(torch.load("skin_lesion_resnet50_best.pth", map_location='cpu'))
    model.eval()
    return model

model = get_model()

# 3. Test transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)
        
    return {
        "prediction": str(class_names[predicted.item()]),
        "confidence": f"{confidence.item() * 100:.2f}%"
    }

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("index.html", "r") as f:
        return f.read()
