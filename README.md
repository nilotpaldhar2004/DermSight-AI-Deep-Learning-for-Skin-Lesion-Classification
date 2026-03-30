# 🔬 DermSight PRO — Deep Residual Learning for Skin Lesion Classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)
![ResNet-50](https://img.shields.io/badge/Model-ResNet--50-blue?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-HAM10000-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=flat-square)

**A production-grade computer vision application that classifies dermoscopic skin images into 7 ISIC diagnostic categories using a fine-tuned ResNet-50 architecture.**

[Live Demo](https://nilotpaldhar2004.github.io/DermSight-AI-Deep-Learning-for-Skin-Lesion-Classification/) · [API Docs](https://dermsight-ai-deep-learning-for-skin.onrender.com/docs) · [Report an Issue](https://github.com/nilotpaldhar2004/DermSight-AI-Deep-Learning-for-Skin-Lesion-Classification/issues)

</div>

---

## 📸 What it does

Upload any dermoscopic image and the model returns an instant probability distribution across all 7 ISIC diagnostic classes — complete with a confidence dial, color-coded risk classification (Malignant / Pre-cancerous / Benign), clinical descriptions, and a downloadable analysis report.

---

## ✨ Features

- **Real-Time Inference** — ResNet-50 classification in under 500ms per image
- **7-Class ISIC Coverage** — MEL, BCC, AKIEC, BKL, NV, DF, VASC
- **FP16 Inference** — Half-precision model fits within Render's 512MB free-tier RAM
- **Confidence Dial** — Animated SVG arc gauge with risk-level color coding
- **Clinical Descriptions** — Per-class pathology notes with urgency classification
- **Downloadable Report** — Structured `.txt` report with all probabilities and clinical notes
- **Keep-Alive Endpoint** — `/ping` prevents Render cold-start delays
- **Responsive UI** — Works cleanly on mobile, tablet, and desktop

---

## 🧠 Neural Architecture

| Component | Detail |
|---|---|
| Backbone | ResNet-50 (ImageNet pre-trained) |
| Classifier Head | Linear(2048→1024) → ReLU → Dropout(0.2) → Linear(1024→512) → ReLU → Dropout(0.1) → Linear(512→128) → ReLU → Linear(128→7) |
| Input Resolution | 224 × 224 px |
| Normalisation | μ = [0.485, 0.456, 0.406] · σ = [0.229, 0.224, 0.225] (ImageNet) |
| Precision | FP16 (Half Precision) |
| Loss Function | Weighted Cross-Entropy (compensates for 10:1 class imbalance) |

### Why ResNet-50?

Traditional deep CNNs degrade with depth due to the **vanishing gradient problem**. ResNet-50 solves this with **residual (skip) connections** that allow gradients to propagate directly to earlier layers. This enables training of much deeper networks without performance degradation — making it ideal for learning fine-grained dermoscopic patterns.

**Transfer Learning:** ImageNet weights provide strong low-level feature priors (edges, textures) which are repurposed for lesion morphology. Only the classifier head was trained from scratch; the convolutional backbone was fine-tuned at a reduced learning rate.

---

## 🔬 Diagnostic Categories

| Code | Full Name | Risk Level |
|---|---|---|
| **MEL** | Melanoma | 🔴 Malignant |
| **BCC** | Basal Cell Carcinoma | 🔴 Malignant |
| **AKIEC** | Actinic Keratoses / Intraepithelial Carcinoma | 🟡 Pre-cancerous |
| **BKL** | Benign Keratosis | 🟢 Benign |
| **NV** | Melanocytic Nevi | 🟢 Benign |
| **DF** | Dermatofibroma | 🟢 Benign |
| **VASC** | Vascular Lesions | 🟢 Benign |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | PyTorch 2.2, torchvision, ResNet-50 |
| Training | HAM10000 Dataset (10,015 dermoscopic images) |
| Backend API | FastAPI, Uvicorn, Python-Multipart |
| Image Processing | Pillow, NumPy |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Deployment | Render (API) + GitHub Pages (Frontend) |

---

## 📂 Project Structure

```
DermSight-AI-Deep-Learning-for-Skin-Lesion-Classification/
│
├── main.py                                              # FastAPI backend server
├── index.html                                           # Frontend dashboard
├── Skin Lesion Classifier [USE ResNet-50] & Analysis.ipynb  # Training notebook
├── requirements.txt                                     # Python dependencies
├── LICENSE.txt                                          # MIT License
├── .gitignore                                           # Git ignore rules
├── README.md                                            # Project documentation
│
└── (gitignored — not committed to repo)
    ├── skin_lesion_resnet50_best.pth                    # Trained model weights (~52MB FP16)
    └── classes.npy                                      # Class label array
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- ~1GB disk space for PyTorch CPU install

### 1. Clone the repository

```bash
git clone https://github.com/nilotpaldhar2004/DermSight-AI-Deep-Learning-for-Skin-Lesion-Classification.git
cd DermSight-AI-Deep-Learning-for-Skin-Lesion-Classification
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add model files

Place these two files in the project root (same directory as `main.py`):

```
skin_lesion_resnet50_best.pth   ← trained model weights
classes.npy                     ← class label array
```

> **Don't have the model?** Run the Jupyter notebook `Skin Lesion Classifier [USE ResNet-50] & Analysis.ipynb` from start to finish — it trains and saves both files automatically.

### 4. Start the server

```bash
python main.py
```

Open `http://localhost:8000` — FastAPI serves `index.html` directly.

---

## 🌐 Deployment

| Component | Host | URL |
|---|---|---|
| Frontend (`index.html`) | GitHub Pages | `https://nilotpaldhar2004.github.io/DermSight-AI-Deep-Learning-for-Skin-Lesion-Classification/` |
| Backend (`main.py`) | Render | `https://dermsight-ai-deep-learning-for-skin.onrender.com` |

### Deploy to Render

> ⚠️ **Critical:** The `requirements.txt` uses `--extra-index-url https://download.pytorch.org/whl/cpu` to install the CPU-only PyTorch build. **Do not remove this line.** The default PyPI torch package includes CUDA and is ~2GB — it will exceed Render's free-tier limits and fail to deploy.

1. Push your code to GitHub (model `.pth` and `classes.npy` are gitignored — upload them separately)
2. In Render: **New → Web Service → connect your GitHub repo**
3. **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. **Environment:** Python 3
5. For the model files, use Render **Disk** (persistent storage) or upload them as part of a Dockerfile

> **Keep-alive tip:** Create a free UptimeRobot monitor pointing at `https://your-app.onrender.com/ping` every 10 minutes to prevent cold-start delays on the free tier.

### Deploy frontend to GitHub Pages

1. **Settings → Pages → Source → main branch → / (root)**
2. GitHub Pages will serve `index.html` within ~60 seconds

---

## 📡 API Reference

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "classes_loaded": true,
  "version": "2.0.0"
}
```

### `GET /ping`

```json
{ "pong": true }
```

### `POST /predict`

Accepts a `multipart/form-data` request with an image file in the `file` field.

**cURL example:**
```bash
curl -X POST https://dermsight-ai-deep-learning-for-skin.onrender.com/predict \
  -F "file=@your_image.jpg"
```

**Response:**
```json
{
  "prediction": "mel",
  "confidence": "84.27",
  "all_probabilities": {
    "mel":   84.27,
    "nv":     8.12,
    "bkl":    3.45,
    "bcc":    2.11,
    "akiec":  1.20,
    "vasc":   0.55,
    "df":     0.30
  },
  "latency_ms": 312.5
}
```

Full interactive documentation is available at `/docs` (Swagger UI) when the server is running.

---

## ⚕️ Medical Disclaimer

This application is developed for **educational and research purposes only**. It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. The model's predictions should not be used as the basis for any clinical decision. Always consult a qualified dermatologist for the assessment of skin lesions.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE.txt](LICENSE.txt) for details.

---

## 👤 Author

**Nilotpal Dhar** · [@nilotpaldhar2004](https://github.com/nilotpaldhar2004) · March 2026

---

<div align="center">
  <sub>Built with PyTorch, FastAPI, and ResNet-50 · Dataset: HAM10000 (ISIC Archive) · Deployed on Render + GitHub Pages</sub>
</div>
