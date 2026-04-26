---
title: Deep Learning For Skin Lesion Classification
emoji: 🔬
colorFrom: teal
colorTo: blue
sdk: docker
app_file: main.py
pinned: false
license: mit
---

# 🔬 DermSight PRO — Deep Residual Learning for Skin Lesion Classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)
![ResNet-50](https://img.shields.io/badge/Model-ResNet--50-blue?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-HAM10000+ISIC2019-orange?style=flat-square)
![Accuracy](https://img.shields.io/badge/Val_Accuracy-82.6%25-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=flat-square)

**A production-grade computer vision application that classifies dermoscopic skin images into 7 ISIC diagnostic categories using a fine-tuned ResNet-50 architecture.**

[🚀 Live Demo](https://nilotpaldhar2004-deep-learning-for-skin-lesion-classification.hf.space) · [📖 API Docs](https://nilotpaldhar2004-deep-learning-for-skin-lesion-classification.hf.space/docs) · [💻 GitHub](https://github.com/nilotpaldhar2004/DermSight-AI-Deep-Learning-for-Skin-Lesion-Classification)

</div>

---

## 📸 What It Does

Upload any dermoscopic image and the model returns an instant probability distribution across all 7 ISIC diagnostic classes — complete with a confidence dial, color-coded risk classification (Malignant / Pre-cancerous / Benign), clinical descriptions, and a downloadable analysis report.

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Validation Accuracy | **82.6%** (held-out val set) |
| Best Epoch | 20 / 20 |
| Val Loss | 0.827 |
| Training Images | 25,331 (combined) |
| Classes | 7 ISIC categories |

> Accuracy measured on a **held-out validation set** — not the training set. This is the honest number.

---

## ✨ Features

- **Real-Time Inference** — ResNet-50 classification in under 500ms per image
- **7-Class ISIC Coverage** — MEL, BCC, AKIEC, BKL, NV, DF, VASC
- **FP16 Inference** — Half-precision model fits within free-tier RAM limits
- **Confidence Dial** — Animated SVG arc gauge with risk-level color coding
- **Clinical Descriptions** — Per-class pathology notes with urgency classification
- **Downloadable Report** — Structured `.txt` report with all probabilities and clinical notes
- **Responsive UI** — Works on mobile, tablet, and desktop

---

## 🧠 Neural Architecture

| Component | Detail |
|---|---|
| Backbone | ResNet-50 (ImageNet pre-trained, IMAGENET1K_V2) |
| Fine-tuned layers | Layer 3 + Layer 4 (not just the head) |
| Classifier Head | Linear(2048→1024) → BatchNorm1d → ReLU → Dropout(0.4) → Linear(1024→512) → ReLU → Dropout(0.3) → Linear(512→128) → ReLU → Linear(128→7) |
| Input Resolution | 224 × 224 px |
| Normalisation | μ = [0.485, 0.456, 0.406]  σ = [0.229, 0.224, 0.225] (ImageNet) |
| Precision | FP16 (Half Precision) |
| Loss Function | Weighted Cross-Entropy (class weights from sklearn.utils.class_weight) |
| Optimizer | AdamW (backbone lr=1e-5, head lr=1e-4, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR (T_max=20) |

### Why ResNet-50?

Traditional deep CNNs degrade with depth due to the **vanishing gradient problem**. ResNet-50 solves this with **residual (skip) connections** that allow gradients to propagate directly to earlier layers — making it ideal for learning fine-grained dermoscopic patterns.

**Transfer Learning:** ImageNet weights provide strong low-level feature priors (edges, textures) which are repurposed for lesion morphology.

---

## 📦 Dataset

| Dataset | Images | Classes |
|---|---|---|
| HAM10000 (ISIC Archive) | 10,015 | 7 |
| ISIC 2019 (unique only) | 15,316 | 7 |
| **Combined Total** | **25,331** | **7** |

**Class imbalance fixes applied:**
- WeightedRandomSampler — every batch sees all 7 classes equally
- Weighted CrossEntropyLoss — minority class losses count proportionally more
- Strong augmentation — RandomRotation(30), ColorJitter, RandomAffine, RandomPerspective

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
| Training | HAM10000 + ISIC 2019 combined (25,331 images), Kaggle GPU T4 |
| Backend API | FastAPI, Uvicorn, Python-Multipart |
| Image Processing | Pillow, NumPy |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Deployment | HuggingFace Spaces (Docker SDK) |
| CI/CD | GitHub Actions — auto-deploy on push to main |

---

## 📂 Project Structure

```
DermSight-AI-Deep-Learning-for-Skin-Lesion-Classification/
│
├── main.py                          # FastAPI backend (port 7860 for HF Spaces)
├── index.html                       # Frontend dashboard
├── Dockerfile                       # HuggingFace Spaces deployment
├── requirements.txt                 # Python dependencies (CPU-only torch)
├── LICENSE                          # MIT License
├── .gitignore                       # Git ignore rules
├── README.md                        # This file
│
├── .github/workflows/
│   └── deploy.yml                   # Auto-deploy to HF Spaces on git push
│
├── Skin Lesion Classifier [USE ResNet-50] & Analysis.ipynb  # Training notebook
│
└── (uploaded directly to HF Space — not in git)
    ├── skin_lesion_resnet50_best.pth    # Trained model weights (~52MB FP16)
    └── classes.npy                      # Class label array
```

---

## 🚀 Run Locally

```bash
git clone https://github.com/nilotpaldhar2004/DermSight-AI-Deep-Learning-for-Skin-Lesion-Classification.git
cd DermSight-AI-Deep-Learning-for-Skin-Lesion-Classification

pip install -r requirements.txt

# Add model files to project root:
# skin_lesion_resnet50_best.pth
# classes.npy

python main.py
# Open http://localhost:7860
```

---

## 📡 API Reference

### `GET /health`
```json
{ "status": "ok", "model_loaded": true, "classes_loaded": true, "version": "3.0.0" }
```

### `POST /predict`
```bash
curl -X POST https://nilotpaldhar2004-deep-learning-for-skin-lesion-classification.hf.space/predict \
  -F "file=@your_image.jpg"
```

**Response:**
```json
{
  "prediction": "mel",
  "confidence": "84.27",
  "all_probabilities": {
    "mel": 84.27, "nv": 8.12, "bkl": 3.45,
    "bcc": 2.11, "akiec": 1.20, "vasc": 0.55, "df": 0.30
  },
  "latency_ms": 312.5
}
```

Full Swagger docs: `/docs`

---

## ⚕️ Medical Disclaimer

This application is developed for **educational and research purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified dermatologist for assessment of skin lesions.

---

## 👤 Author

**Nilotpal Dhar** — CS student, AI/ML enthusiast

[![GitHub](https://img.shields.io/badge/GitHub-nilotpaldhar2004-black?style=flat&logo=github)](https://github.com/nilotpaldhar2004)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-nilotpaldhar2004-orange?style=flat)](https://huggingface.co/nilotpaldhar2004)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-nilotpal--dhar-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/nilotpal-dhar-24b304294/)

---

<div align="center">
  <sub>Built with PyTorch, FastAPI, and ResNet-50 · Dataset: HAM10000 + ISIC 2019 · Deployed on HuggingFace Spaces via Docker + GitHub Actions CI/CD</sub>
</div>
