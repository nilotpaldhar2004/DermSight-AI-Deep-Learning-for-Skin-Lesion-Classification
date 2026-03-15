# DermSight PRO: Neural Image Classifier & Skin Lesion Analysis Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)

DermSight PRO is an end-to-end medical computer vision application designed to assist in the classification of skin lesions using Deep Residual Learning. By leveraging a fine-tuned ResNet-50 architecture, the system provides real-time inference across seven distinct diagnostic categories.

## 🧠 The Neural Architecture

The engine is powered by a **ResNet-50** backbone, pre-trained on ImageNet and fine-tuned on the **HAM10000** dataset (Human Against Machine). 

### Why ResNet-50?
Traditional deep networks suffer from the "vanishing gradient" problem. ResNet-50 solves this using **Residual Blocks** and **Skip Connections**, allowing the model to learn identity mappings and effectively train deeper layers without performance degradation.



### Technical Specifications:
* **Input Layer:** 224x224 RGB Dermoscopic Images.
* **Optimization:** Stochastic Gradient Descent (SGD) with Momentum.
* **Loss Function:** Cross-Entropy Loss (weighted to handle class imbalance).
* **Accuracy:** Calibrated for high sensitivity in detecting **Melanoma (MEL)** and **Basal Cell Carcinoma (BCC)**.

---

## 🔬 Dataset & Classification
The model is trained on the **HAM10000** dataset, classifying lesions into:
1.  **akiec:** Actinic keratoses and intraepithelial carcinoma
2.  **bcc:** Basal cell carcinoma
3.  **bkl:** Benign keratosis-like lesions
4.  **df:** Dermatofibroma
5.  **mel:** Melanoma
6.  **nv:** Melanocytic nevi
7.  **vasc:** Vascular lesions



---

## 🛠️ Key Features
* **Neural Scan Animation:** A high-tech UI feedback loop that visualizes the tensor analysis phase.
* **Dynamic Reporting:** Generates a downloadable `.txt` clinical report containing primary predictions and full probability distributions.
* **Cloud Inference:** Seamlessly connected to a Python-based FastAPI backend hosted on Render.

## 🚀 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/nilotpaldhar2004/DermSight-AI-Deep-Learning-for-Skin-Lesion-Classification.git](https://github.com/nilotpaldhar2004/DermSight-AI-Deep-Learning-for-Skin-Lesion-Classification.git)
