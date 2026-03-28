# 🔬 DermSight PRO: Deep Residual Learning for Skin Lesion Classification

[![Live Demo](https://img.shields.io/badge/Demo-Live%20Site-brightgreen?style=for-the-badge&logo=github)](https://nilotpaldhar2004.github.io/DermSight-AI-Deep-Learning-for-Skin-Lesion-Classification/)
[![Neural Engine](https://img.shields.io/badge/Engine-ResNet--50-blue?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![Backend](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Dataset](https://img.shields.io/badge/Dataset-HAM10000-orange?style=for-the-badge)](https://doi.org/10.7910/DVN/DBW86T)

DermSight PRO is a high-performance computer vision application designed to classify dermoscopic skin images into seven diagnostic categories. By leveraging a fine-tuned **ResNet-50** architecture, the system identifies visual patterns associated with both benign and malignant pathologies.

🚀 **[Access the Live Demo Here](https://nilotpaldhar2004.github.io/DermSight-AI-Deep-Learning-for-Skin-Lesion-Classification/)**

---

## 🧠 Neural Architecture & Engineering

The core of DermSight PRO is built on a **ResNet-50** backbone, pre-trained on ImageNet and optimized for dermatological feature extraction.



### Why ResNet-50?
Traditional deep networks often suffer from the **vanishing gradient problem**. ResNet-50 utilizes **Residual Blocks** and **Skip Connections**, allowing the gradient to propagate effectively to early layers. This enables the training of deeper architectures without performance degradation.

### Technical Optimization
* **Transfer Learning:** Convolutional layers were frozen to preserve low-level feature extraction (edges, textures) while fine-tuning the global average pooling and fully connected layers for clinical dermoscopy.
* **Weighted Cross-Entropy:** To handle the 10:1 class imbalance in the **HAM10000** dataset, a weighted loss function was implemented to increase the penalty for misclassifying high-risk cases like Melanoma (MEL).
* **Inference Pipeline:** Image tensors are resized to $224 \times 224$ and normalized using $\mu=[0.485, 0.456, 0.406]$ and $\sigma=[0.229, 0.224, 0.225]$ to match ImageNet standards.

---

## 🔬 Diagnostic Categories

The engine classifies images into seven specific classes based on the International Skin Imaging Collaboration (ISIC) standards:

| Code | Disease | Type |
| :--- | :--- | :--- |
| **MEL** | Melanoma | Malignant |
| **BCC** | Basal Cell Carcinoma | Malignant |
| **AKIEC** | Actinic Keratoses | Pre-cancerous |
| **BKL** | Benign Keratosis | Benign |
| **NV** | Melanocytic Nevi | Benign |
| **DF** | Dermatofibroma | Benign |
| **VASC** | Vascular Lesions | Benign |

---

## 🛠️ Key Features

* **Neural Scan UI:** A high-fidelity frontend that visualizes the "tensor analysis" phase using custom scanline animations.
* **Dynamic Probability Distribution:** Real-time feedback showing confidence levels across all 7 categories via animated progress bars.
* **Automated Clinical Reporting:** Generates a downloadable `.txt` report including primary predictions, confidence scores, and clinical definitions.
* **Asynchronous Inference:** Powered by a **FastAPI** backend hosted on Render for non-blocking image processing.

---


## 🚀 Installation & Local Setup
      git clone https://github.com/nilotpaldhar2004/DermSight-AI-Deep-Learning-for-Skin-Lesion-Classification.git
      cd DermSight-AI-Deep-Learning-for-Skin-Lesion-Classification
      pip install -r requirements.txt
      uvicorn main:app --reload

## 📈 Project Insights

<p align="center">
  <a href="https://star-history.com/#nilotpaldhar2004/DermSight-AI-Deep-Learning-for-Skin-Lesion-Classification&Date">
    <img src="https://api.star-history.com/svg?repos=nilotpaldhar2004/DermSight-AI-Deep-Learning-for-Skin-Lesion-Classification&theme=dark" alt="Star History Chart" width="600"/>
  </a>
</p>

---
## ⚠️ Medical Disclaimer
This application is developed for educational and research purposes only. It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## 🔹 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

---

Developed by Nilotpal Dhar • March 2026
