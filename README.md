# DermSight PRO: Deep Residual Learning for Skin Lesion Classification

[![Live Demo](https://img.shields.io/badge/Demo-Live%20Site-brightgreen?style=for-the-badge&logo=github)](https://nilotpaldhar2004.github.io/DermSight-AI-Deep-Learning-for-Skin-Lesion-Classification/)
[![Neural Engine](https://img.shields.io/badge/Engine-ResNet--50-blue?style=flat-square)](https://pytorch.org/)
[![Backend](https://img.shields.io/badge/Backend-FastAPI-green?style=flat-square)](https://fastapi.tiangolo.com/)
[![Dataset](https://img.shields.io/badge/Dataset-HAM10000-orange?style=flat-square)](https://doi.org/10.7910/DVN/DBW86T)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)

DermSight PRO is a high-performance computer vision application designed to classify dermoscopic skin images into seven diagnostic categories. By leveraging a fine-tuned Deep Residual Network (ResNet-50), the system identifies visual patterns associated with both benign and malignant pathologies.

 **[Access the Live Demo Here](https://nilotpaldhar2004.github.io/DermSight-AI-Deep-Learning-for-Skin-Lesion-Classification/)**

---

##  Neural Architecture & Engineering

The core of DermSight PRO is built on a **ResNet-50** backbone, pre-trained on ImageNet and optimized for dermatological feature extraction.

### Why ResNet-50?
Traditional deep networks often suffer from the **vanishing gradient problem**. ResNet-50 utilizes **Residual Blocks** and **Skip Connections**, which allow the model to learn identity mappings. This ensures that the gradient can propagate back to early layers, enabling the training of deeper architectures without performance degradation.



### Technical Optimization
* **Transfer Learning:** Froze initial convolutional layers to preserve low-level feature extraction (edges, textures) while fine-tuning the final layers for clinical dermoscopy.
* **Weighted Cross-Entropy:** To handle the 10:1 class imbalance in the HAM10000 dataset, I implemented a weighted loss function to increase the penalty for misclassifying high-risk cases like Melanoma (MEL).
* **Inference Pipeline:** Image tensors are normalized to $(\mu=0.5, \sigma=0.5)$ before passing through the final Softmax layer to produce a probability distribution.

---

##  Diagnostic Categories (HAM10000)

The engine classifies images into seven specific classes based on the Human Against Machine dataset:

1.  **akiec:** Actinic keratoses (Pre-cancerous)
2.  **bcc:** Basal cell carcinoma (Malignant)
3.  **bkl:** Benign keratosis-like lesions
4.  **df:** Dermatofibroma (Benign)
5.  **mel:** Melanoma (Malignant)
6.  **nv:** Melanocytic nevi (Benign)
7.  **vasc:** Vascular lesions (Benign)



---

##  Key Features

* **Neural Scan UI:** A high-fidelity frontend that visualizes the "tensor analysis" phase using a custom scanline animation.
* **Dynamic Probability Distribution:** Real-time feedback showing the confidence level for all 7 categories via animated progress bars.
* **Automated Clinical Reporting:** Generates a downloadable `.txt` report including primary predictions, confidence scores, and clinical definitions.
* **Asynchronous Cloud Inference:** Built with a FastAPI backend hosted on Render for non-blocking image processing.

---

##  Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/nilotpaldhar2004/DermSight-AI-Deep-Learning-for-Skin-Lesion-Classification.git](https://github.com/nilotpaldhar2004/DermSight-AI-Deep-Learning-for-Skin-Lesion-Classification.git)
