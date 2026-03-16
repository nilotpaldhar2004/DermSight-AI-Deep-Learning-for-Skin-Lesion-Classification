# 📡 Telecom Customer Churn AI Predictor

[![Live Demo](https://img.shields.io/badge/Demo-Live%20Site-brightgreen?style=for-the-badge&logo=github)](https://nilotpaldhar2004.github.io/Telecom-AI-Predictor/)
[![Model](https://img.shields.io/badge/Model-Scikit--Learn-orange?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)
[![Backend](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Hosting](https://img.shields.io/badge/Hosting-Render-black?style=for-the-badge&logo=render)](https://render.com/)

A full-stack Machine Learning application that predicts the likelihood of customer turnover (churn) for telecommunications companies. This project uses a **Scikit-Learn** backend served via **FastAPI** and a modern, responsive **Glassmorphism UI**.

🚀 **Live Demo:** [https://nilotpaldhar2004.github.io/Telecom-AI-Predictor/](https://nilotpaldhar2004.github.io/Telecom-AI-Predictor/)  
⚙️ **API Endpoint:** [https://telecom-ai-predictor.onrender.com/](https://telecom-ai-predictor.onrender.com/)

---

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Frontend** | HTML5, CSS3 (Glassmorphism), JavaScript (Fetch API) |
| **Backend** | Python, FastAPI, Uvicorn |
| **Machine Learning** | Scikit-Learn (v1.6.1), Pandas, NumPy |
| **Deployment** | Render (Backend), GitHub Pages (Frontend) |

---

## 🧠 The Problem
Customer churn is a critical metric for telecom providers. Retaining an existing customer is significantly more cost-effective than acquiring a new one. This tool analyzes customer demographics, account information, and service usage to identify high-risk individuals, allowing businesses to take proactive retention measures.



## 🚀 Key Features
- **Real-time Inference:** Connects directly to a hosted FastAPI backend for instant predictions.
- **Risk Visualization:** Displays **Retention Probability** vs. **Churn Risk** with dynamic, color-coded progress bars.
- **Downloadable Reports:** Generates a `.txt` summary of the analysis for record-keeping and business reporting.
- **Responsive Design:** A neon glassmorphism UI optimized for both desktop and mobile viewing.

---

## 📁 Project Structure
- `main.py`: The FastAPI application handling CORS, data validation, and model prediction.
- `model.pkl`: The trained Random Forest model serialized for production.
- `index.html`: The interactive frontend dashboard providing the user interface.
- `requirements.txt`: List of dependencies including pinned versions for environment stability.

---

## ⚙️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/nilotpaldhar2004/Telecom-AI-Predictor.git](https://github.com/nilotpaldhar2004/Telecom-AI-Predictor.git)
   cd Telecom-AI-Predictor
