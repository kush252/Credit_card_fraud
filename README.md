# Credit Card Fraud Detection - End-to-End MLOps Pipeline

Welcome to the **End-to-End MLOps Credit Card Fraud Detection** project. This repository contains the complete lifecycle of a real-world machine learning application—from data exploration and model training to deployment, monitoring, and user interaction.

## 🚀 Project Overview

The main objective of this project is to detect fraudulent credit card transactions accurately. Building a fraud detection system presents a unique set of challenges, particularly the **extreme class imbalance** inherent to financial transaction data. This project tackles those challenges head-on while demonstrating production-ready MLOps practices.

As part of the project, a custom React-based bank website frontend was built to interact dynamically with the fraud detection backend.

---

## 📊 The Data & "Crazy" Problem of Class Imbalance

The dataset consists of anonymized credit card transactions containing PCA transformed features (`V1` to `V28`), along with `scaled_amount` and `scaled_time`. 

When exploring the dataset (`data/eda/eda.ipynb`), the extreme disparity between classes becomes apparent:
* **Total Transactions:** 284,807
* **Valid Transactions (Class 0):** 284,315 (~99.83%)
* **Fraudulent Transactions (Class 1):** 492 (~0.17%)

This 99.83% to 0.17% ratio makes basic accuracy metrics irrelevant and makes the predictive task exceedingly difficult, as a model predicting "Not Fraud" for every transaction would still be technically 99.83% accurate!

---

## 🧠 Model Strategy & Metrics

To account for the severe imbalance, specialized models were developed focusing on **F1 Score**, **ROC-AUC**, and **PR-AUC**. 

### 1. Baseline Model (Logistic Regression)
A Logistic Regression model with `class_weight="balanced"` was initially tested.
* **ROC-AUC:** 0.9718
* **F1 Score:** 0.0959
* *Takeaway:* While the ROC-AUC appeared high, the low F1 score indicated that the model struggled significantly with Precision and Recall.

### 2. Final Architecture (Gradient Boosting)
An optimized Gradient Boosting model was deployed with refined hyperparameters (`n_estimators=313`, `learning_rate=0.0139`, `max_depth=10`) yielding significant improvements:
* **ROC-AUC:** 0.9195
* **F1 Score:** **0.7634** 📈
* **PR-AUC:** 0.7085

#### Final Confusion Matrix (Test Set)
| | Predicted: Valid | Predicted: Fraud |
|---|---|---|
| **Actual: Valid** | 42,478 (TN) | 10 (FP) | 
| **Actual: Fraud** | 21 (FN) | 50 (TP) |

*The model successfully identified the vast majority of frauds, while keeping False Positives—which create friction for real bank customers—extremely low (only 10 out of over 42,000).*

---

## 🛠️ MLOps Architecture & Tech Stack

This repository isn't just a Jupyter Notebook; it's structured as a modular MLOps pipeline.

### Core Stack
* **Machine Learning:** `scikit-learn`, `pandas`, `joblib`
* **Backend Delivery:** `FastAPI` (in `deployment/app.py`)
* **Frontend UI:** `React` / `Vite` (Web Bank Application)
* **Containerization:** `Docker` 

### Pipeline Components
1. **Source Tracking (`src/ pipelines/`)**: Modular python models for predictions, training pipelines, and workers. 
2. **Model Registry (`models/`)**: Versioned model dumps (`.joblib`) with paired metadata tracking hyperparameters and performance metrics (e.g. `final_model_v1_metadata.json`).
3. **Deployment & Monitoring (`deployment/`)**: 
   * API hosting for model inference.
   * Automated monitoring tools/scripts (`monitoring_retraining_pipeline.py`) simulating real-world capabilities to catch data drift and queue model retraining.
4. **Interactive UI (`frontend/`)**: Local React server setup that effectively replicates a bank portal where users can input transaction details and instantly see the fraud validation results powered by the backend API.

---

## 💻 Running the Project

**1. Backend / Model API**
Ensure your `venv` is active and dependencies are installed via `requirements.txt`. Start the backend deployment pipeline:
```bash
# Example Run Command (Adjust to your exact runner)
python deployment/app.py
```

**2. Frontend interface**
The Vite React frontend can be launched by navigating to the frontend directory:
```bash
cd frontend
npm install
npm run dev
```

*From the frontend dashboard, you can test specific transactional vectors and observe the model catching anomalies in real-time.*
