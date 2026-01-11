# 🔐 SentinelPay: End-to-End Financial Fraud Detection System

## 1. Project Overview
SentinelPay is a professional-grade MLOps pipeline designed to detect fraudulent transactions within financial ecosystems. The project leverages a high-dimensional synthetic dataset to predict the probability of fraud ($y \in \{0, 1\}$) using advanced Gradient Boosting and Neural Network architectures. It transitions from experimental prototyping to a modular, production-ready codebase.

**Dataset Source:** [Transactions Fraud Datasets (Kaggle)](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets)

---

## 2. Repository Architecture
The project follows a modular structure to ensure maintainability and scalability in a production environment.

```text
MLOPS-PROJECT-FINANCIAL-DETECTION/
├── artifacts/          # Serialized models (.pkl), scalers, and encoders
├── config/             # YAML configurations for hyperparameters and paths
├── logs/               # Production-level logging for pipeline execution
├── mlruns/             # MLflow experiment tracking metadata
├── notebook/           # EDA and initial model prototyping
│   └── end-to-end-Financial-Fraud-Detection.ipynb
├── src/                # Modular Python source code
│   ├── components/     # Ingestion, Transformation, Model Training
│   ├── pipeline/       # Training and Prediction pipelines
│   └── utils/          # Common utility functions
├── .gitignore          # Version control exclusions
├── README.md           # Documentation
├── requirements.txt    # Dependency manifest (PEP 8 compliant)
└── setup.py            # Package configuration