# 🏦 End-to-End Loan Amount Prediction Pipeline

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Custom_Transformer-ee4c2c)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Gradient_Boosting-F7931E)
![Status](https://img.shields.io/badge/Status-Production_Ready-success)

## 📌 Project Overview
This project is a complete, production-ready Machine Learning pipeline designed to predict approved loan amounts based on applicant financial data. 

Going beyond standard tabular modeling, this project integrates a **custom-built PyTorch Transformer** to process unstructured loan notes and features a rigorous **Algorithmic Fairness & Bias Diagnostic** module to ensure equitable lending predictions across demographic groups.

## ✨ Key Features
* **Modular Architecture:** Clean, PEP8-compliant, and fully decoupled `src/` modules for EDA, preprocessing, modeling, and evaluation.
* **Advanced Feature Engineering:** Dynamic generation of non-leaking financial ratios (e.g., *Asset-to-Income Ratio*, *Total Assets*).
* **Smart Handling of Outliers & NaNs:** Automated IQR-based outlier capping and skewness-aware missing value imputation (Mean vs. Median).
* **From-Scratch NLP Transformer:** A custom PyTorch `TinyTransformer` (Embedding layer, Positional Encoding, Transformer Block) to extract dense semantic features from unstructured text—built completely without pre-trained models.
* **Algorithmic Fairness Validation:** Automated bias detection that compares Mean Absolute Error (MAE) across demographic proxies (Education, Employment) to flag systemic disparities > 20%.
Pipeline Execution Flow:
Data Loading: Ingests raw CSV data.

* Automated EDA: Generates distribution plots and correlation heatmaps (saved to /eda_outputs).

* Preprocessing: Applies intelligent imputation, caps extreme financial outliers (1.5x IQR), and engineers new wealth metrics.

* NLP Processing: Cleans, stems, and tokenizes textual loan notes.

* Transformer Embeddings: Passes tokens through the PyTorch Transformer to generate dense feature vectors.

* Model Training: Merges tabular and text embedding features to train a robust GradientBoostingRegressor.

* Evaluation & Fairness Check: Outputs overall R² and MAE, followed by a detailed fairness diagnostic report split by demographic groups.

## 📊 Model Evaluation & Fairness Metrics
* **The model achieves a strong baseline performance while maintaining strict demographic parity.

* **Overall R² Score: ~85.6%

* **Fairness Diagnostics: The model successfully passes equity checks, demonstrating negligible MAE variance (< 1%) between varying educational backgrounds and employment types, ensuring no single subgroup is unfairly penalized by the algorithm.
## 📂 Project Structure
```text
├── data/
│   └── loan_approval_dataset.csv   # Raw dataset
├── notebooks/
│   └── experiments.ipynb           # Jupyter notebook for exploratory testing
├── src/
│   ├── eda.py                      # Automated distributions, correlations, and plots
│   ├── preprocessing.py            # Imputation, outlier capping, and feature engineering
│   ├── tokenizer.py                # Regex text cleaning and vocabulary building
│   ├── stemming.py                 # NLTK PorterStemmer integration
│   ├── llm_model.py                # PyTorch TinyTransformer architecture
│   ├── loan_model.py               # GradientBoostingRegressor training and evaluation
│   ├── fairness_check.py           # Demographic MAE variance calculations
│   └── pipeline.py                 # Main orchestrator linking all modules
├── main.py                         # Execution script
├── requirements.txt                # Dependency list
└── README.md                       # Project documentation