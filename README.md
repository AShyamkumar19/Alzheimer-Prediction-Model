# AI-Driven Alzheimer's Diagnosis and MMSE Score Optimization

This project uses machine learning and reinforcement learning to:
- Predict Alzheimer's disease diagnosis and MMSE (Mini-Mental State Examination) scores from clinical and lifestyle data.
- Simulate lifestyle interventions to improve predicted cognitive outcomes.

## Project Overview

The pipeline includes:
- **Supervised Learning**:
  - `Random Forest` and `XGBoost` models for classification (Alzheimer's diagnosis) and regression (MMSE prediction).
- **Model Interpretability**:
  - `SHAP` (SHapley Additive exPlanations) to explain important features.
- **Reinforcement Learning Agent**:
  - A `Q-Learning` agent to recommend actions (e.g., improve diet, reduce alcohol) that improve MMSE.

## Dataset

For more information about the dataset, please see** [`data/README.md`](data/README.md).

> Source: [Kaggle Alzheimer's Disease Dataset](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset)

## Models

- `RandomForestClassifier` & `XGBClassifier`: Predict binary diagnosis (`Alzheimer's` or `No Alzheimer's`)
- `RandomForestRegressor` & `XGBRegressor`: Predict MMSE cognitive scores (0–30)
- `Hybrid RF → XGB`: Uses RF for feature selection and XGB for MMSE regression
- `Q-Learning Agent`: Trained on predicted MMSE values to

## Project Structure
- **data/**
  - `alzheimers_disease_data.csv`
  - `README.md`  _Dataset details are described here_
  - `FEATURE ANALYSIS.ipynb`
- **models/**
  - `rf_reg.joblib`
  - `xgb_reg.joblib`
  - `scaler.joblib`
  - `training.py`
- **RL_Agent/**
  - `mmse_rl_agent.py`
- **app/**
  - `predictor.py`
- **utils/**
  - `preprocessing.py`
- `main.py`
- `requirements.txt`
- `README.md`

## How to Run

1. **Clone this repo:**

```bash
git clone https://github.com/AShyamkumar19/Alzheimer-Prediction-Model.git
pip install -r requirements.txt
python main.py
