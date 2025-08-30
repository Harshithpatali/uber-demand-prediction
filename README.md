# Uber Demand Prediction

Predict hourly Uber ride demand using historical data, weather, and temporal features. This project demonstrates building a robust **time series regression model** with advanced feature engineering and model evaluation.

---

## Features

- **Time Series Feature Engineering**
  - Lag features to capture previous demand
  - Rolling mean & standard deviation to detect trends and seasonality
  - Time-based features (hour of day, day of week, weekend)

- **Modeling Techniques**
  - XGBoost Regressor with hyperparameter tuning
  - Random Forest Regressor as an alternative

- **Target Transformation**
  - Log-transform of the target variable to stabilize variance

- **Evaluation & Visualization**
  - Metrics: MAE, RMSE, R², MAPE
  - Feature importance plots
  - Residuals and predicted vs actual plots

- **Model Persistence**
  - Save and load trained models using `joblib` with versioning

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Harshithpatali/uber-demand-prediction.git
cd uber-demand-prediction
