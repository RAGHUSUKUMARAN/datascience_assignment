"""
Part 4: Model Evaluation and Comparison
---------------------------------------
Compares ARIMA and Exponential Smoothing forecasts using MAE, RMSE, and MAPE.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Example setup: replace with your actual fitted models and series ---
# Suppose 'test' is your test set, 'forecast_hw' and 'forecast_arima' are their forecasts.
# For demonstration, we’ll fake some data:
np.random.seed(42)
test = pd.Series(np.random.uniform(80, 85, 12), name='Actual')  # actual exchange rate
forecast_hw = test + np.random.normal(0, 0.3, 12)               # Holt-Winters forecast
forecast_arima = test + np.random.normal(0, 0.5, 12)            # ARIMA forecast

# --- Define evaluation functions ---
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return pd.Series({'MAE': mae, 'RMSE': rmse, 'MAPE': mape}, name=model_name)

# --- Compute metrics for both models ---
results_hw = evaluate_model(test, forecast_hw, 'Holt-Winters')
results_arima = evaluate_model(test, forecast_arima, 'ARIMA')

results_df = pd.concat([results_hw, results_arima], axis=1).T
print("\n🔹 Forecast Error Metrics Comparison:\n")
print(results_df)

# --- Visual comparison ---
plt.figure(figsize=(10,6))
plt.plot(test.values, label='Actual', color='black', marker='o')
plt.plot(forecast_hw.values, label='Holt-Winters Forecast', linestyle='--', marker='x')
plt.plot(forecast_arima.values, label='ARIMA Forecast', linestyle='-.', marker='s')
plt.title("Exchange Rate Forecast Comparison")
plt.xlabel("Time Steps (Test Periods)")
plt.ylabel("Exchange Rate")
plt.legend()
plt.grid(True)
plt.show()

# --- Identify best model ---
best_model = results_df['RMSE'].idxmin()
print(f"\n🏆 Best model based on RMSE: {best_model}\n")

# --- Optional: difference plot for error visualization ---
plt.figure(figsize=(8,4))
plt.bar(['Holt-Winters', 'ARIMA'], results_df['RMSE'], color=['#66c2a5', '#fc8d62'])
plt.title("Model RMSE Comparison")
plt.ylabel("RMSE")
plt.show()
