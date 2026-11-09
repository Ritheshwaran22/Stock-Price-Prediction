# -*- coding: utf-8 -*-
"""
arima.py: Standalone ARIMA Time-Series Forecasting on Dow Jones Historical Data
Modernized for compatibility with Statsmodels >= 0.14 and Pandas >= 2.0
"""

from pathlib import Path
from datetime import datetime
import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

# Modern statsmodels ARIMA import
try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    from statsmodels.tsa.arima_model import ARIMA

script_dir = Path(__file__).resolve().parent
csv_path = script_dir / 'arima.csv'

if not csv_path.exists():
    print(f"Dataset not found at {csv_path}")
    exit(1)

df = pd.read_csv(csv_path, usecols=['Open', 'High', 'Low', 'Close', 'Date'])
df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")
df.set_index('Date', inplace=True)
print(df.info())
print(df.head())

p, d, q = 4, 1, 0
train_data, test_data = train_test_split(df['2015':'2020'], random_state=42, test_size=0.3, shuffle=False) 
training_data = train_data['Close'].values
testing_data = test_data['Close'].values
history = [x for x in training_data]
model_predictions = []
N_test_observations = len(testing_data)

print(f"Running ARIMA({p},{d},{q}) walk-forward validation over {N_test_observations} steps...")
for time_point in range(min(N_test_observations, 50)):  # sample first 50 for quick validation
    try:
        model = ARIMA(history, order=(p, d, q))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0] if hasattr(output, '__getitem__') else float(output)
        model_predictions.append(yhat)
        true_test_value = testing_data[time_point]
        history.append(true_test_value)
    except Exception as e:
        print(f"Step {time_point} error: {e}")
        break

if model_predictions:
    eval_len = len(model_predictions)
    MSE_error = mean_squared_error(testing_data[:eval_len], model_predictions)
    MAE_error = mean_absolute_error(testing_data[:eval_len], model_predictions)
    print(f'Test Mean Squared Error (MSE): {MSE_error:.4f}')
    print(f'Test Mean Absolute Error (MAE): {MAE_error:.4f}')
