"""ScaleData.py: Scales the data supplied using independent scalers per ticker without data leakage"""

import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from . import constants

def get_scaler_dir() -> Path:
    app_dir = Path(__file__).resolve().parent.parent
    scaler_dir = app_dir / 'saveScaler'
    scaler_dir.mkdir(parents=True, exist_ok=True)
    return scaler_dir

def is_scaler_compatible(scaler, series, max_rel_diff: float = 2.0) -> bool:
    """Checks whether a loaded scaler has a center_ consistent with the series price domain."""
    if scaler is None or not hasattr(scaler, 'center_'):
        return False
    try:
        med = float(np.median(series))
        if med == 0 or np.isnan(med):
            return True
        scaler_center = float(scaler.center_[0])
        rel_diff = abs(scaler_center - med) / abs(med)
        return rel_diff <= max_rel_diff
    except Exception:
        return False

# Function to perform scaling on the data for training
def Scale(train, validation, model_name=None):
    scaler_dir = get_scaler_dir()
    ticker_scalers = {}
    scaledTrain, scaledValidation = train.copy(), validation.copy()

    for ticker in constants.TICKER_SET:
        if ticker in train.columns:
            scaler = RobustScaler()
            # Fit strictly on training data
            scaler.fit(train[ticker])
            scaledTrain[ticker] = scaler.transform(train[ticker])
            scaledValidation[ticker] = scaler.transform(validation[ticker])
            ticker_scalers[ticker] = scaler

    # Save dictionary of scalers
    joblib.dump(ticker_scalers, scaler_dir / "ticker_scalers.save")
    if model_name:
        joblib.dump(ticker_scalers, scaler_dir / f"{model_name}_ticker_scalers.save")

    # Fit target close scaler strictly on training target
    target_ticker = constants.TICKER_TO_PREDICT
    close_scaler = RobustScaler()
    target_close_train = np.array(train[target_ticker]['Close']).reshape(-1, 1)
    close_scaler.fit(target_close_train)
    joblib.dump(close_scaler, scaler_dir / "closeScaler.save")
    if model_name:
        joblib.dump(close_scaler, scaler_dir / f"{model_name}_closeScaler.save")
        try:
            import json
            with open(scaler_dir / f"{model_name}_scaler_meta.json", 'w') as mf:
                json.dump({'is_original': True, 'model_name': model_name, 'target_ticker': target_ticker}, mf)
        except Exception:
            pass

    return scaledTrain, scaledValidation

# Function to perform inverse scaling for validation predictions
def inverseScale(train, YTrain, YValidation, YPred, model_name=None):
    scaler_dir = get_scaler_dir()
    close_scaler_path = scaler_dir / f"{model_name}_closeScaler.save" if model_name else scaler_dir / "closeScaler.save"
    if not close_scaler_path.exists():
        close_scaler_path = scaler_dir / "closeScaler.save"

    if close_scaler_path.exists():
        closeScaler = joblib.load(close_scaler_path)
    else:
        closeScaler = RobustScaler()
        closeScaler.fit(np.array(train.loc[:, (constants.TICKER_TO_PREDICT, 'Close')]).reshape(-1, 1))
        joblib.dump(closeScaler, close_scaler_path)

    YValidationInv = closeScaler.inverse_transform(np.array(YValidation).reshape(-1, 1))
    YPredInv = closeScaler.inverse_transform(np.array(YPred).reshape(-1, 1))

    return YPredInv, YValidationInv

# Function to scale data for forecasting using saved scalers where available
def ScaleForecast(forecastStocks, model_name=None):
    scaler_dir = get_scaler_dir()
    ticker_scalers = {}

    if model_name:
        specific_path = scaler_dir / f"{model_name}_ticker_scalers.save"
        if specific_path.exists():
            try:
                ticker_scalers = joblib.load(specific_path)
            except Exception:
                ticker_scalers = {}

    if not ticker_scalers:
        scalers_path = scaler_dir / "ticker_scalers.save"
        if scalers_path.exists():
            try:
                ticker_scalers = joblib.load(scalers_path)
            except Exception:
                ticker_scalers = {}

    scaledForecastStocks = forecastStocks.copy()
    for ticker in constants.TICKER_SET:
        if ticker in forecastStocks.columns:
            scaler = ticker_scalers.get(ticker)
            # Validate that scaler matches ticker's price magnitude
            if (scaler is None or 
                not hasattr(scaler, 'center_') or 
                not is_scaler_compatible(scaler, forecastStocks[ticker]['Close'])):
                scaler = RobustScaler()
                scaler.fit(forecastStocks[ticker])
            scaledForecastStocks[ticker] = scaler.transform(forecastStocks[ticker])

    return scaledForecastStocks

# Function to perform inverse scaling for forecasting
def inverseScaleForecast(forecastStocks, YPred, model_name=None, target_ticker=None, return_meta=False):
    scaler_dir = get_scaler_dir()
    if target_ticker is None:
        target_ticker = constants.TICKER_TO_PREDICT

    target_series = forecastStocks[target_ticker]['Close'] if (target_ticker in forecastStocks.columns and 'Close' in forecastStocks[target_ticker].columns) else None

    closeScaler = None
    is_original_scaler = False

    # Check whether an authentic original training scaler exists on disk
    if model_name:
        meta_path = scaler_dir / f"{model_name}_scaler_meta.json"
        if meta_path.exists():
            try:
                import json
                with open(meta_path, 'r') as mf:
                    meta_info = json.load(mf)
                    is_original_scaler = bool(meta_info.get('is_original', False))
            except Exception:
                is_original_scaler = False

    # was_calibrated is True whenever we cannot authenticate an original training scaler
    was_calibrated = not is_original_scaler

    # 1. Try model-specific scaler if model_name is provided
    if model_name:
        specific_path = scaler_dir / f"{model_name}_closeScaler.save"
        if specific_path.exists():
            try:
                loaded = joblib.load(specific_path)
                if target_series is None or is_scaler_compatible(loaded, target_series):
                    closeScaler = loaded
            except Exception:
                pass

    # 2. Try global closeScaler.save if compatible
    if closeScaler is None:
        global_path = scaler_dir / "closeScaler.save"
        if global_path.exists():
            try:
                loaded = joblib.load(global_path)
                if target_series is not None and is_scaler_compatible(loaded, target_series):
                    closeScaler = loaded
            except Exception:
                pass

    # 3. If missing or incompatible, dynamically calibrate from target ticker's recent data
    if closeScaler is None:
        was_calibrated = True
        closeScaler = RobustScaler()
        if target_series is not None:
            closeScaler.fit(np.array(target_series).reshape(-1, 1))
        else:
            closeScaler.fit(np.array([1.0]).reshape(-1, 1))
        # Persist as model-specific calibrated scaler
        if model_name:
            try:
                joblib.dump(closeScaler, scaler_dir / f"{model_name}_closeScaler.save")
            except Exception:
                pass

    YPredInv = closeScaler.inverse_transform(np.array(YPred).reshape(-1, 1))
    if return_meta:
        return YPredInv, was_calibrated
    return YPredInv
