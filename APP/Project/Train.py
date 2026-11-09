#!/usr/bin/env python

"""Train.py: Main Training Pipeline for Bidirectional LSTM"""

# External Imports
from pathlib import Path
import pandas as pd
import streamlit as st
try:
    from tensorflow.keras.callbacks import TensorBoard
except ImportError:
    TensorBoard = None
from sklearn import metrics
import numpy as np

# Local Imports
from . import constants
from . import DataFetch
from . import DatasetCreation
from . import ModelDefenition
from . import Prediction
from . import SaveModel
from . import ScaleData
from . import plot

def get_columns(num):
    if hasattr(st, "columns"):
        return st.columns(num)
    return st.beta_columns(num)

def trainModel():
    if TensorBoard is None:
        raise ImportError("TensorFlow is not installed in the current environment. Please install tensorflow to train models.")
    # Data Fetch via Yahoo Finance
    stocks = DataFetch.getData(True)
    if stocks is None or stocks.empty:
        raise ValueError(f"No stock data could be fetched for tickers: {constants.TICKER_SET}")

    # Dataset Creation and Scaling of Data strictly chronological without leakage
    train, validation, XTrain, YTrain, XValidation, YValidation = DatasetCreation.initialize(stocks)

    # Model Definition
    model = ModelDefenition.defineModel(
        XTrain, 
        constants.NO_OF_LSTM_UNITS,
        constants.NO_OF_DENSE_UNITS,
        constants.NO_OF_BIDIRECTIONAL_LAYERS,
        constants.NO_OF_DENSE_LAYERS,
        constants.DROPOUTRATE,
        constants.RETURN_SEQUENCES
    )

    # Tensorboard log directory relative to APP
    app_dir = Path(__file__).resolve().parent.parent
    log_dir = app_dir / 'logs' / constants.NAME
    log_dir.mkdir(parents=True, exist_ok=True)
    tensorboard = TensorBoard(log_dir=str(log_dir))

    # Fit model on training sequence strictly without shuffling time series
    model.fit(
        XTrain, YTrain,
        epochs=constants.EPOCHS,
        batch_size=constants.BATCH_SIZE,
        shuffle=False,
        callbacks=[tensorboard],
        verbose=0
    )

    # Save Model
    SaveModel.saveModelInformation(model)

    # Prediction over Validation set
    YPred = Prediction.predictionForValidation(model, XValidation)

    # Scale the prediction back
    YPredInv, YValidationInv = ScaleData.inverseScale(train, YTrain, YValidation, YPred, model_name=getattr(constants, 'NAME', None))

    # Plot the results
    fig = plot.PlotlyPlotValidation(validation, YPredInv, YValidationInv)

    # Metric evaluation
    mae_val = metrics.mean_absolute_error(YValidationInv, YPredInv)
    r2_val = metrics.r2_score(YValidationInv, YPredInv)
    rmse_val = np.sqrt(metrics.mean_squared_error(YValidationInv, YPredInv))

    col1, col2, col3 = get_columns(3)
    with col1:
        st.metric(label="Validation MAE", value=f"{mae_val:.2f}")
    with col2:
        st.metric(label="Validation RMSE", value=f"{rmse_val:.2f}")
    with col3:
        st.metric(label="Validation R² Score", value=f"{r2_val:.3f}")

    return fig    

if __name__ == "__main__":
    fig = trainModel()
    if fig:
        fig.show()
