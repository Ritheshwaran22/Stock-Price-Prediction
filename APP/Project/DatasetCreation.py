"""DatasetCreation.py: Creates datasets from the supplied data with robust lookback support"""

# External Imports
import numpy as np
import pandas as pd

# Internal Imports
from . import constants
from . import SaveData
from . import ScaleData

def createDataset(x, y, timeSteps=1):
    """
    Creates rolling input-target sequence pairs.
    x: feature DataFrame
    y: target Series (Close price)
    timeSteps: number of past days used to predict next day
    """
    X, Y = [], []
    for i in range(len(x) - timeSteps):
        X.append(x.iloc[i:(i + timeSteps)].values)
        Y.append(y.iloc[i + timeSteps])
    return np.array(X), np.array(Y)

def initialize(stocks):
    """
    Splits stock data strictly chronologically, scales it without lookahead leakage,
    and formats into supervised sliding sequences.
    """
    trainSize = len(stocks) - int(constants.VALIDATION_SET_PERCENTAGE * len(stocks))
    train = stocks.iloc[:trainSize]
    validation = stocks.iloc[trainSize:]

    # Scale using training set parameters only
    scaledTrain, scaledValidation = ScaleData.Scale(train, validation, model_name=getattr(constants, 'NAME', None))

    try:
        SaveData.saveCSV(train, "Train")
        SaveData.saveCSV(validation, "Validation")
        SaveData.saveCSV(scaledTrain, "scaledTrain")
        SaveData.saveCSV(scaledValidation, "scaledValidation")
    except Exception:
        pass

    target_ticker = constants.TICKER_TO_PREDICT
    yTrain = scaledTrain[target_ticker]['Close']
    yValidation = scaledValidation[target_ticker]['Close']

    XTrain, YTrain = createDataset(scaledTrain, yTrain, constants.TIMESTEPS)
    XValidation, YValidation = createDataset(scaledValidation, yValidation, constants.TIMESTEPS)

    try:
        SaveData.saveNumpy(XTrain, "XTrain")
        SaveData.saveNumpy(YTrain, "YTrain")
        SaveData.saveNumpy(XValidation, "XValidation")
        SaveData.saveNumpy(YValidation, "YValidation")
    except Exception:
        pass

    return train, validation, XTrain, YTrain, XValidation, YValidation

def initializeForecast(forecastStocks, model_name=None):
    scaledForecast = ScaleData.ScaleForecast(forecastStocks, model_name=model_name)

    # Reorder columns deterministically to match TICKER_SET ordering
    ordered_cols = []
    for ticker in constants.TICKER_SET:
        for price_type in ['Close', 'Volume']:
            if (ticker, price_type) in scaledForecast.columns:
                ordered_cols.append((ticker, price_type))
    if ordered_cols and len(ordered_cols) == len(scaledForecast.columns):
        scaledForecast = scaledForecast[ordered_cols]

    target_ticker = constants.TICKER_TO_PREDICT
    yScaledForecast = scaledForecast[target_ticker]['Close']
    XScaledForecast, YScaledForecast = createDataset(scaledForecast, yScaledForecast, constants.TIMESTEPS)
    
    return XScaledForecast, YScaledForecast
