"""Prediction.py: Prediction based on supplied data"""


# External Imports
try:
    import tensorflow as tf
except ImportError:
    tf = None

# Function to perform prediction


def predictionForValidation(model, XValidation):
    YPred = model.predict(XValidation)
    return YPred
