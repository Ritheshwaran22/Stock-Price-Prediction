"""plot.py: Visualizations of Data"""

# External Imports
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import streamlit as st
import numpy as np

# Internal Imports
from . import constants
from .CurrencyConversion import inverseConvertCurrency

def PlotlyPlotValidation(validation, YPredInv, YValidationInv):
    """
    Plots the validation actual vs predicted close prices using Plotly.
    Properly aligns date indices to match prediction output length.
    """
    y_pred = inverseConvertCurrency(YPredInv).flatten()
    y_val = inverseConvertCurrency(YValidationInv).flatten()

    # Ensure index length matches prediction array length exactly
    val_dates = validation.index[-len(y_pred):]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=val_dates, 
        y=y_val, 
        mode='lines', 
        name='Actual Close Price',
        line=dict(color='#1f77b4', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=val_dates, 
        y=y_pred, 
        mode='lines', 
        name='Predicted Close Price',
        line=dict(color='#ff7f0e', width=2, dash='dot')
    ))

    fig.layout.update(
        title=f"Validation Performance: {constants.TICKER_TO_PREDICT}",
        xaxis_title="Date",
        yaxis_title="Price (Local Currency)",
        template="plotly_white",
        hovermode="x unified",
        xaxis_rangeslider_visible=(constants.PERIOD in ['1y', '5y', '10y', 'max'])
    )
    return fig

# Function to plot the output graph with Matplotlib
def PlotValidation(YPredInv, YValidationInv):
    plt.figure(figsize=(10, 5))
    plt.plot(YValidationInv.flatten(), marker='.', label="True")
    plt.plot(YPredInv.flatten(), 'r', label="Prediction")
    plt.ylabel('Close Price')
    plt.xlabel('Time Step')
    plt.legend()
    plt.title(f"Model Validation: {constants.TICKER_TO_PREDICT}")
    plt.show()

# Function to plot the compound scores of Finviz Sentimental Analysis
def PlotFinvizSentimentalAnalysisResults(news):
    meanNewsData = news.groupby(['Ticker', 'Date'], as_index=False)[['Compound']].mean()
    pivotData = meanNewsData.pivot(index='Date', columns='Ticker', values='Compound').fillna(0)
    pivotData.plot(kind='bar', figsize=(10, 5))
    plt.ylabel("Average Compound Sentiment")
    plt.title("Headline Sentiment by Ticker")
    plt.show()

def PlotlyPlotFinvizSentimentalAnalysisResults(news):
    meanNewsData = news.groupby(['Ticker', 'Date'], as_index=False)[['Compound']].mean()
    pivotData = meanNewsData.pivot(index='Date', columns='Ticker', values='Compound').fillna(0)
    
    fig = go.Figure()
    for ticker in pivotData.columns:
        fig.add_trace(go.Scatter(
            x=pivotData.index, 
            y=pivotData[ticker], 
            mode='lines+markers', 
            name=ticker
        ))
    fig.layout.update(
        title="Headline Sentiment Trend",
        xaxis_title="Date",
        yaxis_title="VADER Compound Score",
        template="plotly_white"
    )
    return fig
