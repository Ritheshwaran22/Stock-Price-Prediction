''' stockCharts.py : Function to draw interactive stock charts '''

# External Imports
import streamlit as st
import plotly.express as px
import pandas as pd

from .searchticker import name_convert
from ..Forecast import fetch_market_data

# Function to draw Close, Open, High, Low Prices of ticker
def getStockChart(ticker_input, parameterList, title):
    """
    Renders an interactive Plotly financial line chart for any requested stock symbol.
    Uses resilient multi-strategy data fetching with automatic ticker resolution.
    """
    try:
        clean_ticker = name_convert(str(ticker_input).strip()).upper()
        tickerData = fetch_market_data(clean_ticker, period='2y')

        if tickerData is None or tickerData.empty:
            st.warning(f"No historical chart data available for ticker '{clean_ticker}'.")
            return None

        # Validate requested parameters exist in dataframe
        available_params = [p for p in parameterList if p in tickerData.columns]
        if not available_params:
            if 'Close' in tickerData.columns:
                available_params = ['Close']
            else:
                st.warning(f"Requested parameters {parameterList} not found in historical data.")
                return None

        fig = px.line(
            tickerData,
            x=tickerData.index,
            y=available_params,
            title=f"{title} — {clean_ticker}"
        )
        fig.layout.update(
            xaxis_rangeslider_visible=True,
            xaxis_title="Trading Date",
            yaxis_title="Market Value",
            template="plotly_white",
            hovermode="x unified"
        )
        return fig

    except ValueError:
        st.error("Ticker provided is invalid!")
        return None
    except Exception as e:
        st.error(f"Error generating chart: {e}")
        return None
