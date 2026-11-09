''' stockCharts.py : Function to draw stock charts'''

# External Imports
import streamlit as st
import plotly.express as px
import yfinance as yf

# Function to draw Close and Open Prices of ticker
def getStockChart(ticker, parameterList, title):
    try:
        tickerData = yf.Ticker(ticker).history(period='max')
        if tickerData is None or tickerData.empty:
            st.warning(f"No historical chart data available for ticker '{ticker}'.")
            return None

        # Validate requested parameters exist in dataframe
        available_params = [p for p in parameterList if p in tickerData.columns]
        if not available_params:
            st.warning(f"Requested parameters {parameterList} not found in historical data.")
            return None

        fig = px.line(tickerData, x=tickerData.index, y=available_params, title=f"{title} - {ticker.upper()}")
        fig.layout.update(
            xaxis_rangeslider_visible=True,
            xaxis_title="Date",
            yaxis_title="Price / Value",
            template="plotly_white"
        )
        return fig

    except ValueError:
        st.error("Ticker provided is invalid!")
        return None
    except Exception as e:
        st.error(f"Error generating chart: {e}")
        return None
