''' mainOverview.py : Page for navigating the various overview pages '''

# External Imports
import streamlit as st

# Internal Imports
from . import companyOverview
from . import financialsOverview
from . import stockCharts
from .searchticker import name_convert
from ..Forecast import generateUniversalForecast

POPULAR_OVERVIEW_STOCKS = [
    ("Reliance Industries (RELIANCE.NS)", "RELIANCE.NS"),
    ("Apple (AAPL)", "AAPL"),
    ("Microsoft (MSFT)", "MSFT"),
    ("NVIDIA (NVDA)", "NVDA"),
    ("Tesla (TSLA)", "TSLA"),
    ("Google (GOOGL)", "GOOGL"),
    ("Amazon (AMZN)", "AMZN"),
    ("Tata Consultancy Services (TCS.NS)", "TCS.NS"),
    ("Infosys (INFY.NS)", "INFY.NS"),
    ("HDFC Bank (HDFCBANK.NS)", "HDFCBANK.NS"),
    ("State Bank of India (SBIN.NS)", "SBIN.NS"),
    ("Tata Motors (TATAMOTORS.NS)", "TATAMOTORS.NS"),
    ("Other / Custom Ticker...", "CUSTOM")
]

# Function calling Overview Pages
def mainOverview():
    current_default = st.session_state.get('selected_ticker', 'RELIANCE.NS')
    pop_codes = [item[1] for item in POPULAR_OVERVIEW_STOCKS]
    default_idx = pop_codes.index(current_default) if current_default in pop_codes else pop_codes.index("CUSTOM")

    st.sidebar.header("🔍 Stock Search & Selection")
    stock_choice = st.sidebar.selectbox(
        label="Select Stock",
        options=range(len(POPULAR_OVERVIEW_STOCKS)),
        index=default_idx,
        format_func=lambda i: POPULAR_OVERVIEW_STOCKS[i][0],
        help="Quickly pick a leading stock or choose Custom to type any name."
    )

    chosen_code = POPULAR_OVERVIEW_STOCKS[stock_choice][1]
    if chosen_code == "CUSTOM":
        companyName = st.sidebar.text_input(
            'Enter Company Name / Ticker',
            value=current_default if current_default not in pop_codes else 'RELIANCE.NS',
            help="Type any symbol or company name (e.g., Reliance Industries, AAPL, SBIN.NS, MSFT, TSLA)"
        )
    else:
        companyName = chosen_code

    tickerInformation = st.sidebar.selectbox(
        label='Overview Section',
        options=['Price Predictions & Forecast', 'Ticker Information', 'Financials', 'Charts'],
        index=0,
        help='Select data view for this ticker'
    )

    if not companyName.strip():
        st.info("Please enter a company name or ticker symbol in the sidebar.")
        return

    with st.spinner('Fetching Market Data...'):
        ticker = name_convert(companyName).upper()
        st.session_state['selected_ticker'] = ticker

        st.caption(f"Active Stock: **`{ticker}`** | Section: **`{tickerInformation}`**")

        if tickerInformation == 'Price Predictions & Forecast':
            st.markdown(f"### 🔮 Forward Price Predictions — `{ticker}`")
            st.write(f"Generating multi-day forward forecast and price trajectory for **{ticker}**:")
            generateUniversalForecast(ticker, period=3, lookback='1y')

        elif tickerInformation == 'Ticker Information':
            companyOverview.getCompanyOverview(ticker)
            
            st.markdown("---")
            st.markdown(f"### 🔮 Instant Price Forecast for `{ticker}`")
            with st.expander(f"Click to View 3-Day Forward Forecast for {ticker}", expanded=True):
                generateUniversalForecast(ticker, period=3, lookback='1y')

        elif tickerInformation == 'Financials':
            financialsOverview.getFinancialsOverview(ticker)

        elif tickerInformation == 'Charts':
            try:
                fig_close = stockCharts.getStockChart(ticker, ['Close'], f'{ticker} - Historical Close Price')
                if fig_close:
                    st.plotly_chart(fig_close, use_container_width=True)

                fig_vol = stockCharts.getStockChart(ticker, ['Volume'], f'{ticker} - Traded Volume')
                if fig_vol:
                    st.plotly_chart(fig_vol, use_container_width=True)

            except Exception as e:
                st.warning(f"Unable to render stock charts: {e}")
