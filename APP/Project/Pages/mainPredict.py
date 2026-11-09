''' mainPredict.py : To display forecasting page for all stocks '''

# External Imports
import os
import streamlit as st
import glob
from pathlib import Path

# Internal Imports
from Project import constants
from Project.Forecast import (
    getForecast, 
    generateUniversalForecast, 
    generateMultiStockBoard,
    getuploadedForecast, 
    getuploadedForecast2, 
    parse_model_metadata
)
from Project.Pages.searchticker import name_convert

BASKET_INDIAN = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'INFY.NS', 'BHARTIARTL.NS', 'ITC.NS', 'TATAMOTORS.NS', 'LT.NS']
BASKET_US = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AMD', 'NFLX', 'JPM']
BASKET_INDICES = ['^NSEI', '^BSESN', '^GSPC', '^IXIC', 'BTC-USD', 'ETH-USD']

POPULAR_SHORTCUTS = [
    "RELIANCE.NS", "SBIN.NS", "TCS.NS", "HDFCBANK.NS", "TATAMOTORS.NS", "INFY.NS",
    "AAPL", "MSFT", "NVDA", "TSLA", "GOOGL", "AMZN", "^NSEI", "^GSPC"
]

def mainPredict():
    st.subheader("Global Stock Price Forecasting")
    st.write(
        "Forecast future price trajectories for **any stock in the world** or view live multi-stock "
        "predictions across Indian and global market sectors."
    )

    # Sidebar: Universal Search & Custom Ticker Input
    st.sidebar.header("🔍 Search Any Stock Worldwide")
    default_ticker = st.session_state.get('selected_ticker', 'SBIN.NS')

    user_query = st.sidebar.text_input(
        label="Enter Any Company Name or Ticker",
        value=default_ticker,
        help="Type ANY stock worldwide (e.g., Zomato, Reliance, SBI, Apple, Tesla, TCS, MRF, NVDA, BTC-USD)"
    )

    resolved_ticker = name_convert(user_query).upper()
    st.session_state['selected_ticker'] = resolved_ticker
    constants.TICKER_TO_PREDICT = resolved_ticker
    constants.TICKER_SET = [resolved_ticker]

    st.sidebar.caption(f"Resolved Yahoo Finance Symbol: **`{resolved_ticker}`**")

    # Quick Select Shortcuts in Sidebar
    st.sidebar.markdown("**Quick Picks:**")
    shortcut_cols = st.sidebar.columns(3)
    for idx, sym in enumerate(POPULAR_SHORTCUTS[:6]):
        with shortcut_cols[idx % 3]:
            if st.button(sym.split('.')[0], key=f"quick_{sym}"):
                st.session_state['selected_ticker'] = sym
                resolved_ticker = sym
                constants.TICKER_TO_PREDICT = sym
                constants.TICKER_SET = [sym]

    # Forecast Controls
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Forecast Parameters")
    forecast_days = int(st.sidebar.number_input(
        label='Forecast Horizon (Days Ahead)',
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help='Number of forward business days to project into the future (1 to 10)'
    ))
    constants.PERIOD_TO_FORECAST = forecast_days

    lookback_window = st.sidebar.selectbox(
        label='Historical Data Lookback',
        options=['3mo', '6mo', '1y', '2y', '5y'],
        index=2,
        help='Historical data window used for pattern learning and trend modeling'
    )

    # 4 Main Tabs
    tab_all, tab_single, tab_models, tab_upload = st.tabs([
        "📊 All Stocks Predictions Board (Multi-Stock Screener)",
        "🔮 Single Stock Deep Forecast",
        "💾 Pre-Trained & Saved Models Catalog",
        "📤 Upload Custom Model"
    ])

    # Tab 1: All Stocks Predictions Board (Multi-Stock Screener)
    with tab_all:
        st.markdown("### 📊 Market-Wide Predictions Board (All Stocks Screener)")
        st.write(
            "Scan multiple stocks at once to compare their current prices, projected returns, "
            "and technical buy/sell signals."
        )

        col_b1, col_b2 = st.columns([2, 1])
        with col_b1:
            basket_choice = st.selectbox(
                "Select Market Sector / Stock Basket to Analyze",
                [
                    "🇮🇳 Top 10 Indian Nifty Leaders (Reliance, SBI, TCS, HDFC, Tata Motors...)",
                    "🇺🇸 Top 10 US Mega-Caps (Apple, Microsoft, Nvidia, Tesla, Google...)",
                    "🌐 Global Major Indices & Crypto (Nifty 50, Sensex, S&P 500, Bitcoin...)",
                    "✍️ Custom Watchlist (Enter your own tickers)"
                ]
            )

        if "Indian" in basket_choice:
            active_basket = BASKET_INDIAN
        elif "US" in basket_choice:
            active_basket = BASKET_US
        elif "Global" in basket_choice:
            active_basket = BASKET_INDICES
        else:
            custom_basket_str = st.text_area(
                "Enter comma-separated tickers to analyze",
                value="RELIANCE.NS, SBIN.NS, AAPL, MSFT, NVDA, TSLA",
                help="Type any valid tickers separated by commas"
            )
            active_basket = [t.strip().upper() for t in custom_basket_str.split(',') if t.strip()]

        st.info(f"Target Basket: `{', '.join(active_basket)}` | Horizon: `{forecast_days} Day(s) Ahead`")

        if st.button("⚡ Scan & Predict All Selected Stocks", key="btn_run_multiboard"):
            with st.spinner("Fetching market data and running machine learning predictions for all stocks..."):
                generateMultiStockBoard(active_basket, horizon=forecast_days)

    # Tab 2: Single Stock Deep AI Forecast
    with tab_single:
        st.markdown(f"### 🔮 Deep AI Price Forecast — `{resolved_ticker}`")
        st.write(
            f"Generate multi-day forward projections, confidence interval envelope, "
            f"and technical momentum indicators for **`{resolved_ticker}`**."
        )

        st.info(f"Active Target Stock: **`{resolved_ticker}`** | Horizon: **`{forecast_days} Day(s)`** | Lookback: **`{lookback_window}`**")

        if st.button(f"Generate Forecast for {resolved_ticker}", key="btn_run_single_universal"):
            with st.spinner(f"Analyzing historical trends and projecting forward prices for {resolved_ticker}..."):
                generateUniversalForecast(resolved_ticker, period=forecast_days, lookback=lookback_window)

    # Tab 3: Pre-Trained & Saved Models Catalog
    with tab_models:
        st.markdown("### 💾 Pre-Trained & Saved Models Catalog")
        st.write("Browse and load dedicated neural network models saved on disk.")

        app_dir = Path(__file__).resolve().parent.parent.parent
        info_dir = app_dir / 'savedModelInformation' / 'modelInformation'
        weights_dir = app_dir / 'savedModelInformation' / 'modelWeights'

        model_json_files = [str(p) for p in info_dir.glob("*.json") if not p.name.endswith('scaler_meta.json')]
        model_weight_files = [str(p) for p in weights_dir.glob("*.h5")]

        if not model_json_files or not model_weight_files:
            st.warning("No saved models found in `savedModelInformation`. Please train a model first in the Train tab.")
        else:
            def model_sort_key(p):
                meta = parse_model_metadata(p)
                return (0 if meta.get('target') == resolved_ticker else 1, os.path.basename(p))

            sorted_models = sorted(model_json_files, key=model_sort_key)
            model_display_names = []
            for f in sorted_models:
                m = parse_model_metadata(f)
                target_str = m.get('target') or 'Unknown'
                tag = "⭐ (Matches Active Stock)" if target_str == resolved_ticker else ""
                model_display_names.append(f"{os.path.basename(f)} {tag}".strip())

            selected_idx = st.selectbox(
                label='Select Saved Model Architecture (.json)',
                options=range(len(sorted_models)),
                format_func=lambda idx: model_display_names[idx],
                help='Choose a pre-trained Bidirectional LSTM model'
            )
            selectedModelName = sorted_models[selected_idx]

            target_stem = Path(selectedModelName).stem
            matched_weight = next((wf for wf in model_weight_files if Path(wf).stem == target_stem), model_weight_files[0])
            selectedModelWeights = matched_weight

            meta = parse_model_metadata(selectedModelName)
            if meta['target'] and meta['tickers']:
                st.info(
                    f"📌 **Model Configuration:** Target: `{meta['target']}` | "
                    f"Required Features: `{', '.join(meta['tickers'])}` | "
                    f"Architecture: Bidirectional LSTM | "
                    f"Trained: `{meta['timestamp_str'] or 'Pre-Trained'}`"
                )

            if st.button("Run Saved Model Forecast", key="btn_run_saved"):
                with st.spinner(f"Running saved model inference for {meta.get('target', resolved_ticker)}..."):
                    getForecast(forecast_days, selectedModelName, selectedModelWeights)

    # Tab 4: Upload Custom Model
    with tab_upload:
        st.markdown("### 📤 Upload Custom Model")
        st.write("Upload your own Keras model architecture (.json) and weights (.zip containing .h5) or full model bundle.")

        upload_choice = st.radio(
            "Upload Format",
            ["Architecture & Weights (.json + .zip)", "Bundled Archive (.zip)"]
        )

        if upload_choice == "Architecture & Weights (.json + .zip)":
            uploadedmodel = st.file_uploader(label='Upload Model Architecture (.json)', type=['json'], accept_multiple_files=False)
            uploadedweight = st.file_uploader(label='Upload Model Weights (.zip containing .h5)', type=['zip'], accept_multiple_files=False)
            if uploadedmodel is not None and uploadedweight is not None:
                if st.button('Run Custom Uploaded Forecast', key='btn_run_upload1'):
                    with st.spinner("Generating forecast from uploaded artifacts..."):
                        getuploadedForecast(forecast_days, uploadedmodel, uploadedweight)
            else:
                st.info("Please upload both the model architecture JSON and the weights archive.")
        else:
            stream = st.file_uploader('Upload TF.Keras Saved Model archive (.zip)', type=['zip'])
            if stream is not None:
                if st.button('Run Bundled Forecast', key='btn_run_upload2'):
                    with st.spinner("Generating forecast from bundled model archive..."):
                        getuploadedForecast2(forecast_days, stream)
            else:
                st.info("Please upload a valid model archive (.zip).")
