''' mainTrain.py : Page for training process'''

# External Imports
import time
import streamlit as st
import yfinance as yf
from PIL import Image
from urllib.request import urlopen

# Internal Imports
from .. import constants
from .. import Train

def get_columns(num):
    if hasattr(st, "columns"):
        return st.columns(num)
    return st.beta_columns(num)

# Function to create UI for train 
def mainTrain():
    st.subheader("Model Training Configuration")
    st.write("Configure hyperparameters and train a custom Bidirectional LSTM model for your selected equities.")

    default_target = st.session_state.get('selected_ticker', 'AAPL')
    training_mode = st.sidebar.radio(
        label='Training Architecture Mode',
        options=['Single Stock (Self Close + Volume)', 'Correlated Multi-Stock Basket'],
        index=0,
        help='Single stock mode trains a lightweight dedicated 2D model. Correlated mode includes related tickers.'
    )

    uinputticker = st.sidebar.text_input(
        label='Target Ticker to Predict', 
        value=default_target, 
        help='Ticker to forecast (e.g. AAPL, MSFT, TSLA, SBIN.NS, RELIANCE.NS)'
    )
    constants.TICKER_TO_PREDICT = uinputticker.strip().upper()
    st.session_state['selected_ticker'] = constants.TICKER_TO_PREDICT

    if training_mode == 'Correlated Multi-Stock Basket':
        st.sidebar.subheader('Related Tickers')
        userInputTickers = st.sidebar.text_area(
            label='Related Tickers (Comma separated)', 
            value="MSFT" if constants.TICKER_TO_PREDICT == "AAPL" else "ICICIBANK.NS", 
            help='Enter correlated tickers separated by commas to provide market context'
        )
        ticker_set = [i.strip().upper() for i in userInputTickers.split(',') if i.strip()]
        if constants.TICKER_TO_PREDICT not in ticker_set:
            ticker_set.append(constants.TICKER_TO_PREDICT)
        constants.TICKER_SET = ticker_set
    else:
        constants.TICKER_SET = [constants.TICKER_TO_PREDICT]

    # Yfinance Controls
    st.sidebar.subheader('Data Parameters')
    constants.PERIOD = st.sidebar.select_slider(
        label='Historical Period',
        value='5y',
        options=['1mo', '5mo', '1y', '5y', '10y', 'max'], 
        help='Range of historical data to fetch'
    )
    
    # Training Controls
    st.sidebar.subheader('Training Parameters')
    constants.EPOCHS = st.sidebar.slider(
        label='Epochs', 
        min_value=1, 
        max_value=100, 
        value=20, 
        step=1, 
        help='Number of training iterations'
    )
    constants.BATCH_SIZE = st.sidebar.slider(
        label='Batch Size', 
        min_value=8, 
        max_value=256, 
        value=64, 
        step=8, 
        help='Batch size of input sequences'
    )
    constants.VALIDATION_SET_PERCENTAGE = 0.01 * (st.sidebar.slider(
        label='Validation Split (%)', 
        min_value=10, 
        max_value=40, 
        value=15, 
        step=5, 
        help='Percentage of data reserved for validation'
    ))
    constants.LEARNING_RATE = st.sidebar.slider(
        label='Learning Rate', 
        min_value=0.001, 
        max_value=0.05, 
        value=0.01, 
        step=0.005, 
        help='Optimizer learning rate'
    )

    # Model Controls
    st.sidebar.subheader('Model Parameters')
    constants.NO_OF_LSTM_UNITS = st.sidebar.slider(
        label='LSTM Units', 
        min_value=32, 
        max_value=512, 
        value=128, 
        step=32, 
        help='Neurons in LSTM layer'
    )
    constants.NO_OF_DENSE_UNITS = st.sidebar.slider(
        label='Dense Units', 
        min_value=16, 
        max_value=256, 
        value=32, 
        step=16, 
        help='Neurons in dense layer'
    )
    constants.NO_OF_BIDIRECTIONAL_LAYERS = int(st.sidebar.number_input(
        label='Number of LSTM Layers', 
        min_value=1, 
        max_value=3, 
        value=1, 
        step=1
    ))
    constants.NO_OF_DENSE_LAYERS = int(st.sidebar.number_input(
        label='Number of Dense Layers', 
        min_value=1, 
        max_value=3, 
        value=1, 
        step=1
    ))
    constants.DROPOUTRATE = st.sidebar.slider(
        label='Dropout Rate', 
        min_value=0.01, 
        max_value=0.5, 
        value=0.05, 
        step=0.01
    )
    
    constants.RETURN_SEQUENCES = (constants.NO_OF_BIDIRECTIONAL_LAYERS > 1)
    constants.NAME = f"{constants.TICKER_TO_PREDICT}-SEQ-{'-'.join(constants.TICKER_SET)}-PRED-{int(time.time())}"

    if constants.NO_OF_LSTM_UNITS <= constants.NO_OF_DENSE_UNITS:
        st.warning('Recommendation: LSTM units should typically be greater than Dense units for optimal representation.')

    col_btn, _ = get_columns(2)
    with col_btn:
        start_training = st.sidebar.button(label='Start Model Training', help='Train model using configured parameters')

    if start_training:
        try:
            num_tickers = len(constants.TICKER_SET)
            cols = get_columns(num_tickers)
            for i, tick in enumerate(constants.TICKER_SET):
                with cols[i]:
                    try:
                        info = yf.Ticker(tick).info
                        st.caption(f"**{tick}**")
                        if 'previousClose' in info and 'currency' in info:
                            st.write(f"Prev Close: {info['previousClose']} {info['currency']}")
                    except Exception:
                        st.caption(f"**{tick}** (Info unavailable)")

            placeholder = st.empty()
            with st.spinner("Training Bidirectional LSTM Model in progress..."):
                fig = Train.trainModel()
            
            if fig:
                placeholder.plotly_chart(fig, use_container_width=True)
                st.success(f"Model successfully trained and saved as `{constants.NAME}`!")
            else:
                st.error("Training completed but chart generation failed.")

        except ValueError as ve:
            st.error(f"Value Error: Please verify ticker symbols: {ve}")
        except KeyError as ke:
            st.error(f"Key Error: Missing expected market data field: {ke}")
        except Exception as e:
            st.error(f"An error occurred during training: {e}")
