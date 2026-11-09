''' sentimentalAnalysisOverview.py : Get Overview about sentimental analysis'''

# External Imports
import streamlit as st
import pandas as pd

# Internal Imports
from .. import FinvizSentimentalAnalysis

def get_columns(num):
    if hasattr(st, "columns"):
        return st.columns(num)
    return st.beta_columns(num)

# Function for sentimental analysis page
def startSentimentalAnalysis():
    st.subheader("Financial News Sentiment Analysis")
    st.write("Scrapes recent financial headlines from Finviz.com and computes sentiment scores using NLTK VADER.")
    
    st.sidebar.subheader('Tickers for Sentiment')
    current_active = st.session_state.get('selected_ticker', 'AAPL')
    default_list = f"{current_active}, MSFT, GOOGL" if current_active not in ["MSFT", "GOOGL"] else "AAPL, MSFT, GOOGL"
    userInputTickers = st.sidebar.text_area(
        label='Tickers (Comma separated)',
        value=default_list,
        help='Enter ticker symbols to analyze recent headlines'
    )
    tickers = [i.strip().upper() for i in userInputTickers.split(',') if i.strip()]

    if st.sidebar.button(label='Analyze Sentiment', help='Click to retrieve news headlines and score sentiment'):
        with st.spinner("Fetching headlines and performing sentiment analysis..."):
            try:
                finvizMeanScores = FinvizSentimentalAnalysis.finvizSentimentalAnalysis(tickers)
                
                col1, col2 = get_columns(2)
                with col1:
                    st.markdown('### Finviz Headline Sentiment Scores')
                    if finvizMeanScores is not None and not finvizMeanScores.empty:
                        score_df = pd.DataFrame({
                            'Ticker': finvizMeanScores.index,
                            'Sentiment Score (-100 to +100)': finvizMeanScores.values.round(2)
                        })
                        st.dataframe(score_df, use_container_width=True)

                        for ticker, score in finvizMeanScores.items():
                            sentiment_label = "🟢 Bullish / Positive" if score > 5 else ("🔴 Bearish / Negative" if score < -5 else "⚪ Neutral")
                            st.write(f"**{ticker}:** {score:.2f} ({sentiment_label})")
                    else:
                        st.warning("No recent news headlines could be retrieved from Finviz for the specified tickers.")

                with col2:
                    st.markdown('### Social Media Sentiment (Twitter / X)')
                    st.info("ℹ️ Twitter / X sentiment tracking is currently in development and not yet available. News headline sentiment is displayed on the left.")

            except Exception as e:
                st.error(f"Error performing sentiment analysis: {e}")
