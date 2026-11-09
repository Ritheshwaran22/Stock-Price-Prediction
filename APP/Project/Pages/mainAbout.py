''' mainAbout.py : Information about the project and authors'''

# External Imports
import os
import streamlit as st
import codecs
import streamlit.components.v1 as stc
from pathlib import Path

def mainAbout(about_html=None, width=700, height=575):
    if about_html is None:
        about_html = Path(__file__).resolve().parent.parent.parent / 'main_about.html'
    
    if os.path.exists(about_html):
        with codecs.open(about_html, 'r', encoding='utf-8') as about_file:
            page = about_file.read()
            stc.html(page, width=width, height=height, scrolling=False)
    else:
        st.info("Stock Market Management System — Interactive Analytics & Forecasting Platform.")

    st.write('\n')
    st.markdown('## Project Architecture')
    st.markdown("""
    - **Frontend:** Streamlit Interactive Financial Dashboard
    - **Market Data:** Yahoo Finance (`yfinance`) API
    - **Deep Learning:** Bidirectional LSTM with customized hyperparameters
    - **Sentiment Engine:** Finviz News Scraper with NLTK VADER Scoring
    - **Visualizations:** Plotly Express & Graph Objects
    """)
    st.write('\n')
    st.markdown('## Repository : [![Github](https://img.shields.io/badge/Made_With-Github-black?style=for-the-badge&logo=Github)](https://github.com/GeezFORCE/MainProject)')
    st.write('\n')
    st.markdown('## License : [![Creative Commons](http://ForTheBadge.com/images/badges/cc-0.svg)](https://creativecommons.org/share-your-work/public-domain/cc0/)')
