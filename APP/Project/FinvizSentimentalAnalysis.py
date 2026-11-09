"""FinvizSentimentalAnalysis.py: Performs sentiment analysis on news from Finviz.com"""

import os
import pandas as pd
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from datetime import date as dt_date

# NLTK imports and safe Initialization
import nltk

def ensure_vader_downloaded():
    """Ensures VADER lexicon is downloaded once safely without repeated downloads."""
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        SentimentIntensityAnalyzer()
    except LookupError:
        try:
            nltk.download('vader_lexicon', quiet=True)
        except Exception:
            pass

ensure_vader_downloaded()
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Internal Imports
from . import constants
from . import SaveData

def finvizSentimentalAnalysis(tickers):
    """
    Scrapes headlines for specified tickers from Finviz, performs VADER sentiment scoring,
    and returns aggregated average compound sentiment scores.
    """
    newsTables = {}
    user_agent = os.environ.get('FINVIZ_USER_AGENT', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')

    for ticker in tickers:
        clean_ticker = ticker.strip().upper()
        if not clean_ticker:
            continue
        url = constants.FINVIZ_URL + clean_ticker
        try:
            req = Request(url=url, headers={'User-Agent': user_agent})
            with urlopen(req, timeout=10) as response:
                html = BeautifulSoup(response, 'lxml')
                table = html.find(id='news-table')
                if table:
                    newsTables[clean_ticker] = table
        except Exception as e:
            # Continue gracefully if a single ticker fails to scrape
            continue

    if not newsTables:
        return pd.Series(dtype=float)

    parsedData = []

    for ticker, newsTable in newsTables.items():
        current_date = dt_date.today()
        for row in newsTable.findAll('tr'):
            if not row or not row.a or not row.td:
                continue
            newsTitle = row.a.text.strip()
            dateData = row.td.text.strip().split(' ')

            if len(dateData) == 1:
                time = dateData[0]
                date_val = current_date
            else:
                raw_date = dateData[0]
                time = dateData[1]
                try:
                    if raw_date.lower() == 'today':
                        date_val = dt_date.today()
                    else:
                        date_val = pd.to_datetime(raw_date).date()
                    current_date = date_val
                except Exception:
                    date_val = current_date

            parsedData.append([ticker, date_val, time, newsTitle])

    if not parsedData:
        return pd.Series(dtype=float)

    news = pd.DataFrame(parsedData, columns=['Ticker', 'Date', 'Time', 'News Title'])

    vader = SentimentIntensityAnalyzer()
    news['Compound'] = news['News Title'].apply(
        lambda title: vader.polarity_scores(title)['compound']
    )

    try:
        SaveData.saveCSV(news, "FinvizNews")
    except Exception:
        pass

    # Group by Ticker and Date with numeric_only=True for Pandas 2.x compatibility
    meanNewsData = news.groupby(['Ticker', 'Date'], as_index=False)[['Compound']].mean()
    tickerMeanScores = meanNewsData.groupby('Ticker')['Compound'].mean() * 100
    return tickerMeanScores
