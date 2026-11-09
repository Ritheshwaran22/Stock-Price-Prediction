"""DataFetch.py: Fetches and normalizes stock data from Yahoo Finance"""

# External Imports
import yfinance as yf
import pandas as pd

# Internal Imports
from . import constants
from . import SaveData
from .CurrencyConversion import convertCurrency

def getData(save=False, period=None):
    if period is None:
        period = constants.PERIOD

    seen = set()
    tickers = [t for t in constants.TICKER_SET if not (t in seen or seen.add(t))]
    if not tickers:
        tickers = [constants.TICKER_TO_PREDICT]

    # Fetch data from Yahoo Finance
    stocks = yf.download(
        tickers=tickers,
        period=period,
        interval=constants.INTERVAL,
        group_by='Ticker',
        auto_adjust=True,
        prepost=False,
        threads=True,
        progress=False
    )

    if stocks is None or stocks.empty:
        return pd.DataFrame()

    if save:
        try:
            SaveData.saveCSV(stocks, "OrginalStocks")
        except Exception:
            pass

    # Ensure MultiIndex structure (Ticker, PriceType) consistently
    if not isinstance(stocks.columns, pd.MultiIndex):
        # Single ticker case: create MultiIndex
        single_ticker = tickers[0]
        stocks.columns = pd.MultiIndex.from_product([[single_ticker], stocks.columns])
    elif stocks.columns.names and stocks.columns.names[0] != 'Ticker':
        # Invert levels if needed
        if 'Close' in stocks.columns.levels[0]:
            stocks = stocks.swaplevel(0, 1, axis=1)

    # Retain Close and Volume, drop Low, Open, High safely if they exist
    cols_to_drop = []
    for col in ['Low', 'Open', 'High']:
        if col in stocks.columns.get_level_values(1):
            cols_to_drop.append(col)
    if cols_to_drop:
        stocks = stocks.drop(columns=cols_to_drop, level=1)

    # Impute missing values using forward fill, then backward fill (strictly forward in time)
    stocks = stocks.ffill().bfill()

    # Currency conversion for international equities
    stocks = convertCurrency(stocks)

    if save:
        try:
            SaveData.saveCSV(stocks, "stocks")
        except Exception:
            pass

    return stocks
