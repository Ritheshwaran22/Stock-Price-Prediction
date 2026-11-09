"""CurrencyConversion.py: Convert Non-USD stock data to USD and vice-versa safely with fallback"""

# External Imports
import yfinance as yf
import numpy as np

# Internal Imports
from . import constants

# Cache to avoid repeated info network calls
_CURRENCY_CACHE = {}

FALLBACK_USD_RATES = {
    'INR': 87.7,
    'EUR': 0.92,
    'GBP': 0.78,
    'CAD': 1.39,
    'AUD': 1.52,
    'JPY': 152.0,
    'CHF': 0.88,
    'CNY': 7.24,
    'HKD': 7.78,
    'USD': 1.0
}

CURRENCY_SYMBOLS = {
    'INR': '₹',
    'USD': '$',
    'EUR': '€',
    'GBP': '£',
    'JPY': '¥',
    'CAD': 'CA$',
    'AUD': 'A$',
    'CHF': 'Fr ',
    'CNY': '¥',
    'HKD': 'HK$'
}

def get_currency_symbol(curr_code: str) -> str:
    if not curr_code:
        return ""
    return CURRENCY_SYMBOLS.get(curr_code.upper(), f"{curr_code} ")

def get_ticker_currency(ticker: str) -> str:
    clean_ticker = ticker.upper()
    if clean_ticker in _CURRENCY_CACHE:
        return _CURRENCY_CACHE[clean_ticker]

    # Fast pattern matching on exchange suffix to avoid blocking network timeouts
    if clean_ticker.endswith('.NS') or clean_ticker.endswith('.BO'):
        _CURRENCY_CACHE[clean_ticker] = 'INR'
        return 'INR'
    elif clean_ticker.endswith('.L'):
        _CURRENCY_CACHE[clean_ticker] = 'GBP'
        return 'GBP'
    elif clean_ticker.endswith('.TO'):
        _CURRENCY_CACHE[clean_ticker] = 'CAD'
        return 'CAD'
    elif '.' not in clean_ticker:
        # Standard US equities (e.g. AAPL, MSFT, GOOG)
        _CURRENCY_CACHE[clean_ticker] = 'USD'
        return 'USD'

    try:
        info = yf.Ticker(clean_ticker).info
        curr = info.get('currency', 'USD')
        _CURRENCY_CACHE[clean_ticker] = curr
        return curr
    except Exception:
        return 'USD'

def convertCurrency(stocks):
    """Converts non-USD equities to USD if currency converter is available, else uses fallback rates."""
    try:
        ticker_levels = stocks.columns.get_level_values(0).unique()
        for ticker in ticker_levels:
            tickerCurrency = get_ticker_currency(ticker)
            if tickerCurrency and tickerCurrency != 'USD':
                conversionRate = None
                try:
                    from py_currency_converter import convert
                    rates = convert(base=tickerCurrency, amount=1, to=['USD'])
                    conversionRate = rates.get('USD')
                except Exception:
                    pass

                if conversionRate is None or conversionRate <= 0:
                    # Fallback rate: if 1 USD = 87.7 INR, then 1 INR = 1 / 87.7 USD
                    usd_val = FALLBACK_USD_RATES.get(tickerCurrency.upper(), 1.0)
                    conversionRate = 1.0 / usd_val if usd_val > 0 else 1.0

                if (ticker, 'Close') in stocks.columns:
                    stocks.loc[:, (ticker, 'Close')] = stocks[ticker]['Close'] * conversionRate
    except Exception:
        pass

    return stocks

def inverseConvertCurrency(val, target_ticker=None):
    """Converts USD predictions back to the local ticker currency with protection against double-conversion."""
    try:
        if target_ticker is None:
            target_ticker = getattr(constants, 'TICKER_TO_PREDICT', 'SBIN.NS')
        earliertickercurrency = get_ticker_currency(target_ticker)
        if earliertickercurrency and earliertickercurrency != 'USD':
            val_arr = np.array(val, dtype=float)

            # Double-conversion protection:
            # If values are already in the scale of the local market (e.g. INR prices > 50), do not re-multiply!
            if earliertickercurrency.upper() == 'INR' and np.all(val_arr > 50.0):
                return val
            if earliertickercurrency.upper() == 'JPY' and np.all(val_arr > 300.0):
                return val

            conversionRate = None
            try:
                from py_currency_converter import convert
                rates = convert(base='USD', amount=1, to=[earliertickercurrency])
                conversionRate = rates.get(earliertickercurrency)
            except Exception:
                pass

            if conversionRate is None or conversionRate <= 0:
                conversionRate = FALLBACK_USD_RATES.get(earliertickercurrency.upper(), 1.0)

            return val_arr * conversionRate
    except Exception:
        pass

    return val
