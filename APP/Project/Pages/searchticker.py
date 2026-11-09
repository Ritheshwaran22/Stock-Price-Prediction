# External Imports
import re
try:
    import yfinance as yf
except ImportError:
    yf = None

# Map of common company names to standard tickers for instant offline resolution
COMMON_TICKERS = {
    # Indian Equities (NSE)
    "STATE BANK OF INDIA": "SBIN.NS",
    "SBI": "SBIN.NS",
    "ICICI BANK": "ICICIBANK.NS",
    "ICICI": "ICICIBANK.NS",
    "HDFC BANK": "HDFCBANK.NS",
    "HDFC": "HDFCBANK.NS",
    "RELIANCE": "RELIANCE.NS",
    "RELIANCE INDUSTRIES": "RELIANCE.NS",
    "TATA CONSULTANCY SERVICES": "TCS.NS",
    "TCS": "TCS.NS",
    "INFOSYS": "INFY.NS",
    "TATA MOTORS": "TATAMOTORS.NS",
    "ITC": "ITC.NS",
    "BHARTI AIRTEL": "BHARTIARTL.NS",
    "AIRTEL": "BHARTIARTL.NS",
    "KOTAK MAHINDRA BANK": "KOTAKBANK.NS",
    "KOTAK": "KOTAKBANK.NS",
    "LARSEN & TOUBRO": "LT.NS",
    "LT": "LT.NS",
    "HINDUSTAN UNILEVER": "HINDUNILVR.NS",
    "HUL": "HINDUNILVR.NS",
    "AXIS BANK": "AXISBANK.NS",
    "MARUTI SUZUKI": "MARUTI.NS",
    "MARUTI": "MARUTI.NS",
    "BAJAJ FINANCE": "BAJFINANCE.NS",
    "WIPRO": "WIPRO.NS",
    "HCL TECH": "HCLTECH.NS",
    "ASIAN PAINTS": "ASIANPAINT.NS",
    "SUN PHARMA": "SUNPHARMA.NS",
    "TITAN": "TITAN.NS",
    "ADANI ENTERPRISES": "ADANIENT.NS",
    "ADANI PORTS": "ADANIPORTS.NS",
    "TATA STEEL": "TATASTEEL.NS",
    "NTPC": "NTPC.NS",
    "POWER GRID": "POWERGRID.NS",
    "ONGC": "ONGC.NS",
    "COAL INDIA": "COALINDIA.NS",
    "NIFTY": "^NSEI",
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    # US & Global Equities
    "APPLE": "AAPL",
    "MICROSOFT": "MSFT",
    "GOOGLE": "GOOGL",
    "ALPHABET": "GOOGL",
    "AMAZON": "AMZN",
    "TESLA": "TSLA",
    "NVIDIA": "NVDA",
    "META": "META",
    "FACEBOOK": "META",
    "NETFLIX": "NFLX",
    "ADVANCED MICRO DEVICES": "AMD",
    "AMD": "AMD",
    "INTEL": "INTC",
    "BERKSHIRE HATHAWAY": "BRK-B",
    "JPMORGAN CHASE": "JPM",
    "JPMORGAN": "JPM",
    "VISA": "V",
    "WALMART": "WMT",
    "JOHNSON & JOHNSON": "JNJ",
    "PROCTER & GAMBLE": "PG",
    "MASTERCARD": "MA",
    "HOME DEPOT": "HD",
    "CHEVRON": "CVX",
    "EXXON MOBIL": "XOM",
    "EXXON": "XOM",
    "DISNEY": "DIS",
    "WALT DISNEY": "DIS",
    "COCA COLA": "KO",
    "PEPSI": "PEP",
    "PEPSICO": "PEP",
    "COSTCO": "COST",
    "BROADCOM": "AVGO",
    "ADOBE": "ADBE",
    "SALESFORCE": "CRM",
    "CISCO": "CSCO",
    "ORACLE": "ORCL",
    "QUALCOMM": "QCOM",
    "PFIZER": "PFE",
    "BOEING": "BA",
    "ALIBABA": "BABA",
    "TAIWAN SEMICONDUCTOR": "TSM",
    "TSMC": "TSM",
    "S&P 500": "^GSPC",
    "DOW JONES": "^DJI",
    "NASDAQ": "^IXIC",
    "BITCOIN": "BTC-USD",
    "ETHEREUM": "ETH-USD"
}

def name_convert(company_name: str) -> str:
    """
    Resolves a user-entered company name or symbol to a valid Yahoo Finance ticker.
    Gracefully handles direct tickers, known company mappings, and web fallback without crashing.
    """
    clean_name = company_name.strip()
    if not clean_name:
        return "SBIN.NS"

    upper_name = clean_name.upper()

    # 1. Exact match in common dictionary mapping
    if upper_name in COMMON_TICKERS:
        return COMMON_TICKERS[upper_name]

    # 2. Smart Substring / Alias matching for company names
    # Sort keys by length descending so longer phrases match first
    for key, ticker in sorted(COMMON_TICKERS.items(), key=lambda item: len(item[0]), reverse=True):
        if key in upper_name or upper_name in key:
            return ticker

    # 3. If user entered what looks like a ticker symbol (e.g. AAPL, SBIN.NS, ^NSEI, TSLA)
    if re.match(r'^[A-Za-z0-9\.\-\^]{1,15}$', clean_name):
        return upper_name

    # Optional Google Search fallback with error protection
    try:
        from googlesearch import search
        searchval = f"{clean_name} ticker yahoo finance"
        links = []
        for url in search(searchval, lang='en', num_results=1):
            links.append(url)
            break
        if links:
            url_str = str(links[0]).rstrip('/')
            parts = url_str.split('/')
            candidate = parts[-1]
            if '=' in candidate:
                candidate = candidate.split('=')[-1]
            if candidate and re.match(r'^[A-Za-z0-9\.\-]+$', candidate):
                return candidate.upper()
    except Exception:
        pass

    # Fallback to the uppercase input
    return upper_name
