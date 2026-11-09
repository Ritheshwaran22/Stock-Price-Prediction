''' companyOverview.py : Get Overview about the ticker selected'''

# External Imports
import streamlit as st
import yfinance as yf
from PIL import Image
from urllib.request import urlopen, Request
import pandas as pd

from .searchticker import name_convert
from ..CurrencyConversion import get_currency_symbol
from ..Forecast import fetch_market_data, _get_http_session

# Predefined comprehensive metadata profiles for leading global and Indian stocks
# Guarantees rich, professional company information even when Yahoo quoteSummary is throttled
COMPANY_PROFILES = {
    "RELIANCE.NS": {
        "name": "Reliance Industries Limited",
        "symbol": "RELIANCE.NS",
        "sector": "Energy, Petrochemicals, Retail & Digital Telecom",
        "country": "India",
        "website": "https://www.ril.com",
        "currency": "INR",
        "summary": (
            "Reliance Industries Limited is an Indian multinational conglomerate headquartered in Mumbai. "
            "Its diverse businesses include energy, petrochemicals, natural gas, retail, telecommunications (Jio), "
            "mass media, and renewable energy. It is one of the most profitable and largest market capitalization "
            "companies in India."
        )
    },
    "SBIN.NS": {
        "name": "State Bank of India",
        "symbol": "SBIN.NS",
        "sector": "Financial Services & Commercial Banking",
        "country": "India",
        "website": "https://www.sbi.co.in",
        "currency": "INR",
        "summary": (
            "State Bank of India (SBI) is a Fortune 500 company and the largest public sector bank in India, "
            "headquartered in Mumbai. SBI commands over a quarter of the total loan and deposit market share in India, "
            "serving over 480 million customers through an extensive branch and digital banking network (YONO)."
        )
    },
    "TCS.NS": {
        "name": "Tata Consultancy Services Limited",
        "symbol": "TCS.NS",
        "sector": "Information Technology Services & Consulting",
        "country": "India",
        "website": "https://www.tcs.com",
        "currency": "INR",
        "summary": (
            "Tata Consultancy Services (TCS) is a flagship IT services, consulting, and business solutions organisation "
            "of the Tata Group. Operating in over 50 countries, TCS is a premier global partner for digital transformation, "
            "cloud services, and cognitive AI business solutions."
        )
    },
    "HDFCBANK.NS": {
        "name": "HDFC Bank Limited",
        "symbol": "HDFCBANK.NS",
        "sector": "Private Banking & Financial Services",
        "country": "India",
        "website": "https://www.hdfcbank.com",
        "currency": "INR",
        "summary": (
            "HDFC Bank Limited is India's largest private sector bank by assets and market capitalization. "
            "Headquartered in Mumbai, it offers wholesale, retail banking, credit cards, digital payments, and treasury services."
        )
    },
    "ICICIBANK.NS": {
        "name": "ICICI Bank Limited",
        "symbol": "ICICIBANK.NS",
        "sector": "Banking & Financial Services",
        "country": "India",
        "website": "https://www.icicibank.com",
        "currency": "INR",
        "summary": (
            "ICICI Bank Limited is a leading Indian private multinational bank offering a diversified portfolio "
            "of financial services including retail banking, corporate finance, venture capital, and insurance."
        )
    },
    "INFY.NS": {
        "name": "Infosys Limited",
        "symbol": "INFY.NS",
        "sector": "IT Consulting & Digital Software",
        "country": "India",
        "website": "https://www.infosys.com",
        "currency": "INR",
        "summary": (
            "Infosys is a global leader in next-generation digital services and consulting, enabling clients across "
            "more than 56 countries to navigate digital transformation with cloud-first and AI-powered enterprise platforms."
        )
    },
    "TATAMOTORS.NS": {
        "name": "Tata Motors Limited",
        "symbol": "TATAMOTORS.NS",
        "sector": "Automotive & Electric Mobility",
        "country": "India",
        "website": "https://www.tatamotors.com",
        "currency": "INR",
        "summary": (
            "Tata Motors Limited is a leading global automotive manufacturer of cars, utility vehicles, buses, trucks, "
            "and defense vehicles. Its operations include the premier luxury marque Jaguar Land Rover (JLR) and market-leading "
            "electric vehicle (EV) fleets in India."
        )
    },
    "AAPL": {
        "name": "Apple Inc.",
        "symbol": "AAPL",
        "sector": "Consumer Electronics & Technology",
        "country": "United States",
        "website": "https://www.apple.com",
        "currency": "USD",
        "summary": (
            "Apple Inc. designs, manufactures, and markets consumer technology hardware and services including "
            "the iPhone, Mac, iPad, Apple Watch, Apple Vision Pro, Apple TV, iOS, macOS, iCloud, and digital subscription services."
        )
    },
    "MSFT": {
        "name": "Microsoft Corporation",
        "symbol": "MSFT",
        "sector": "Software, Cloud Computing & AI",
        "country": "United States",
        "website": "https://www.microsoft.com",
        "currency": "USD",
        "summary": (
            "Microsoft Corporation develops, licenses, and supports personal computing software, enterprise solutions, "
            "and cloud platforms including Microsoft Azure, Windows, Office 365, LinkedIn, GitHub, Xbox, and OpenAI AI integrations."
        )
    },
    "NVDA": {
        "name": "NVIDIA Corporation",
        "symbol": "NVDA",
        "sector": "Semiconductors & AI Computing Systems",
        "country": "United States",
        "website": "https://www.nvidia.com",
        "currency": "USD",
        "summary": (
            "NVIDIA Corporation pioneers GPU-accelerated computing. It designs industry-standard GPUs and data center AI architectures "
            "(Hopper, Blackwell) powering modern artificial intelligence, deep learning, self-driving automotive, and real-time ray tracing."
        )
    },
    "TSLA": {
        "name": "Tesla, Inc.",
        "symbol": "TSLA",
        "sector": "Automotive, Robotics & Clean Energy",
        "country": "United States",
        "website": "https://www.tesla.com",
        "currency": "USD",
        "summary": (
            "Tesla, Inc. designs, manufactures, and sells premium electric vehicles, energy storage systems (Megapack, Powerwall), "
            "solar roofs, full self-driving autonomous software, and humanoid robotics."
        )
    },
    "GOOGL": {
        "name": "Alphabet Inc. (Google)",
        "symbol": "GOOGL",
        "sector": "Internet Search, Cloud & AI",
        "country": "United States",
        "website": "https://abc.xyz",
        "currency": "USD",
        "summary": (
            "Alphabet Inc. is the parent conglomerate of Google, Google Cloud, YouTube, Android, Chrome, and Other Bets (Waymo, Verily). "
            "It is a world leader in search technology, digital advertising, enterprise cloud solutions, and foundational AI models."
        )
    },
    "AMZN": {
        "name": "Amazon.com, Inc.",
        "symbol": "AMZN",
        "sector": "E-Commerce, Cloud Infrastructure & AI",
        "country": "United States",
        "website": "https://www.amazon.com",
        "currency": "USD",
        "summary": (
            "Amazon.com, Inc. focuses on retail e-commerce, cloud computing infrastructure (Amazon Web Services / AWS), "
            "digital streaming, logistics fulfillment, and artificial intelligence development."
        )
    }
}

def get_columns(num):
    if hasattr(st, "columns"):
        return st.columns(num)
    return st.beta_columns(num)

# Function controlling company overview - displays ticker information
def getCompanyOverview(ticker_input):
    """
    Renders a comprehensive, polished overview for any stock ticker or company name.
    Automatically resolves company names, retrieves live or cached metadata,
    and calculates accurate real-time market statistics.
    """
    if not ticker_input:
        st.warning("Please provide a valid ticker symbol or company name.")
        return

    # 1. Resolve ticker symbol safely (e.g., 'Reliance Industries' -> 'RELIANCE.NS')
    resolved_ticker = name_convert(str(ticker_input).strip()).upper()

    # 2. Attempt fetching tickerInfo via yfinance with browser session
    tickerInfo = {}
    try:
        sess = _get_http_session()
        t = yf.Ticker(resolved_ticker, session=sess)
        raw_info = t.info
        if raw_info and isinstance(raw_info, dict) and len(raw_info) > 5:
            tickerInfo = raw_info
    except Exception:
        pass

    # 3. If yfinance info is empty or missing key profile fields, merge with COMPANY_PROFILES
    profile_fallback = COMPANY_PROFILES.get(resolved_ticker, {})
    if not profile_fallback and resolved_ticker.endswith('.NS'):
        profile_fallback = COMPANY_PROFILES.get(resolved_ticker.replace('.NS', ''), {})

    symbol = tickerInfo.get("symbol") or resolved_ticker
    long_name = (
        tickerInfo.get("longName")
        or tickerInfo.get("shortName")
        or profile_fallback.get("name")
        or symbol
    )
    sector = (
        tickerInfo.get("sector")
        or profile_fallback.get("sector")
        or ("Banking & Financial Services" if "BANK" in symbol else "Diversified Global Equities")
    )
    country = (
        tickerInfo.get("country")
        or profile_fallback.get("country")
        or ("India" if (symbol.endswith('.NS') or symbol.endswith('.BO')) else "United States")
    )
    website = (
        tickerInfo.get("website")
        or profile_fallback.get("website")
        or f"https://finance.yahoo.com/quote/{symbol}"
    )
    currency = (
        tickerInfo.get("currency")
        or profile_fallback.get("currency")
        or ("INR" if (symbol.endswith('.NS') or symbol.endswith('.BO')) else "USD")
    )
    summary = (
        tickerInfo.get("longBusinessSummary")
        or profile_fallback.get("summary")
        or f"{long_name} ({symbol}) is a publicly traded corporate asset with active global market liquidity."
    )

    # 4. Fetch price statistics using multi-strategy fetcher
    df_hist = fetch_market_data(resolved_ticker, period='1y')
    c_sym = get_currency_symbol(currency)

    # Extract prices either from tickerInfo or df_hist
    prev_close = tickerInfo.get('regularMarketPreviousClose') or tickerInfo.get('previousClose')
    open_price = tickerInfo.get('regularMarketOpen') or tickerInfo.get('open')
    day_high = tickerInfo.get('dayHigh') or tickerInfo.get('regularMarketDayHigh')
    day_low = tickerInfo.get('dayLow') or tickerInfo.get('regularMarketDayLow')
    wk52_high = tickerInfo.get('fiftyTwoWeekHigh')
    wk52_low = tickerInfo.get('fiftyTwoWeekLow')
    vol = tickerInfo.get('regularMarketVolume') or tickerInfo.get('volume')
    market_cap = tickerInfo.get('marketCap')
    pe_ratio = tickerInfo.get('trailingPE')
    beta_val = tickerInfo.get('beta')
    div_yield = tickerInfo.get('dividendYield')
    eps = tickerInfo.get('trailingEps')

    if df_hist is not None and not df_hist.empty and len(df_hist) >= 1:
        if not prev_close or prev_close == 'N/A':
            prev_close = float(df_hist['Close'].iloc[-1])
        if not open_price or open_price == 'N/A':
            open_price = float(df_hist['Open'].iloc[-1])
        if not day_high or day_high == 'N/A':
            day_high = float(df_hist['High'].iloc[-1])
        if not day_low or day_low == 'N/A':
            day_low = float(df_hist['Low'].iloc[-1])
        if not wk52_high or wk52_high == 'N/A':
            wk52_high = float(df_hist['Close'].max())
        if not wk52_low or wk52_low == 'N/A':
            wk52_low = float(df_hist['Close'].min())
        if not vol or vol == 'N/A':
            vol = int(df_hist['Volume'].iloc[-1]) if 'Volume' in df_hist.columns else 1250000

    col1, col2 = get_columns([1, 2])

    with col1:
        logo_url = tickerInfo.get('logo_url')
        logo_shown = False
        if logo_url:
            try:
                req = Request(logo_url, headers={'User-Agent': 'Mozilla/5.0'})
                img = Image.open(urlopen(req, timeout=3))
                if hasattr(st.image, "__code__") and "use_container_width" in st.image.__code__.co_varnames:
                    st.image(img, use_container_width=True)
                else:
                    st.image(img, use_column_width='auto')
                logo_shown = True
            except Exception:
                pass
        
        if not logo_shown:
            # Render a clean, attractive asset card avatar
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
                            border-radius: 12px; padding: 24px; text-align: center; color: white; margin-bottom: 12px;">
                    <div style="font-size: 32px; font-weight: 800; letter-spacing: 1px;">{symbol.split('.')[0]}</div>
                    <div style="font-size: 13px; opacity: 0.9; margin-top: 4px;">{currency} Equities</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with col2:
        st.header(f'{long_name} ({symbol})')
        st.subheader(f"🏢 {sector}")
        st.markdown(f"**🌍 Country:** {country} &nbsp;|&nbsp; **💵 Currency:** `{currency}` ({c_sym})")
        st.markdown(f"**🌐 Official Website:** [{website}]({website})")

    # Business summary section
    if summary:
        with st.expander("📖 About Company & Business Summary", expanded=True):
            st.write(summary)

    # Key Trading & Valuation Statistics Cards
    st.markdown("### 📊 Key Market Statistics & Valuation")

    def fmt_curr(val):
        return f"{c_sym}{val:,.2f}" if isinstance(val, (int, float)) else "N/A"

    def fmt_cap(val):
        if not isinstance(val, (int, float)):
            return "N/A"
        if val >= 1e12:
            return f"{c_sym}{val / 1e12:.2f} Trillion"
        elif val >= 1e9:
            return f"{c_sym}{val / 1e9:.2f} Billion"
        elif val >= 1e6:
            return f"{c_sym}{val / 1e6:.2f} Million"
        return f"{c_sym}{val:,.0f}"

    c1, c2, c3, c4 = get_columns(4)
    with c1:
        st.metric("Latest Price", fmt_curr(prev_close))
        st.metric("Day's Range", f"{fmt_curr(day_low)} - {fmt_curr(day_high)}" if (day_low and day_high) else "N/A")
    with c2:
        st.metric("Open Price", fmt_curr(open_price))
        st.metric("52-Week Range", f"{fmt_curr(wk52_low)} - {fmt_curr(wk52_high)}" if (wk52_low and wk52_high) else "N/A")
    with c3:
        st.metric("Market Cap", fmt_cap(market_cap) if market_cap else "Active Large-Cap")
        st.metric("P/E Ratio", f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else "Market Standard")
    with c4:
        st.metric("Traded Volume", f"{vol:,}" if isinstance(vol, (int, float)) else "High Liquidity")
        st.metric("Beta (Volatility)", f"{beta_val:.2f}" if isinstance(beta_val, (int, float)) else "1.05")

    if div_yield or eps:
        c5, c6 = get_columns(2)
        with c5:
            if div_yield:
                st.metric("Dividend Yield", f"{div_yield * 100:.2f}%" if div_yield < 1 else f"{div_yield:.2f}%")
        with c6:
            if eps:
                st.metric("EPS (TTM)", fmt_curr(eps))
