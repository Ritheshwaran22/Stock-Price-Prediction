''' companyOverview.py : Get Overview about the ticker selected'''

# External Imports
import streamlit as st
import yfinance as yf
from PIL import Image
from urllib.request import urlopen

def get_columns(num):
    if hasattr(st, "columns"):
        return st.columns(num)
    return st.beta_columns(num)

# Function controlling company overview - displays ticker information
def getCompanyOverview(ticker):
    try:
        tickerInfo = yf.Ticker(ticker).info
        if not tickerInfo or not isinstance(tickerInfo, dict):
            st.warning("Please provide a Valid Ticker!")
            return

        col1, col2 = get_columns(2)

        with col1:
            logo_url = tickerInfo.get('logo_url')
            if logo_url:
                try:
                    img = Image.open(urlopen(logo_url, timeout=3))
                    if hasattr(st.image, "__code__") and "use_container_width" in st.image.__code__.co_varnames:
                        st.image(img, use_container_width=True)
                    else:
                        st.image(img, use_column_width='auto')
                except Exception:
                    st.info("Logo not available for this ticker.")
            else:
                st.info("Logo not available.")

        with col2:
            symbol = tickerInfo.get("symbol", ticker)
            long_name = tickerInfo.get("longName", symbol)
            st.header(f'{long_name} ({symbol})')

            sector = tickerInfo.get('sector')
            if sector:
                st.subheader(sector)
            else:
                st.warning("Sector: Data not available.")

            country = tickerInfo.get('country')
            if country:
                st.write(f"**Country:** {country}")
            else:
                st.warning("Country: Data not available.")

            website = tickerInfo.get('website')
            if website:
                st.write(f"**Website:** {website}")
            else:
                st.warning("Website: Data not available.")

            currency = tickerInfo.get('currency')
            if currency:
                st.write(f"**Currency:** {currency}")
            else:
                st.warning("Currency: Data not available.")

        summary = tickerInfo.get('longBusinessSummary')
        if summary:
            with st.expander("📖 About Company & Business Summary", expanded=False):
                st.write(summary)

        # Key Trading & Valuation Statistics Cards
        st.markdown("### 📊 Key Market Statistics & Valuation")
        
        # Determine currency formatting
        curr = tickerInfo.get('currency', 'USD')
        from ..CurrencyConversion import get_currency_symbol
        c_sym = get_currency_symbol(curr)

        prev_close = tickerInfo.get('regularMarketPreviousClose') or tickerInfo.get('previousClose')
        open_price = tickerInfo.get('regularMarketOpen') or tickerInfo.get('open')
        day_high = tickerInfo.get('dayHigh') or tickerInfo.get('regularMarketDayHigh')
        day_low = tickerInfo.get('dayLow') or tickerInfo.get('regularMarketDayLow')
        wk52_high = tickerInfo.get('fiftyTwoWeekHigh')
        wk52_low = tickerInfo.get('fiftyTwoWeekLow')
        market_cap = tickerInfo.get('marketCap')
        pe_ratio = tickerInfo.get('trailingPE')
        beta_val = tickerInfo.get('beta')
        div_yield = tickerInfo.get('dividendYield')
        vol = tickerInfo.get('regularMarketVolume') or tickerInfo.get('volume')
        eps = tickerInfo.get('trailingEps')

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
            st.metric("Previous Close", fmt_curr(prev_close))
            st.metric("Day's Range", f"{fmt_curr(day_low)} - {fmt_curr(day_high)}" if day_low and day_high else "N/A")
        with c2:
            st.metric("Open Price", fmt_curr(open_price))
            st.metric("52-Week Range", f"{fmt_curr(wk52_low)} - {fmt_curr(wk52_high)}" if wk52_low and wk52_high else "N/A")
        with c3:
            st.metric("Market Cap", fmt_cap(market_cap))
            st.metric("P/E Ratio (Trailing)", f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else "N/A")
        with c4:
            st.metric("Volume", f"{vol:,}" if isinstance(vol, (int, float)) else "N/A")
            st.metric("Beta", f"{beta_val:.2f}" if isinstance(beta_val, (int, float)) else "N/A")

        if div_yield or eps:
            c5, c6 = get_columns(2)
            with c5:
                if div_yield:
                    st.metric("Dividend Yield", f"{div_yield * 100:.2f}%" if div_yield < 1 else f"{div_yield:.2f}%")
            with c6:
                if eps:
                    st.metric("EPS (TTM)", fmt_curr(eps))

    except ValueError:
        st.warning("Please provide a Valid Ticker!")
    except KeyError:
        st.warning("Either Ticker name is too long or that Isn't a valid Ticker!")
    except Exception as e:
        st.error(f"Ticker information could not be retrieved: {e}")
