''' financialsOverview.py : Get Quarterly financials, major and institutional 
                            holders about the ticker selected'''

# External Imports
import streamlit as st
import yfinance as yf
import plotly.express as px
import pandas as pd

# Function to get financials related to ticker
def getFinancialsOverview(ticker):
    st.header('Financials')

    try:
        t = yf.Ticker(ticker)

        st.subheader('Major Holders')
        majorHolders = t.major_holders
        if majorHolders is not None and not majorHolders.empty:
            # Handle different versions of yfinance major_holders structure
            if len(majorHolders.columns) >= 2:
                col_val = majorHolders.columns[0]
                col_name = majorHolders.columns[1]
                # Convert percentage strings to floats if needed
                vals = majorHolders[col_val].astype(str).str.rstrip('%').astype(float, errors='ignore')
                st.dataframe(majorHolders, use_container_width=True)
                try:
                    fig_pie = px.pie(majorHolders, values=vals, names=majorHolders[col_name], title="Major Holders Breakdown")
                    st.plotly_chart(fig_pie, use_container_width=True)
                except Exception:
                    pass
            else:
                st.dataframe(majorHolders, use_container_width=True)
        else:
            st.info("Major holders data not available for this ticker.")

        st.subheader('Institutional Holders')
        institutionalHolders = t.institutional_holders
        if institutionalHolders is not None and not institutionalHolders.empty:
            st.dataframe(institutionalHolders, use_container_width=True)
            if 'Holder' in institutionalHolders.columns and 'Shares' in institutionalHolders.columns:
                try:
                    fig_bar = px.bar(institutionalHolders.head(10), x='Holder', y='Shares', title="Top Institutional Holders")
                    st.plotly_chart(fig_bar, use_container_width=True)
                except Exception:
                    pass
        else:
            st.info("Institutional holders data not available for this ticker.")
        
    except (KeyError, AttributeError, ValueError) as e:
        st.error(f"Please provide a Valid Ticker to get Financial info: {e}")
    except Exception as e:
        st.error(f"Unable to fetch financial information: {e}")
