# External Imports
import os
import sys
import tempfile
from pathlib import Path
import streamlit as st

# Ensure application directory is in sys.path
app_dir = str(Path(__file__).resolve().parent)
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# Set writable MPLCONFIGDIR for matplotlib to avoid permission warnings
mpl_cache_dir = Path(tempfile.gettempdir()) / 'matplotlib'
mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(mpl_cache_dir))

# Internal Imports
from Project.Pages.mainOverview import mainOverview
from Project.Pages.mainAbout import mainAbout
from Project.Pages.mainTrain import mainTrain
from Project.Pages.mainPredict import mainPredict
from Project.Pages.sentimentalAnalysisOverview import startSentimentalAnalysis
from Project import constants

# Helper function for backwards-compatible expander
def get_expander(label):
    if hasattr(st, "expander"):
        return st.expander(label)
    return st.beta_expander(label)

def display_help_video(video_filename, help_text):
    with get_expander("Help"):
        st.write(help_text)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        video_path = os.path.join(base_dir, 'Project', 'Pages', 'Videos', video_filename)
        if os.path.exists(video_path):
            with open(video_path, 'rb') as videoFile:
                videoBytes = videoFile.read()
                st.video(videoBytes)
        else:
            st.info(f"Help video '{video_filename}' is not available.")

def run_app():
    # Initialize global session state for active stock ticker
    if 'selected_ticker' not in st.session_state:
        st.session_state['selected_ticker'] = 'AAPL'

    # Sidebar component for Main Navigation
    # Controls Overview, Train, Forecast, Sentimental Analysis and About Pages
    st.sidebar.header('Global Stock Price Forecasting')

    mainNavigation = st.sidebar.selectbox(
        label='Navigation',
        options=['Overview', 'Train', 'Forecast', 'Sentimental Analysis', 'About']
    )

    # Routing for Main Navigation
    if mainNavigation == 'Overview':
        display_help_video(
            'Overview-Editted.mp4',
            "Get information, charts and financials of any valid stock simply by entering it's name"
        )
        st.markdown('# Overview')
        mainOverview()

    elif mainNavigation == 'Train':
        display_help_video(
            'Train-Editted.mp4',
            "Create custom machine learning models for specific stocks"
        )
        st.markdown('# Train')
        mainTrain()

    elif mainNavigation == 'Forecast':
        display_help_video(
            'Forecast-Editted.mp4',
            "Forecast the future price for any stock for a period of maximum 10 days"
        )
        st.markdown('# Forecast')
        mainPredict()

    elif mainNavigation == 'Sentimental Analysis':
        display_help_video(
            'SA-Editted.mp4',
            "Get scores based on market sentiment for any stock"
        )
        st.markdown('# Sentimental Analysis')
        startSentimentalAnalysis()

    elif mainNavigation == 'About':
        st.markdown('# About')
        base_dir = os.path.dirname(os.path.abspath(__file__))
        about_html_path = os.path.join(base_dir, 'main_about.html')
        mainAbout(about_html_path)

if __name__ == '__main__':
    run_app()
else:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None:
            run_app()
    except Exception:
        pass
