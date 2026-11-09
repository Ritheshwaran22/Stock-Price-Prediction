# Stock Market Management System

A modernized, interactive financial analytics and stock market forecasting web platform built on **Streamlit**, powered by **Yahoo Finance (`yfinance`)**, and driven by custom deep learning architectures (Bidirectional LSTM) alongside classical econometric models (ARIMA, CNN, Random Forest, SVM).

---

## 1. Project Overview & Architecture

The Stock Market Management System provides real-time market overviews, interactive technical charts, company financials, news sentiment analysis, custom model training, and forward multi-day autoregressive price forecasting.

### Architecture Diagram
```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Web UI                         │
│               (APP/app.py - Main Entry)                     │
└───────┬──────────────┬──────────────┬──────────────┬────────┘
        │              │              │              │
        ▼              ▼              ▼              ▼
   [Overview]       [Train]       [Forecast]    [Sentiment]
   (yfinance)    (Bi-LSTM Fit)  (Autoregressive)  (Finviz/VADER)
        │              │              │              │
        ├──────────────┴──────────────┴──────────────┘
        ▼
   [Data Layer]
   - DataFetch.py: Yahoo Finance OHLCV Ingestion & Imputation
   - ScaleData.py: Per-Ticker RobustScaler (Zero Data Leakage)
   - DatasetCreation.py: Rolling Temporal Window Sequencing
   - SaveModel.py / SaveData.py: Robust Local Persistence
```

---

## 2. Main Features

1. **Market Overview & Financials:**
   - Real-time quote retrieval, market cap, sector, country, and descriptive business summaries.
   - Interactive Plotly price history charts with interactive time-series rangesliders.
   - Major holders breakdown and institutional share distribution tables.
2. **Interactive Model Training:**
   - On-demand Bidirectional LSTM training for custom target and correlated equity sets.
   - Dynamic hyperparameter configuration: Epochs, Batch Size, Learning Rate, LSTM Units, Dense Units, Layers, and Dropout Rate.
   - Real-time training progress tracking, validation metrics (MAE, RMSE, R² Score), and actual vs. predicted curves.
3. **Autoregressive Multi-Day Price Forecasting:**
   - 1-to-10 day recursive forward forecast with dynamic sequence updates.
   - Comparison tables displaying projected percentage changes and color-coded gains/losses.
   - Pretrained model selection as well as secure upload validation for custom trained models.
4. **Financial News Sentiment Analysis:**
   - Automated financial news headline scraping from Finviz.
   - NLP sentiment intensity scoring using NLTK VADER.
   - Aggregated bullish/bearish/neutral scoring across multiple tickers.
5. **Standalone Econometric & Machine Learning Suite:**
   - Classical ARIMA and Auto-ARIMA time-series models.
   - 1D Convolutional Neural Network (CNN) and hybrid CNN-LSTM models.
   - Random Forest and Support Vector Regression (SVR) baselines.

---

## 3. Technology Stack

* **Frontend:** Streamlit $\ge$ 1.20
* **Data Provider:** Yahoo Finance (`yfinance`)
* **Deep Learning:** TensorFlow / Keras (Bidirectional LSTM, CNN, CNN-LSTM)
* **Classical ML & Statistics:** Scikit-Learn, Statsmodels, pmdarima
* **NLP Sentiment:** NLTK (VADER Lexicon), BeautifulSoup4, lxml
* **Data & Visualizations:** Pandas 2.x, NumPy, Plotly Express & Graph Objects

---

## 4. Setup & Installation

### Prerequisites
* Python 3.9 or 3.10 installed on your system.

### 1. Clone & Navigate to Repository
```bash
git clone <repository-url>
cd Stock_PriceCode
```

### 2. Create and Activate Virtual Environment
On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
On Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Environment Configuration
Copy the sample environment file:
```bash
copy .env.example .env
```
*(No external third-party API keys are required. Market data is fetched dynamically via Yahoo Finance).*

---

## 5. Launching the Application

Run the Streamlit application from the project root:

```bash
streamlit run APP/app.py
```

Once launched, open your web browser to:
```
http://localhost:8501
```

---

## 6. Standalone Machine Learning Experiments

The repository includes standalone scripts for research and comparative benchmarking:
* **`arima.py`:** Autoregressive Integrated Moving Average time-series model on historical Dow Jones data.
* **`ARIMA_Trainin.py`:** Auto-ARIMA stationarity tests, seasonal decomposition, and stacked LSTM benchmarks on SBI stock data.
* **`CNN.py`:** 1D Convolutional Neural Network for daily closing price forecasting.
* **`cnn_LSTM.py`:** TimeDistributed Conv1D feature extraction coupled with recurrent LSTM cells.
* **`Classification.py`:** AR, VAR, Multi-Layer Perceptron (MLP), and LSTM binary price movement classification.
* **`Random_Forest.py`:** Random Forest ensemble regressor with mean absolute error scoring.
* **`svm.py`:** Support Vector Regression (Linear, Polynomial, and RBF kernels).

Run any standalone experiment directly:
```bash
python arima.py
python CNN.py
python Random_Forest.py
```

---

## 7. Running Automated Tests

Run the unit test suite to verify core components, path traversal protections, dataset generation, and symbol validation:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## 8. Security & Best Practices

* **Zero Hardcoded Secrets:** All third-party vendor keys have been permanently removed.
* **Zip Slip Protection:** Safe archive extraction in `APP/Project/Forecast.py` validates member file paths before extraction to prevent directory traversal vulnerabilities.
* **Strict Temporal Splitting:** Preprocessing transformations and scalers are fitted exclusively on chronological training splits to eliminate lookahead data leakage.
* **Standard Relative Paths:** All file reads, dataset saves, and model weight exports use `pathlib.Path` resolution relative to the application directory.

