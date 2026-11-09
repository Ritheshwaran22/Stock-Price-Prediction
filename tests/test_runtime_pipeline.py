"""
tests/test_runtime_pipeline.py: Automated verification of real model loading, 
inference, autoregressive forecasting, chart generation, and module integrity
using D:\\Stock_PriceCode\\.venv interpreter.
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Ensure APP directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BASE_DIR / 'APP'
sys.path.insert(0, str(APP_DIR))

import tensorflow as tf
from Project import constants
from Project.Forecast import parse_model_metadata, getModelPrediction, getStockChart
from Project.ScaleData import is_scaler_compatible, inverseScaleForecast, ScaleForecast
from Project.CurrencyConversion import get_ticker_currency, get_currency_symbol, inverseConvertCurrency, convertCurrency
from Project.DatasetCreation import initializeForecast, createDataset

class TestRuntimePipeline(unittest.TestCase):

    def test_import_every_application_module(self):
        """Verifies that all application modules import without error."""
        modules_to_test = [
            'app',
            'Project.constants',
            'Project.DataFetch',
            'Project.DatasetCreation',
            'Project.ModelDefenition',
            'Project.Prediction',
            'Project.SaveModel',
            'Project.ScaleData',
            'Project.plot',
            'Project.CurrencyConversion',
            'Project.Forecast',
            'Project.FinvizSentimentalAnalysis',
            'Project.Pages.mainOverview',
            'Project.Pages.companyOverview',
            'Project.Pages.financialsOverview',
            'Project.Pages.stockCharts',
            'Project.Pages.searchticker',
            'Project.Pages.mainTrain',
            'Project.Pages.mainPredict',
            'Project.Pages.sentimentalAnalysisOverview',
            'Project.Pages.mainAbout',
        ]
        for mod_name in modules_to_test:
            mod = __import__(mod_name, fromlist=['*'])
            self.assertIsNotNone(mod, f"Failed to import {mod_name}")

    def test_real_saved_model_inference(self):
        """REAL MODEL INFERENCE (Not Mocked): Loads real JSON architecture and H5 weights."""
        json_path = APP_DIR / 'savedModelInformation' / 'modelInformation' / 'SBIN.NS-SEQ-ICICIBANK.NS-SBIN.NS-PRED-1637590255.json'
        weights_path = APP_DIR / 'savedModelInformation' / 'modelWeights' / 'SBIN.NS-SEQ-ICICIBANK.NS-SBIN.NS-PRED-1637590255.h5'

        self.assertTrue(json_path.exists(), "Model JSON architecture missing")
        self.assertTrue(weights_path.exists(), "Model H5 weights missing")

        with open(json_path, 'r') as f:
            model = tf.keras.models.model_from_json(f.read())
        model.load_weights(str(weights_path))

        # Expected input shape: (None, 1, 4)
        input_shape = model.input_shape
        self.assertEqual(input_shape, (None, 1, 4))

        # Execute real forward pass with calibrated 4-feature tensor
        dummy_input = np.array([[[0.1, 0.2, 0.3, 0.4]]], dtype=np.float32)
        real_output = model.predict(dummy_input, verbose=0)
        self.assertEqual(real_output.shape, (1, 1))
        self.assertFalse(np.isnan(real_output[0, 0]), "Model prediction returned NaN")

    def test_autoregressive_forecasting_pipeline(self):
        """Verifies multi-step autoregressive forecasting loop with strict feature ordering."""
        json_path = APP_DIR / 'savedModelInformation' / 'modelInformation' / 'SBIN.NS-SEQ-ICICIBANK.NS-SBIN.NS-PRED-1637590255.json'
        weights_path = APP_DIR / 'savedModelInformation' / 'modelWeights' / 'SBIN.NS-SEQ-ICICIBANK.NS-SBIN.NS-PRED-1637590255.h5'

        meta = parse_model_metadata(str(json_path))
        self.assertEqual(meta['target'], 'SBIN.NS')
        self.assertIn('ICICIBANK.NS', meta['tickers'])

        constants.TICKER_TO_PREDICT = meta['target']
        constants.TICKER_SET = meta['tickers']

        with open(json_path, 'r') as f:
            model = tf.keras.models.model_from_json(f.read())
        model.load_weights(str(weights_path))

        # Synthetic 20 trading days in USD matching SBIN.NS ~1006 INR ($11.47 USD)
        dates = pd.date_range('2026-10-01', periods=20, freq='B')
        cols = pd.MultiIndex.from_tuples([
            ('ICICIBANK.NS', 'Close'), ('ICICIBANK.NS', 'Volume'),
            ('SBIN.NS', 'Close'), ('SBIN.NS', 'Volume')
        ], names=['Ticker', 'Price'])

        icici_c = np.linspace(14.5, 14.8, 20)
        icici_v = np.full(20, 10000000)
        sbin_c = np.linspace(11.2, 11.47, 20)
        sbin_v = np.full(20, 15000000)

        data = np.column_stack([icici_c, icici_v, sbin_c, sbin_v])
        stocks = pd.DataFrame(data, index=dates, columns=cols)

        # 3-step autoregressive loop
        sim_stocks = stocks.copy()
        preds_usd = []
        calibrated_flags = []

        for step in range(3):
            ypred_inv, was_cal = getModelPrediction(
                model, sim_stocks, model_name=json_path.stem, target_ticker=meta['target']
            )
            calibrated_flags.append(was_cal)
            pred_usd = float(np.squeeze(ypred_inv[-1]))
            preds_usd.append(pred_usd)

            last_row = sim_stocks.tail(1).copy()
            last_row.loc[:, (meta['target'], 'Close')] = pred_usd
            sim_stocks = pd.concat([sim_stocks, last_row])

        self.assertEqual(len(preds_usd), 3)
        # Predictions in USD should stay in realistic range (e.g. $10 - $13)
        for p in preds_usd:
            self.assertGreater(p, 8.0)
            self.assertLess(p, 16.0)

        # Limitation transparency check: legacy model without preserved scaler must flag calibration
        self.assertTrue(any(calibrated_flags), "Dynamic calibration flag must be True for legacy unscaled model")

    def test_combined_chart_generation(self):
        """Verifies Plotly interactive chart structure with historical and forecast traces."""
        hist_dates = ['2026-10-01', '2026-10-02', '2026-10-03']
        hist_prices = [1000.0, 1005.0, 1006.40]
        forecast_dates = ['2026-10-06', '2026-10-07', '2026-10-08']
        forecast_prices = [1002.50, 1008.00, 1010.20]

        fig = getStockChart(hist_dates, hist_prices, forecast_dates, forecast_prices, 'SBIN.NS', '₹')
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.data), 2)  # Historical trace + Forecast trace
        self.assertEqual(fig.data[0].name, 'Historical Actual')
        self.assertIn('Forecast', fig.data[1].name)
        # Verify connecting bridge starts from last historical point
        self.assertEqual(fig.data[1].x[0], hist_dates[-1])
        self.assertEqual(fig.data[1].y[0], hist_prices[-1])

    def test_currency_conversion_offline_resilience(self):
        """Verifies currency conversion fallbacks when external APIs are unreachable."""
        self.assertEqual(get_ticker_currency('SBIN.NS'), 'INR')
        self.assertEqual(get_ticker_currency('RELIANCE.BO'), 'INR')
        self.assertEqual(get_ticker_currency('VOD.L'), 'GBP')
        self.assertEqual(get_ticker_currency('SHOP.TO'), 'CAD')
        self.assertEqual(get_ticker_currency('AAPL'), 'USD')

        # Test USD to INR inverse conversion
        val_inr = inverseConvertCurrency(10.0)
        self.assertAlmostEqual(val_inr, 877.0, delta=50.0)

        # Test double-conversion protection (SBI price already in INR must not be re-multiplied)
        sbin_already_inr = 1033.40
        safe_val = inverseConvertCurrency(sbin_already_inr, target_ticker='SBIN.NS')
        self.assertEqual(safe_val, 1033.40, "Double-conversion bug: INR price was re-multiplied by exchange rate!")

    def test_single_stock_saved_model_inference(self):
        """Verifies that modern 2-feature single-stock models load and infer cleanly."""
        json_path = APP_DIR / 'savedModelInformation' / 'modelInformation' / 'AAPL-SEQ-AAPL-PRED-1700000001.json'
        weights_path = APP_DIR / 'savedModelInformation' / 'modelWeights' / 'AAPL-SEQ-AAPL-PRED-1700000001.h5'

        self.assertTrue(json_path.exists())
        self.assertTrue(weights_path.exists())

        with open(json_path, 'r') as f:
            model = tf.keras.models.model_from_json(f.read())
        model.load_weights(str(weights_path))

        self.assertEqual(model.input_shape, (None, 1, 2))
        dummy_input = np.array([[[0.5, 0.5]]], dtype=np.float32)
        out = model.predict(dummy_input, verbose=0)
        self.assertEqual(out.shape, (1, 1))
        self.assertFalse(np.isnan(out[0, 0]))

if __name__ == '__main__':
    unittest.main()

