"""
tests/test_components.py: Unit tests for core modernized components of the Stock Market Management System
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import zipfile

# Import modules from APP
import sys
app_dir = Path(__file__).resolve().parent.parent / 'APP'
sys.path.insert(0, str(app_dir))

from Project.Pages.searchticker import name_convert, COMMON_TICKERS
from Project.DatasetCreation import createDataset
from Project.Forecast import parse_model_metadata
from Project.ScaleData import is_scaler_compatible
from Project.CurrencyConversion import get_ticker_currency, get_currency_symbol, inverseConvertCurrency
from sklearn.preprocessing import RobustScaler

def safe_extract_zip_test(zip_file, target_dir):
    """Mirror implementation for isolation testing."""
    target_path = Path(target_dir).resolve()
    with zipfile.ZipFile(zip_file) as archive:
        for member in archive.infolist():
            member_path = (target_path / member.filename).resolve()
            if not str(member_path).startswith(str(target_path)):
                raise ValueError(f"Security error: Archive member '{member.filename}' attempts path traversal.")
        archive.extractall(target_path)

class TestStockAppCore(unittest.TestCase):

    def test_ticker_name_conversion_known(self):
        """Tests that known common companies resolve to correct tickers instantly."""
        self.assertEqual(name_convert("State Bank of India"), "SBIN.NS")
        self.assertEqual(name_convert("SBI"), "SBIN.NS")
        self.assertEqual(name_convert("Apple"), "AAPL")
        self.assertEqual(name_convert("Microsoft"), "MSFT")
        self.assertEqual(name_convert("Tesla"), "TSLA")
        self.assertEqual(name_convert("NVIDIA"), "NVDA")
        self.assertEqual(name_convert("Reliance"), "RELIANCE.NS")
        self.assertEqual(name_convert("TCS"), "TCS.NS")
        self.assertEqual(name_convert("Infosys"), "INFY.NS")
        self.assertEqual(name_convert("Tata Motors"), "TATAMOTORS.NS")

    def test_ticker_name_conversion_direct_symbol(self):
        """Tests that direct ticker symbols are normalized cleanly."""
        self.assertEqual(name_convert("aapl"), "AAPL")
        self.assertEqual(name_convert("sbin.ns"), "SBIN.NS")
        self.assertEqual(name_convert("^nsei"), "^NSEI")

    def test_create_dataset_dimensions(self):
        """Tests that rolling time series dataset creation preserves strict shapes without leakage."""
        timesteps = 3
        data_len = 20
        num_features = 4
        
        # Synthetic feature dataframe and target series
        features = pd.DataFrame(np.arange(data_len * num_features).reshape(data_len, num_features))
        target = pd.Series(np.arange(data_len))

        X, Y = createDataset(features, target, timeSteps=timesteps)

        self.assertEqual(len(X), data_len - timesteps)
        self.assertEqual(len(Y), data_len - timesteps)
        self.assertEqual(X.shape, (data_len - timesteps, timesteps, num_features))
        # Ensure target Y[0] matches the immediate step after the lookback window
        self.assertEqual(Y[0], target.iloc[timesteps])

    def test_safe_zip_extraction_valid(self):
        """Tests that valid ZIP files extract without security exceptions."""
        with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as dest_dir:
            sample_file = Path(src_dir) / "model.h5"
            sample_file.write_text("model_data")

            zip_path = Path(src_dir) / "valid_model.zip"
            with zipfile.ZipFile(zip_path, 'w') as z:
                z.write(sample_file, arcname="model.h5")

            safe_extract_zip_test(zip_path, dest_dir)
            extracted = Path(dest_dir) / "model.h5"
            self.assertTrue(extracted.exists())

    def test_safe_zip_extraction_path_traversal_blocked(self):
        """Tests that malicious zip files attempting directory traversal (Zip Slip) are blocked."""
        with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as dest_dir:
            zip_path = Path(src_dir) / "malicious.zip"
            with zipfile.ZipFile(zip_path, 'w') as z:
                z.writestr("../../evil.txt", "malicious_content")

            with self.assertRaises(ValueError) as context:
                safe_extract_zip_test(zip_path, dest_dir)
            self.assertIn("Security error", str(context.exception))

    def test_model_metadata_parsing(self):
        """Tests parsing of target ticker, ticker inputs, and timestamp from model filename."""
        sample_name = "SBIN.NS-SEQ-ICICIBANK.NS-SBIN.NS-PRED-1637590255.json"
        meta = parse_model_metadata(sample_name)
        self.assertEqual(meta['target'], "SBIN.NS")
        self.assertIn("ICICIBANK.NS", meta['tickers'])
        self.assertIn("SBIN.NS", meta['tickers'])
        self.assertEqual(meta['timestamp'], 1637590255)

    def test_scaler_compatibility_check(self):
        """Tests that alien scalers (e.g. AAPL) are detected as incompatible for different stocks (e.g. SBIN)."""
        alien_scaler = RobustScaler()
        alien_scaler.fit([[147.0], [148.0], [146.0]])
        sbin_prices = pd.Series([11.2, 11.4, 11.3, 11.5])
        self.assertFalse(is_scaler_compatible(alien_scaler, sbin_prices))

        native_scaler = RobustScaler()
        native_scaler.fit([[11.2], [11.3], [11.4]])
        self.assertTrue(is_scaler_compatible(native_scaler, sbin_prices))

    def test_currency_symbols_and_conversion(self):
        """Tests currency resolution, symbols, and rate conversion logic."""
        self.assertEqual(get_ticker_currency("SBIN.NS"), "INR")
        self.assertEqual(get_currency_symbol("INR"), "₹")
        self.assertEqual(get_ticker_currency("AAPL"), "USD")
        self.assertEqual(get_currency_symbol("USD"), "$")
        # Ensure native INR prices are preserved and not inflated to 93,000
        self.assertEqual(inverseConvertCurrency(1033.40, target_ticker="SBIN.NS"), 1033.40)

if __name__ == '__main__':
    unittest.main()
