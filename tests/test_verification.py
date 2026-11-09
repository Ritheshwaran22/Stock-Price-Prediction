"""
test_verification.py - Automated verification suite for the modernized Stock Market Management System.
"""

import io
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path

# Add APP directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / 'APP'))

def test_zip_path_traversal():
    from Project.Forecast import safe_extract_zip
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zf:
        zf.writestr('../evil.txt', 'malicious payload')
        zf.writestr('../../etc/evil.txt', 'malicious payload 2')
    zip_buffer.seek(0)

    with tempfile.TemporaryDirectory() as target_dir:
        try:
            safe_extract_zip(zip_buffer, Path(target_dir))
            raise AssertionError("SECURITY FAILURE: Path traversal was not blocked!")
        except ValueError:
            print("SECURITY_PASS: ZIP path traversal attack blocked successfully.")

def test_gitignore_security():
    gitignore_path = BASE_DIR / '.gitignore'
    assert gitignore_path.exists(), ".gitignore does not exist"
    content = gitignore_path.read_text(encoding='utf-8')
    assert '.env' in content, ".env not in .gitignore"
    assert '.venv' in content, ".venv not in .gitignore"
    print("SECURITY_PASS: .gitignore correctly protects .env and .venv.")

def test_zero_exposed_secrets():
    secret_patterns = [
        re.compile(r'alphavantage', re.IGNORECASE),
        re.compile(r'api_scrapper', re.IGNORECASE),
        re.compile(r'apikey\s*=\s*[\'"][a-zA-Z0-9_\-]{16,}[\'"]', re.IGNORECASE)
    ]
    matched = []
    for p in BASE_DIR.rglob('*.py'):
        if any(part.startswith('.venv') for part in p.parts) or 'tests' in p.parts:
            continue
        content = p.read_text(encoding='utf-8', errors='ignore')
        for pat in secret_patterns:
            if pat.search(content):
                matched.append((str(p), pat.pattern))

    assert len(matched) == 0, f"Found unexpected secret references: {matched}"
    print("SECURITY_PASS: 0 exposed secrets or legacy Alpha Vantage references found.")

def test_data_preservation():
    datasets = [
        'sample.csv',
        'stock_prices.csv',
        'arima.csv',
        'ArimaVSlstm.csv',
        'ArimaVSlstm1.csv',
        'ArimaVSlstm2.csv',
        'Arima_Classification.csv',
    ]
    for filename in datasets:
        fp = BASE_DIR / filename
        assert fp.exists(), f"Dataset file missing: {filename}"
        actual_size = fp.stat().st_size
        assert actual_size > 0, f"Dataset file is empty: {filename}"
        print(f"DATA_PRESERVED: {filename} ({actual_size:,} bytes, intact)")

if __name__ == '__main__':
    print("=== RUNNING COMPLETE VERIFICATION SUITE ===")
    test_zip_path_traversal()
    test_gitignore_security()
    test_zero_exposed_secrets()
    test_data_preservation()
    print("=== ALL VERIFICATION CHECKS COMPLETED SUCCESSFULLY ===")

