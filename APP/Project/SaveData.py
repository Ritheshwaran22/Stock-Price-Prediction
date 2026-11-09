"""SaveData.py: Saves data in various formats"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

def get_saved_data_dir() -> Path:
    # Resolve savedData directory relative to APP directory
    app_dir = Path(__file__).resolve().parent.parent
    data_dir = app_dir / 'savedData'
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

def saveCSV(data, filename: str):
    target_dir = get_saved_data_dir()
    data.to_csv(target_dir / f'{filename}.csv')

def saveNumpy(data, filename: str):
    target_dir = get_saved_data_dir()
    np.save(target_dir / f'{filename}.npy', data)
