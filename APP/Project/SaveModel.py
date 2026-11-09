"""SaveModel.py: Saves Model in h5 and JSON"""

import os
from pathlib import Path
from . import constants

def get_app_dir() -> Path:
    return Path(__file__).resolve().parent.parent

def saveModelInformation(model):
    app_dir = get_app_dir()
    
    models_dir = app_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    info_dir = app_dir / "savedModelInformation" / "modelInformation"
    info_dir.mkdir(parents=True, exist_ok=True)
    
    weights_dir = app_dir / "savedModelInformation" / "modelWeights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Save complete model in h5 format
    model_path = models_dir / f"{constants.NAME}.h5"
    try:
        model.save(str(model_path), save_format='h5')
    except Exception:
        model.save(str(model_path))

    # Save model architecture JSON if supported by Keras version
    if hasattr(model, 'to_json'):
        try:
            model_json = model.to_json()
            with open(info_dir / f"{constants.NAME}.json", "w") as json_file:
                json_file.write(model_json)
        except Exception:
            pass

    # Save model weights
    weights_path = weights_dir / f"{constants.NAME}.h5"
    model.save_weights(str(weights_path))
