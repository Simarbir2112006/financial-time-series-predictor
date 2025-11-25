import numpy as np
import joblib
import os
from src.config import ARTIFACTS_DIR, TARGET_VOL_DAILY

def allocation_logic(pred_return, current_volatility):
    """Calculates the portfolio allocation based on prediction and vol."""
    vol = max(current_volatility, 0.001) 
    base_allocation = TARGET_VOL_DAILY / vol
    final_alloc = base_allocation if pred_return > 0 else 0.0
    return np.clip(final_alloc, 0.0, 2.0)

def save_artifact(obj, filename):
    """Saves a python object (model/list) to artifacts."""
    path = os.path.join(ARTIFACTS_DIR, filename)
    joblib.dump(obj, path)
    print(f"[Utils] Saved {filename} to artifacts.")

def load_artifact(filename):
    """Loads a python object from artifacts."""
    path = os.path.join(ARTIFACTS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{filename} not found in {ARTIFACTS_DIR}")
    return joblib.load(path)