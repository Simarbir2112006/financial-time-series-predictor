import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
from src.config import (TRAIN_FILE, WINNING_WEIGHTS, WINNING_LGBM_PARAMS, 
                        WINNING_RIDGE_ALPHA)
from src.pipeline import HullMasterPipeline, select_features
from src.model import HullBiHybridRegressor
from src.utils import save_artifact

def main():
    print(">>> STARTING TRAINING PIPELINE <<<")
    
    # 1. Load Data
    print(f"[Train] Loading data from {TRAIN_FILE}...")
    try:
        train = pd.read_csv(TRAIN_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find {TRAIN_FILE}. Please put data in the data/ folder.")
        return

    train = train.sort_values('date_id').reset_index(drop=True)

    # 2. Feature Engineering
    print("[Train] Running Feature Engineering Pipeline...")
    pipeline = HullMasterPipeline()
    train_eng = pipeline.run(train)

    # 3. Feature Selection
    print("[Train] Selecting Features...")
    selected_features = select_features(train_eng, top_k=60)
    print(f"[Train] Selected {len(selected_features)} features.")

    # 4. Train Model
    print("[Train] Fitting Bi-Hybrid Model...")
    model = HullBiHybridRegressor(
        weights=WINNING_WEIGHTS,
        lgbm_params=WINNING_LGBM_PARAMS,
        ridge_alpha=WINNING_RIDGE_ALPHA
    )
    
    X = train_eng[selected_features]
    y = train_eng['forward_returns']
    model.fit(X, y)

    # 5. Save Artifacts
    print("[Train] Saving Artifacts...")
    save_artifact(model, 'hull_model.pkl')
    save_artifact(selected_features, 'selected_features.pkl')

    print(">>> TRAINING COMPLETE. READY FOR PREDICTION. <<<")

if __name__ == "__main__":
    main()