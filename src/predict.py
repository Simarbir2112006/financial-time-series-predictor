import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
import os
from src.pipeline import HullMasterPipeline
from src.utils import load_artifact, allocation_logic
from src.config import TRAIN_FILE, DATA_DIR

# Global Buffer
HISTORY_BUFFER = pd.DataFrame()

def initialize_buffer():
    global HISTORY_BUFFER
    print(f"[Predict] Warming up buffer with data from {TRAIN_FILE}...")
    
    if os.path.exists(TRAIN_FILE):
        train = pd.read_csv(TRAIN_FILE)
        train = train.sort_values('date_id')
        
        HISTORY_BUFFER = train.iloc[-500:].copy()
        print(f"[Predict] Buffer warmed up. Current size: {len(HISTORY_BUFFER)} rows.")
    else:
        print("[Warning] Train file not found! Predictions for first 60 rows will be garbage (NaNs).")

def predict_next_batch(new_data: pd.DataFrame):
    """
    Simulates receiving a new batch of market data.
    """
    global HISTORY_BUFFER
    
    # 1. Load Artifacts (Load once in production, but okay here)
    model = load_artifact('hull_model.pkl')
    selected_features = load_artifact('selected_features.pkl')
    pipeline = HullMasterPipeline()

    # 2. Append New Data to History
    # We must ensure new_data matches the columns of HISTORY_BUFFER
    HISTORY_BUFFER = pd.concat([HISTORY_BUFFER, new_data], axis=0, ignore_index=True)
    
    # Keep buffer manageable (last 600 to be safe)
    if len(HISTORY_BUFFER) > 600:
        HISTORY_BUFFER = HISTORY_BUFFER.iloc[-600:]

    # 3. Feature Engineering
    # CRITICAL: We run the pipeline on the WHOLE buffer to get rolling stats correct
    buffer_eng = pipeline.run(HISTORY_BUFFER.copy())
    
    # 4. Slice out ONLY the new rows for prediction
    num_new = len(new_data)
    
    # We take the last 'num_new' rows
    current_features = buffer_eng.iloc[-num_new:][selected_features]
    
    # 5. Predict
    # Handle NaNs just in case (replace with 0 or mean) to prevent crashing
    current_features = current_features.fillna(0)
    
    pred_return = model.predict(current_features)
    
    # 6. Allocation Logic
    allocations = []
    for i in range(num_new):
        # We need to access the specific row index in buffer_eng corresponding to the prediction
        # buffer_eng index is 0 to N. The new rows are at the end.
        idx = len(buffer_eng) - num_new + i
        
        if 'V_mean_index' in buffer_eng.columns:
            vol = buffer_eng.iloc[idx]['V_mean_index']
        else:
            vol = 0.01 
            
        alloc = allocation_logic(pred_return[i], vol)
        allocations.append(alloc)

    return allocations

if __name__ == "__main__":
    # --- 1. INITIALIZE CONTEXT ---
    initialize_buffer()

    # --- 2. RUN REAL PREDICTION ---
    try:
        test_path = os.path.join(DATA_DIR, 'test.csv')
        
        if os.path.exists(test_path):
            print(f">>> READING TEST DATA FROM {test_path} <<<")
            test_data = pd.read_csv(test_path)
            
            print(f"Running prediction on {len(test_data)} rows...")
            
            # We pass the whole test file. 
            # In a real API, this would be one row at a time.
            full_allocations = predict_next_batch(test_data)
            
            print(f"Predictions generated successfully.")
            print(f"Sample (First 5): {full_allocations[:5]}")
            print(f"Sample (Last 5):  {full_allocations[-5:]}")
            
            # Save results if you want
            # pd.DataFrame({'prediction': full_allocations}).to_csv('submission.csv', index=False)
            
        else:
            print("test.csv not found in data folder.")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()