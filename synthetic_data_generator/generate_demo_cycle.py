import pandas as pd
import numpy as np
import os, sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.config import TRAIN_FILE

def generate_market_cycle():
    print(">>> GENERATING DEMO SCENARIO...")
    
    # 1. Get Template from Train Data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    repo_root = os.path.join(base_dir, "..")

    train_file_path = TRAIN_FILE
    if not os.path.isabs(TRAIN_FILE):
        train_file_path = os.path.join(repo_root, TRAIN_FILE)

    print(f"Reading columns from {train_file_path}...")

    if not os.path.exists(train_file_path):
        print("Error: train.csv not found.")
        return
    
    template_df = pd.read_csv(train_file_path, nrows=1)
    columns = template_df.columns
    
    # 2. Define Helper to make segments
    def create_segment(days, vol_level, trend_start, trend_end):
        data = {}
        for col in columns:
            # Default noise
            data[col] = np.random.uniform(0, 0.01, days)
            
            # Specific Overrides
            if col == 'date_id':
                data[col] = [0] * days # Placeholder, will fix later
            elif col.startswith('V'):
                # Add jitter to volatility so it's not a perfectly straight line
                noise = np.random.normal(0, vol_level * 0.1, days)
                data[col] = np.clip([vol_level] * days + noise, 0.001, 1.0)
            elif col.startswith('M') or col.startswith('P'):
                data[col] = np.linspace(trend_start, trend_end, days)
                
        return pd.DataFrame(data)

    # 3. BUILD THE STORY
    
    # Phase 1: BULL MARKET (50 Days)
    # Low Vol (0.005), Price goes 100 -> 120
    # Expected: High Leverage (~150% - 200%)
    df_bull = create_segment(50, vol_level=0.005, trend_start=100, trend_end=120)
    
    # Phase 2: CRASH (30 Days)
    # High Vol (0.025), Price crashes 120 -> 90
    # Expected: Cut to Cash (~0% - 20%)
    df_bear = create_segment(30, vol_level=0.035, trend_start=120, trend_end=90)
    
    # Phase 3: RECOVERY (40 Days)
    # Normal Vol (0.01), Price recovers 90 -> 105
    # Expected: Normal Weight (~80% - 100%)
    df_recovery = create_segment(40, vol_level=0.012, trend_start=90, trend_end=105)
    
    # 4. Combine & Fix Dates
    full_demo = pd.concat([df_bull, df_bear, df_recovery], ignore_index=True)
    
    # Make date_ids sequential starting from 2000
    full_demo['date_id'] = range(2000, 2000 + len(full_demo))
    
    # 5. Save
    data_dir = os.path.join(repo_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, "bull_market_test.csv")
    
    full_demo.to_csv(output_path, index=False)
    
    print(f">>> SUCCESS! Created {output_path}")
    print(">>> Upload this file to Streamlit to see the AI adapt to changing markets!")

if __name__ == "__main__":
    generate_market_cycle()