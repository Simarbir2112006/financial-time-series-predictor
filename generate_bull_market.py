import pandas as pd
import numpy as np
import os
from src.config import TRAIN_FILE

def generate_true_bull_market():
    print(f"Reading columns from {TRAIN_FILE}...")
    if not os.path.exists(TRAIN_FILE):
        print("Error: train.csv not found.")
        return

    # Load just the header/first row to get column names
    template_df = pd.read_csv(TRAIN_FILE, nrows=1)
    columns = template_df.columns
    
    days = 50
    data = {}

    # Initialize all columns with 0
    for col in columns:
        data[col] = [0] * days

    # --- INTELLIGENT FILLING ---
    for col in columns:
        # 1. DATE: Sequential
        if col == 'date_id':
            # Start from a future date
            start_date = template_df['date_id'].max() + 100
            data[col] = range(1000, 1000 + days)
        
        # 2. VOLATILITY (V): Super Low (0.5% daily) -> Leads to High Allocation
        elif col.startswith('V'):
            data[col] = [0.005] * days
            
        # 3. MOMENTUM (M) & PRICE (P): Strong Upward Trend
        elif col.startswith('M') or col.startswith('P'):
            # Linear growth from 100 to 120
            data[col] = np.linspace(100, 120, days)
            
        # 4. SENTIMENT (S): Positive
        elif col.startswith('S'):
            data[col] = [1.0] * days
            
        # 5. OTHERS: Random Noise (to prevent constant value errors)
        else:
            data[col] = np.random.uniform(0.01, 0.02, days)

    df_bull = pd.DataFrame(data)
    
    # Save
    output_path = 'data/bull_market_test.csv'
    df_bull.to_csv(output_path, index=False)
    print(f">>> Success! Generated {output_path} with {len(df_bull.columns)} columns.")
    print(">>> This file now mirrors the exact structure of train.csv but with Low Volatility.")

if __name__ == "__main__":
    generate_true_bull_market()