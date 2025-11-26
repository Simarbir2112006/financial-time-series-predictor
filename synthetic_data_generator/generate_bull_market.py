import pandas as pd
import numpy as np
import os, sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.config import TRAIN_FILE

def generate_true_bull_market():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    repo_root = os.path.join(base_dir, "..")

    train_file_path = TRAIN_FILE
    if not os.path.isabs(TRAIN_FILE):
        train_file_path = os.path.join(repo_root, TRAIN_FILE)

    print(f"Reading columns from {train_file_path}...")

    if not os.path.exists(train_file_path):
        print("Error: train.csv not found.")
        return

    # Load just the header/first row to get column names
    template_df = pd.read_csv(train_file_path, nrows=1)
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
    
    data_dir = os.path.join(repo_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, "bull_market_test.csv")

    df_bull.to_csv(output_path, index=False)
    print(f">>> Success! Generated {output_path} with {len(df_bull.columns)} columns.")
    print(">>> This file now mirrors the exact structure of train.csv but with Low Volatility.")

if __name__ == "__main__":
    generate_true_bull_market()