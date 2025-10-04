# src/data_processing.py
import pandas as pd
import numpy as np
from src import config

def load_data(path):
    """Loads data from a CSV file."""
    return pd.read_csv(path, comment='#')

def preprocess_data(df):
    """
    Cleans, engineers features, and prepares data for training.
    Returns features (X) and target (y).
    """
    # Filter for clear dispositions
    df_filtered = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()

    # Create the target variable
    df_filtered[config.TARGET_COLUMN] = df_filtered['koi_disposition'].apply(
        lambda x: 1 if x == 'CONFIRMED' else 0
    )
    y = df_filtered[config.TARGET_COLUMN]

    # Drop target and other non-feature columns
    cols_to_drop = config.COLS_TO_DROP + [config.TARGET_COLUMN]
    X = df_filtered.drop(columns=cols_to_drop, errors='ignore')

    # Ensure all columns are numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    
    # --- Advanced Feature Engineering ---
    X['signal_to_noise_ratio_composite'] = X['koi_max_mult_ev'] / (X['koi_max_sngle_ev'] + 1e-6)

    # --- Robust Missing Value Imputation (THE FIX IS HERE) ---
    for col in X.columns:
        if X[col].isnull().any():
            median_val = X[col].median()
            
            # If a column is entirely empty, its median will be NaN.
            # In this case, we'll fill with 0 as a safe default.
            if pd.isna(median_val):
                fill_value = 0
            else:
                fill_value = median_val
            
            # Use direct assignment instead of inplace=True to avoid the CopyWarning
            X[col] = X[col].fillna(fill_value)
            
    print(f"Data preprocessed with {X.shape[1]} initial features. All NaNs handled.")
    return X, y