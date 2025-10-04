# src/config.py

# --- File Paths ---
RAW_DATA_PATH = "data/raw/kepler_data.csv"
MODEL_SAVE_PATH = "saved_models/exoplanet_model.h5"
SCALER_SAVE_PATH = "saved_models/scaler.gz"
FEATURE_SELECTOR_SAVE_PATH = "saved_models/feature_selector.gz" # New line

# --- Data Processing ---
COLS_TO_DROP = [
    'rowid', 'kepid', 'kepoi_name', 'kepler_name', 'koi_disposition', 
    'koi_vet_stat', 'koi_vet_date', 'koi_pdisposition', 'koi_disp_prov', 
    'koi_comment', 'koi_fittype', 'koi_parm_prov', 'koi_tce_delivname', 
    'koi_trans_mod', 'koi_datalink_dvr', 'koi_datalink_dvs', 'koi_sparprov'
]
TARGET_COLUMN = 'is_exoplanet'

# --- Model Training ---
MODEL_PARAMS = {
    'epochs': 50,  # Increased epochs since EarlyStopping will find the best one
    'batch_size': 32,
    'validation_split': 0.2 # Increased validation split for more robust early stopping
}

TEST_SIZE = 0.2
RANDOM_STATE = 42