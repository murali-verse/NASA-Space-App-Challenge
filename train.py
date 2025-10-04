# train.py
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from src import config
from src.data_processing import load_data, preprocess_data
from src.model_training import train_and_evaluate

def run_training():
    """Orchestrates the improved model training pipeline."""
    print("1. Loading data...")
    df = load_data(config.RAW_DATA_PATH)
    
    print("2. Preprocessing data and engineering features...")
    X, y = preprocess_data(df)
    
    # --- Handle Class Imbalance ---
    print("3. Calculating class weights for imbalance...")
    class_labels = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=y)
    class_weight_dict = dict(zip(class_labels, weights))
    print(f"Class weights: {class_weight_dict}")

    print("4. Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE, 
        stratify=y
    )
    
    print("5. Training and evaluating the robust model...")
    train_and_evaluate(X_train, y_train, X_test, y_test, class_weight=class_weight_dict)
    
    print("\n--- Training pipeline complete! ---")

if __name__ == '__main__':
    run_training()