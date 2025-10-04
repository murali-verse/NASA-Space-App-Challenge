# src/model_training.py
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from src import config

def build_robust_model(input_shape):
    """
    Defines a more robust neural network with Dropout regularization.
    """
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),  # Dropout layer to prevent overfitting
        Dense(64, activation='relu'),
        Dropout(0.2),  # Dropout layer
        Dense(1, activation='sigmoid')
    ])
    
    # Use Adam optimizer with a slightly lower learning rate for stability
    optimizer = Adam(learning_rate=0.0005)
    
    model.compile(optimizer=optimizer, 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

def train_and_evaluate(X_train, y_train, X_test, y_test, class_weight=None):
    """
    Selects features, scales data, trains the model, evaluates, and saves all artifacts.
    """
    # --- Feature Selection ---
    # Select the top 40 best features based on the F-statistic
    selector = SelectKBest(f_classif, k=40)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    print(f"Feature selection complete. Using {X_train_selected.shape[1]} best features.")

    # --- Scaling ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # --- Model Training ---
    model = build_robust_model(X_train_scaled.shape[1])
    print("\n--- Model Summary ---")
    model.summary()

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=5,  # Stop after 5 epochs with no improvement
        restore_best_weights=True
    )

    print("\n--- Starting Model Training ---")
    history = model.fit(
        X_train_scaled, y_train, 
        epochs=config.MODEL_PARAMS['epochs'], 
        batch_size=config.MODEL_PARAMS['batch_size'],
        validation_split=config.MODEL_PARAMS['validation_split'],
        callbacks=[early_stopping],
        class_weight=class_weight, # Use class weights here
        verbose=2
    )

    # --- Evaluation ---
    print("\n--- Evaluating Model on Best Weights ---")
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

    # --- Save Artifacts ---
    model.save(config.MODEL_SAVE_PATH)
    joblib.dump(scaler, config.SCALER_SAVE_PATH)
    joblib.dump(selector, config.FEATURE_SELECTOR_SAVE_PATH) # Save the selector!
    
    print(f"\nModel saved to {config.MODEL_SAVE_PATH}")
    print(f"Scaler saved to {config.SCALER_SAVE_PATH}")
    print(f"Feature selector saved to {config.FEATURE_SELECTOR_SAVE_PATH}")