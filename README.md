# 🪐 Kepler Exoplanet Explorer 🛰️  

An **end-to-end deep learning project** to classify potential exoplanets from NASA's **Kepler Space Telescope** data.  
This repository includes the **complete pipeline**: data cleaning, feature engineering, model training, and deployment as an interactive web application using **Streamlit**.  

---

## ✨ Demo  
The application provides an **interactive web interface** to:  
- Upload Kepler data  
- Instantly classify objects as **confirmed exoplanets** or **false positives**  
- View **confidence scores** and visualizations  

👉 Built with **Streamlit** for simplicity and interactivity.  

---

## 🚀 Features  
- **Advanced Data Processing**: Cleans raw NASA KOI data, engineers features (e.g., `signal_to_noise_ratio_composite`), and handles missing values.  
- **Intelligent Feature Selection**: Uses **scikit-learn’s SelectKBest** to keep only the most impactful features.  
- **Robust Deep Learning Model**: TensorFlow/Keras neural network with **Dropout regularization** and **Early Stopping**.  
- **Class Imbalance Handling**: Applies **class weights** to avoid bias toward frequent classes.  
- **Interactive Web UI**: Upload your dataset and get instant predictions.  
- **Modular & Scalable**: Clean separation of **data**, **model**, and **application layers**.  

---

## 🛠️ Technology Stack  
**Backend & Modeling**: Python, TensorFlow, Keras, Scikit-learn, Pandas, NumPy  
**Frontend**: Streamlit  
**Dataset**: NASA Kepler Objects of Interest (KOI)  

---

## 📁 Project Structure  
```bash
exoplanet-classifier/
│
├── data/
│   └── raw/
│       └── kepler_data.csv          # NASA KOI dataset
│
├── saved_models/
│   ├── exoplanet_model.h5           # Trained DNN model
│   ├── scaler.gz                    # Fitted scaler
│   └── feature_selector.gz          # Saved feature selector
│
├── src/
│   ├── __init__.py
│   ├── config.py                    # Global configs
│   ├── data_processing.py           # Data cleaning & feature engineering
│   └── model_training.py            # Model architecture & training
│
├── app.py                           # Streamlit app
├── train.py                         # Training entry point
├── requirements.txt                 # Dependencies
└── README.md
