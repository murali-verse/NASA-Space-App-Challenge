🪐 Kepler Exoplanet Explorer 🛰️
An end-to-end deep learning project to classify potential exoplanets from NASA's Kepler Space Telescope data. This repository contains the complete pipeline for data cleaning, feature engineering, model training, and deployment as an interactive web application using Streamlit.

✨ Demo
The application provides an interactive interface to upload Kepler data and receive instant predictions on whether an object is a confirmed exoplanet or a false positive, along with a confidence score.

The interactive web application built with Streamlit.

## 🚀 Features
Advanced Data Processing: Implements robust cleaning, feature engineering (signal_to_noise_ratio_composite), and handles missing values.

Intelligent Feature Selection: Uses scikit-learn's SelectKBest to automatically identify the most impactful features for prediction, reducing noise.

Robust Deep Learning Model: A TensorFlow/Keras-based neural network with Dropout regularization and Early Stopping to prevent overfitting and maximize performance.

Handles Class Imbalance: Calculates and applies class weights during training to ensure the model doesn't get biased towards the more frequent class.

Interactive Web UI: A user-friendly front end built with Streamlit that allows users to upload their own data and visualize prediction results.

Modular & Scalable Codebase: The project is structured with a clear separation of concerns (data, model, app) for easy maintenance and extension.

## 🛠️ Technology Stack
Backend & Modeling: Python, TensorFlow, Keras, Scikit-learn, Pandas, NumPy

Frontend: Streamlit

Data: NASA Kepler Objects of Interest (KOI) Dataset

## 📁 Project Structure
The repository is organized in a modular structure for clarity and scalability:

exoplanet-classifier/
│
├── data/
│   └── raw/
│       └── kepler_data.csv
│
├── saved_models/
│   ├── exoplanet_model.h5
│   ├── scaler.gz
│   └── feature_selector.gz
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_processing.py
│   └── model_training.py
│
├── app.py
├── train.py
├── requirements.txt
└── README.md
## ⚙️ Setup and Installation
Follow these steps to set up the project environment on your local machine.

### 1. Clone the Repository
Bash

git clone https://github.com/your-username/exoplanet-classifier.git
cd exoplanet-classifier
### 2. Create a Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies.

Bash

# For Python 3
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
### 3. Install Dependencies
Install all the required libraries using the requirements.txt file.

Bash

pip install -r requirements.txt
### 4. Download the Data
You need to download the cumulative Kepler Objects of Interest (KOI) dataset.

Go to the NASA Exoplanet Archive: Cumulative KOI Table

Click "Download Table" and select "Comma-separated values (CSV)".

Save the downloaded file as kepler_data.csv inside the data/raw/ directory.

## 📖 Usage
The project has two main entry points: training the model and running the application.

### 1. Train the Model
To train the neural network from scratch, run the train.py script from the root directory. This will perform all the data processing steps and save the trained model, scaler, and feature selector to the saved_models/ directory.

Bash

python train.py
You should see output detailing the data processing, model training progress, and final test accuracy.

### 2. Run the Web Application
Once the models are trained and saved, you can launch the interactive Streamlit application.

Bash

streamlit run app.py
Your default web browser will open with the application running at http://localhost:8501. You can now upload a compatible CSV file to get predictions.

## 🧠 Model Details
The classification model is a Deep Neural Network (DNN) designed for high accuracy and robustness.

Architecture: A sequential model with multiple dense layers and ReLU activation.

Regularization: Dropout layers are used to prevent overfitting.

Optimization: The model is trained using the Adam optimizer and binary_crossentropy loss, with EarlyStopping to find the optimal number of training epochs.

Performance: The model achieves an accuracy of ~98% on the held-out test set, effectively distinguishing between confirmed exoplanets and false positives.

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

## 🙏 Acknowledgments
This project uses data collected by the Kepler mission. We gratefully acknowledge the entire Kepler team and NASA for making this invaluable dataset publicly available through the .
