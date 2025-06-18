#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Training Script for Dacia Price Prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
import joblib
from datetime import datetime

def train_model():
    print("Starting model training...")
    
    # Load data from local CSV file
    csv_path = "cars_clean.csv"
    
    try:
        df = pd.read_csv(csv_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Filter used cars for training the model
    df_model = df[df['is_new'] == 'No'][['price', 'km', 'car_year', 'model', 'time_on_market', 'zipcode', 'title']].copy()
    print(f"Training data filtered. Shape: {df_model.shape}")
    
    # Clean data
    # Replace infinite values with NaN and then fill NaN values
    df_model.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_model = df_model.fillna(1)
    
    # Avoid division by zero
    df_model['km'] = df_model['km'].replace(0, 1)
    
    # Prepare features and target
    X = df_model.drop(columns=['price'])
    y = df_model['price']
    
    # Define column types
    categorical_cols = ['model', 'zipcode']
    text_col = 'title'
    
    # Create text transformer
    text_transformer = CountVectorizer(max_features=100)
    
    # Create preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('text', Pipeline([('vect', text_transformer)]), text_col)
    ], remainder='passthrough')
    
    # Create and train model
    model = Pipeline([
        ('prep', preprocessor),
        ('reg', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    print("Training model...")
    model.fit(X, y)
    print("Model training completed!")
    
    # Save the model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"dacia_price_model_{timestamp}.joblib"
    joblib.dump(model, model_filename)
    print(f"Model saved as {model_filename}")
    
    # Also save as latest model for easy reference
    joblib.dump(model, "dacia_price_model_latest.joblib")
    print("Model also saved as dacia_price_model_latest.joblib")
    
    return model

if __name__ == "__main__":
    train_model()
