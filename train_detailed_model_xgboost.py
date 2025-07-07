#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
XGBoost model training script for Dacia auto data
This script connects to MySQL, loads data from the 'detailed' table,
performs feature engineering, and trains an XGBoost model.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import pymysql
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# Load environment variables
load_dotenv()

# MySQL connection parameters
MYSQL_CONFIG = {
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'database': os.getenv('DB_NAME'),
    'port': int(os.getenv('DB_PORT', 3306))
}

def connect_to_mysql():
    """Establish connection to MySQL database"""
    try:
        conn = pymysql.connect(**MYSQL_CONFIG)
        print("Successfully connected to MySQL database")
        return conn
    except pymysql.Error as err:
        print(f"Error connecting to MySQL: {err}")
        raise
def load_data_from_mysql(conn):
    """Load data from MySQL 'detailed' table"""
    today = datetime.today().strftime('%Y-%m-%d')
    query = f"SELECT * FROM detailed WHERE date_scrape != '{today}'"
    df = pd.read_sql(query, conn)
    print(f"Loaded {len(df)} records from MySQL")
    return df

def preprocess_data(df):
    """Preprocess the data for XGBoost"""
    print("Starting data preprocessing...")
    
    # Make a copy to avoid modifying the original dataframe
    processed_df = df.copy()
    
    # 1. Handle numeric features
    print("Processing numeric features...")
    
    # Convert kilometer readings to numeric values
    processed_df['kilometerstand_numeric'] = processed_df['kilometerstand'].fillna('0')
    # Extract numeric part using regex
    processed_df['kilometerstand_numeric'] = processed_df['kilometerstand_numeric'].str.extract(r'(\d+[\d\.]*)')[0].str.replace('.', '').fillna(0).astype(float)
    
    # Extract numeric values from power (PS)
    processed_df['leistung_numeric'] = processed_df['leistung'].fillna('0')
    processed_df['leistung_numeric'] = processed_df['leistung_numeric'].str.extract(r'(\d+)')[0].fillna(0).astype(float)
    
    # Convert price to numeric with robust error handling
    processed_df['price_clean'] = processed_df['price'].fillna('0')
    # Extract numeric part using regex (handles cases like '3000 VB')
    processed_df['price_numeric'] = processed_df['price_clean'].str.extract(r'(\d+[\d\.]*)')[0].str.replace('.', '').fillna(0).astype(float)
    
    # Extract year from registration date
    processed_df['registration_year'] = processed_df['erstzulassung'].fillna('')
    processed_df['registration_year'] = processed_df['registration_year'].str.extract(r'(\d{4})')[0].fillna(0).astype(int)
    # For entries with only month, estimate based on other data
    # If year is 0, try to infer from other data or set to median
    if (processed_df['registration_year'] == 0).any():
        median_year = processed_df[processed_df['registration_year'] > 0]['registration_year'].median()
        processed_df.loc[processed_df['registration_year'] == 0, 'registration_year'] = median_year
    
    # 2. Handle categorical features
    print("Processing categorical features...")
    
    # Fill missing categorical values
    categorical_cols_to_fill = ['umweltplakette', 'schadstoffklasse', 'material_innenausstattung', 
                               'aussenfarbe', 'fahrzeugzustand']
    for col in categorical_cols_to_fill:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].fillna('Unknown')
    
    # One-hot encoding for categorical variables
    categorical_cols = ['marke', 'modell', 'fahrzeugzustand', 'kraftstoffart', 
                        'getriebe', 'fahrzeugtyp', 'aussenfarbe', 'material_innenausstattung',
                        'seller_type', 'schadstoffklasse']
    
    # Only encode columns that exist in the dataframe
    categorical_cols = [col for col in categorical_cols if col in processed_df.columns]
    
    # Apply one-hot encoding
    processed_df = pd.get_dummies(processed_df, columns=categorical_cols, drop_first=True)
    
    # Label encoding for ordinal variables
    if 'umweltplakette' in processed_df.columns:
        le = LabelEncoder()
        processed_df['umweltplakette_encoded'] = le.fit_transform(processed_df['umweltplakette'].fillna('Unknown'))
    
    # 3. Text feature processing
    print("Processing text features...")
    
    # Fill NaN values in text columns
    processed_df['description'] = processed_df['description'].fillna('')
    processed_df['extras'] = processed_df['extras'].fillna('')
    
    # Create TF-IDF features from description and extras (limit to top features)
    tfidf = TfidfVectorizer(max_features=50, stop_words='english')
    
    # Only create TF-IDF features if there's enough data
    if len(processed_df) > 10:  # Arbitrary threshold
        description_features = tfidf.fit_transform(processed_df['description'])
        description_df = pd.DataFrame(
            description_features.toarray(), 
            columns=[f'desc_{i}' for i in range(description_features.shape[1])]
        )
        
        # Reset the vectorizer for extras
        tfidf = TfidfVectorizer(max_features=30, stop_words='english')
        extras_features = tfidf.fit_transform(processed_df['extras'])
        extras_df = pd.DataFrame(
            extras_features.toarray(), 
            columns=[f'extra_{i}' for i in range(extras_features.shape[1])]
        )
        
        # Concatenate with original dataframe
        processed_df = pd.concat([processed_df, description_df, extras_df], axis=1)
    
    # 4. Feature engineering
    print("Engineering additional features...")
    
    # Vehicle age calculation
    current_year = datetime.now().year
    processed_df['vehicle_age'] = current_year - processed_df['registration_year']
    
    # Price per kilometer ratio (value indicator)
    processed_df['price_per_km'] = processed_df['price_numeric'] / processed_df['kilometerstand_numeric'].replace(0, 1)
    
    # Binary features from text columns
    processed_df['has_navigation'] = processed_df['extras'].str.contains('Navigation|Navi', case=False, na=False).astype(int)
    processed_df['has_climate'] = processed_df['extras'].str.contains('Klima', case=False, na=False).astype(int)
    processed_df['has_bluetooth'] = processed_df['extras'].str.contains('Bluetooth', case=False, na=False).astype(int)
    processed_df['is_accident_free'] = processed_df['description'].str.contains('Unfallfrei', case=False, na=False).astype(int)
    processed_df['is_service_history'] = processed_df['extras'].str.contains('Scheckheftgepflegt', case=False, na=False).astype(int)
    processed_df['has_leather'] = processed_df['extras'].str.contains('Leder', case=False, na=False).astype(int)
    processed_df['has_alloy_wheels'] = processed_df['extras'].str.contains('Leichtmetallfelgen|Alufelgen', case=False, na=False).astype(int)
    
    # 5. Missing value treatment
    print("Handling missing values...")
    
    # Fill missing numeric values with median
    numeric_cols = ['kilometerstand_numeric', 'leistung_numeric', 'price_numeric', 
                    'vehicle_age', 'price_per_km', 'registration_year']
    for col in numeric_cols:
        processed_df[col] = processed_df[col].fillna(processed_df[col].median())
    
    return processed_df

def select_features(df, target_col='price_numeric'):
    """Select and scale features for model training"""
    print("Selecting and scaling features...")
    
    # Define base features
    base_features = [
        'kilometerstand_numeric', 'leistung_numeric', 'vehicle_age',
        'registration_year', 'has_navigation', 'has_climate', 
        'has_bluetooth', 'is_accident_free', 'is_service_history',
        'has_leather', 'has_alloy_wheels'
    ]
    
    # Add one-hot encoded columns (filter for ones that exist)
    encoded_cols = [col for col in df.columns if 
                   any(col.startswith(prefix + '_') for prefix in 
                      ['marke', 'modell', 'fahrzeugzustand', 'kraftstoffart', 'getriebe', 
                       'fahrzeugtyp', 'aussenfarbe', 'material_innenausstattung'])]
    
    # Add TF-IDF features
    tfidf_cols = [col for col in df.columns if col.startswith('desc_') or col.startswith('extra_')]
    
    # Combine all features
    all_features = base_features + encoded_cols + tfidf_cols
    
    # Filter to only include columns that exist in the dataframe
    features = [col for col in all_features if col in df.columns and col != target_col]
    
    # Scale numeric features
    numeric_features = [col for col in 
                       ['kilometerstand_numeric', 'leistung_numeric', 'vehicle_age', 
                        'price_per_km', 'registration_year'] 
                       if col in features]
    
    if numeric_features:
        scaler = StandardScaler()
        df[numeric_features] = scaler.fit_transform(df[numeric_features])
        
        # Save the scaler for future use
        os.makedirs('models', exist_ok=True)
        joblib.dump(scaler, 'models/scaler.pkl')
    
    return df[features], features

def train_xgboost_model(X, y, features):
    """Train XGBoost model with hyperparameter tuning"""
    print("Training XGBoost model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Basic XGBoost model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    # If dataset is large enough, perform hyperparameter tuning
    if len(X) > 100:  # Arbitrary threshold
        print("Performing hyperparameter tuning...")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            verbose=1,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
    else:
        # Train with default parameters
        model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Evaluation Metrics:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Save feature importance plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
    plt.title('Feature Importance')
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/feature_importance.png')
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/xgboost_model.pkl')
    
    return model, feature_importance

def predict_price(model, features, new_data):
    """Make predictions using the trained model"""
    # Preprocess new data the same way as training data
    # This would require applying the same transformations
    
    # Make predictions
    predictions = model.predict(new_data[features])
    return predictions

def main():
    """Main function to execute the entire pipeline"""
    try:
        # Connect to MySQL
        conn = connect_to_mysql()
        
        # Load data
        df = load_data_from_mysql(conn)
        
        # Close connection
        conn.close()
        
        # Check if we have enough data
        if len(df) < 10:
            print("Not enough data to train a model. Need at least 10 records.")
            return
        
        # Preprocess data
        processed_df = preprocess_data(df)
        
        # Define target variable - assuming we're predicting price
        target_col = 'price_numeric'
        
        # Select features
        X, features = select_features(processed_df, target_col)
        y = processed_df[target_col]
        
        # Train model
        model, feature_importance = train_xgboost_model(X, y, features)
        
        print("\nModel training complete!")
        print(f"Model and scaler saved in the 'models' directory")
        print(f"Feature importance plot saved in the 'plots' directory")
        
        # Save feature importance to CSV
        feature_importance.to_csv('models/feature_importance.csv', index=False)
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
