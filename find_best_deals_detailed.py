#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Best Deals Finder for Dacia Cars (Detailed Version)
This script loads the trained XGBoost model and finds the best deals from today's listings
using the detailed dataset features
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os
import webbrowser
from pathlib import Path
import pymysql
from dotenv import load_dotenv
import re
from sklearn.feature_extraction.text import TfidfVectorizer

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


def load_data_from_mysql():
    """Load data from MySQL 'detailed' table"""
    try:
        conn = connect_to_mysql()
        query = "SELECT * FROM detailed WHERE date_scrape >= CURDATE() - INTERVAL 2 DAY"
        df = pd.read_sql(query, conn)
        conn.close()
        print(f"Loaded {len(df)} recent records from MySQL")
        return df
    except Exception as e:
        print(f"Error loading data from MySQL: {e}")
        return None


def preprocess_data(df):
    """Preprocess the data for prediction"""
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
    
    # 3. Feature engineering
    print("Engineering additional features...")
    
    # Calculate vehicle age
    current_year = datetime.now().year
    processed_df['vehicle_age'] = current_year - processed_df['registration_year']
    
    # Price per kilometer ratio (value indicator)
    processed_df['price_per_km'] = processed_df['price_numeric'] / processed_df['kilometerstand_numeric'].replace(0, 1)
    
    # Create binary features from text columns
    if 'extras' in processed_df.columns:
        processed_df['has_navigation'] = processed_df['extras'].str.contains('Navigation|Navi', case=False, na=False).astype(int)
        processed_df['has_climate'] = processed_df['extras'].str.contains('Klima|Climatronic|Klimaautomatik', case=False, na=False).astype(int)
        processed_df['has_bluetooth'] = processed_df['extras'].str.contains('Bluetooth', case=False, na=False).astype(int)
        processed_df['has_leather'] = processed_df['extras'].str.contains('Leder', case=False, na=False).astype(int)
        processed_df['has_alloy_wheels'] = processed_df['extras'].str.contains('Alufelgen|Leichtmetallfelgen', case=False, na=False).astype(int)
    
    if 'description' in processed_df.columns:
        processed_df['is_accident_free'] = processed_df['description'].str.contains('unfallfrei|Unfallfrei|kein Unfall', case=False, na=False).astype(int)
        processed_df['is_service_history'] = processed_df['description'].str.contains('Scheckheft|Serviceheft|Wartungsheft', case=False, na=False).astype(int)
        
    # 4. TF-IDF feature extraction for text columns
    print("Extracting text features...")
    
    # Process 'extras' column with TF-IDF
    if 'extras' in processed_df.columns:
        # Fill NaN values
        processed_df['extras'] = processed_df['extras'].fillna('')
        
        # Create TF-IDF features for extras (max 20 features)
        tfidf_extras = TfidfVectorizer(max_features=20, stop_words=['und', 'mit', 'der', 'die', 'das'])
        extras_matrix = tfidf_extras.fit_transform(processed_df['extras'])
        extras_df = pd.DataFrame(extras_matrix.toarray(), columns=[f'extra_{i}' for i in range(extras_matrix.shape[1])])
        
        # Add TF-IDF features to processed dataframe
        processed_df = pd.concat([processed_df.reset_index(drop=True), extras_df], axis=1)
    
    # Process 'description' column with TF-IDF
    if 'description' in processed_df.columns:
        # Fill NaN values
        processed_df['description'] = processed_df['description'].fillna('')
        
        # Create TF-IDF features for description (max 50 features)
        tfidf_desc = TfidfVectorizer(max_features=50, stop_words=['und', 'mit', 'der', 'die', 'das'])
        desc_matrix = tfidf_desc.fit_transform(processed_df['description'])
        desc_df = pd.DataFrame(desc_matrix.toarray(), columns=[f'desc_{i}' for i in range(desc_matrix.shape[1])])
        
        # Add TF-IDF features to processed dataframe
        processed_df = pd.concat([processed_df.reset_index(drop=True), desc_df], axis=1)
    
    # 4. Missing value treatment
    print("Handling missing values...")
    
    # Fill missing numeric values with median
    numeric_cols = ['kilometerstand_numeric', 'leistung_numeric', 'price_numeric', 
                    'vehicle_age', 'price_per_km', 'registration_year']
    for col in numeric_cols:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].fillna(processed_df[col].median())
    
    return processed_df


def find_best_deals_detailed():
    """Find the best deals using the detailed data and XGBoost model"""
    print("Starting best deals search using detailed data...")
    
    # Load the trained model and scaler
    model_path = "models/xgboost_model.pkl"
    scaler_path = "models/scaler.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Error: Model file {model_path} or scaler file {scaler_path} not found.")
        print("Please run train_detailed_model_xgboost.py first.")
        return None
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print("Model and scaler loaded successfully.")
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        return None
    
    # Load recent data from MySQL
    df = load_data_from_mysql()
    if df is None or len(df) == 0:
        print("No recent data found in the database.")
        return None
    
    print(f"Processing {len(df)} recent listings...")
    
    # Preprocess data
    processed_df = preprocess_data(df)
    
    # Get the feature names used during training
    # These should match the features used in train_detailed_model_xgboost.py
    base_features = [
        'kilometerstand_numeric', 'leistung_numeric', 'vehicle_age',
        'registration_year', 'has_navigation', 'has_climate', 
        'has_bluetooth', 'is_accident_free', 'is_service_history',
        'has_leather', 'has_alloy_wheels'
    ]
    
    # Add one-hot encoded columns
    encoded_cols = [col for col in processed_df.columns if 
                   any(col.startswith(prefix + '_') for prefix in 
                      ['marke', 'modell', 'fahrzeugzustand', 'kraftstoffart', 'getriebe', 
                       'fahrzeugtyp', 'aussenfarbe', 'material_innenausstattung'])]
    
    # Add TF-IDF text features
    tfidf_cols = [col for col in processed_df.columns if col.startswith('desc_') or col.startswith('extra_')]
    
    # Combine all features
    all_features = base_features + encoded_cols + tfidf_cols
    
    # Filter to only include columns that exist in the dataframe
    features = [col for col in all_features if col in processed_df.columns]
    
    print(f"Using {len(features)} features for prediction")
    
    # Prepare data for prediction
    X_new = processed_df[features]
    
    # Handle missing columns that were in the training data but not in new data
    try:
        # Load feature names from the model
        model_features = model.get_booster().feature_names
        
        # Check for missing features
        missing_features = [feat for feat in model_features if feat not in X_new.columns]
        if missing_features:
            print(f"Adding {len(missing_features)} missing features that were in training data")
            for feat in missing_features:
                X_new[feat] = 0  # Add missing columns with zeros
        
        # Ensure columns are in the same order as during training
        X_new = X_new[model_features]
        
        # Predict prices
        processed_df['predicted_price'] = model.predict(X_new)
        print("Price prediction completed.")
    except Exception as e:
        print(f"Error during price prediction: {e}")
        # This might happen if the columns don't match the training data
        print("This could be due to missing features in the new data that were present during training.")
        return None
    
    # Calculate price difference (predicted - actual)
    processed_df['price_difference'] = processed_df['predicted_price'] - processed_df['price_numeric']
    processed_df['price_difference_percent'] = (processed_df['price_difference'] / processed_df['price_numeric']) * 100
    
    # Find the best deals (where actual price is lower than predicted)
    best_deals = processed_df.sort_values(by='price_difference', ascending=False)
    best_deals = best_deals[best_deals['price_difference'] > 0]  # Only positive differences
    
    print(f"Found {len(best_deals)} potential good deals.")
    
    if len(best_deals) == 0:
        print("No good deals found in recent listings.")
        return None
    
    # Select relevant columns for the report
    report_columns = [
        'ad_id', 'title', 'kilometerstand', 'erstzulassung', 'marke', 'modell',
        'price', 'price_numeric', 'predicted_price', 'price_difference', 'price_difference_percent'
    ]
    
    # Only include columns that exist
    report_columns = [col for col in report_columns if col in best_deals.columns]
    
    best_deals_report = best_deals[report_columns].copy()
    
    # Generate HTML report
    generate_html_report(best_deals_report)
    
    return best_deals_report


def generate_html_report(deals_df):
    """Generate an HTML report with the best deals"""
    if deals_df is None or len(deals_df) == 0:
        print("No deals to report.")
        return
    
    # Format the DataFrame for better display
    formatted_df = deals_df.copy()
    
    # Format numeric columns
    if 'price_numeric' in formatted_df.columns:
        formatted_df['price_numeric'] = formatted_df['price_numeric'].apply(lambda x: f"€{x:,.2f}")
    if 'predicted_price' in formatted_df.columns:
        formatted_df['predicted_price'] = formatted_df['predicted_price'].apply(lambda x: f"€{x:,.2f}")
    if 'price_difference' in formatted_df.columns:
        formatted_df['price_difference'] = formatted_df['price_difference'].apply(lambda x: f"€{x:,.2f}")
    if 'price_difference_percent' in formatted_df.columns:
        formatted_df['price_difference_percent'] = formatted_df['price_difference_percent'].apply(lambda x: f"{x:.1f}%")
    
    # Create HTML with clickable IDs
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dacia Best Deals Report (Detailed)</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #2c3e50; }
            table { border-collapse: collapse; width: 100%; margin-top: 20px; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; color: #333; }
            tr:hover { background-color: #f5f5f5; }
            .good-deal { background-color: #d4edda; }
            .great-deal { background-color: #c3e6cb; }
            .amazing-deal { background-color: #b1dfbb; }
        </style>
    </head>
    <body>
        <h1>Dacia Best Deals Report (Detailed Analysis)</h1>
        <p>Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        <p>This report uses the advanced XGBoost model trained on detailed car features.</p>
    """
    
    # Add the table with clickable IDs if ad_id column exists
    if 'ad_id' in formatted_df.columns:
        html_table = formatted_df.to_html(escape=False, formatters={
            'ad_id': lambda id: f'<a href="https://www.kleinanzeigen.de/s-anzeige/{id}" target="_blank">{id}</a>'
        }, classes='deals-table')
    else:
        html_table = formatted_df.to_html(escape=False, classes='deals-table')
    
    html_content += html_table
    
    html_content += """
    </body>
    </html>
    """
    
    # Create reports directory if it doesn't exist
    reports_dir = Path("reports")
    if not reports_dir.exists():
        reports_dir.mkdir(exist_ok=True)
    
    # Save the HTML report in the reports folder
    report_filename = f"dacia_best_deals_detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    report_path = reports_dir / report_filename
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"HTML report generated: {report_path}")
    
    # Open the report in the default browser
    browser_path = report_path.absolute().as_uri()
    webbrowser.open(browser_path)


if __name__ == "__main__":
    best_deals = find_best_deals_detailed()
    if best_deals is not None and not best_deals.empty:
        print(f"Found {len(best_deals)} good deals in recent listings!")
        # Display top 5 deals in console
        print("\nTop 5 Best Deals:")
        display_cols = ['title', 'price', 'predicted_price', 'price_difference', 'price_difference_percent']
        display_cols = [col for col in display_cols if col in best_deals.columns]
        print(best_deals.head(5)[display_cols])
    else:
        print("No good deals found or there was an error in the process.")
        print("Make sure the model is trained and the database has recent listings.")
