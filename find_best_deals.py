#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Best Deals Finder for Dacia Cars
This script loads the trained model and finds the best deals from today's listings
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os
import webbrowser
from pathlib import Path

def find_best_deals():
    print("Starting best deals search...")
    
    # Load the latest trained model
    model_path = "dacia_price_model_latest.joblib"
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found. Please run train_model.py first.")
        return None
    
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Load data from GitHub repository
    url = 'https://raw.githubusercontent.com/aaldescu/dacia-auto-kleinanzeigen/refs/heads/main/cars_clean.csv'
    
    try:
        df = pd.read_csv(url)
        print(f"Data loaded successfully. Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Filter new listings
    df_new = df[df['is_new'] == 'Yes'][['km', 'car_year', 'model', 'time_on_market', 'zipcode', 'title']].copy()
    print(f"New listings filtered. Count: {len(df_new)}")
    
    # Handle missing values
    df_new['time_on_market'] = df_new['time_on_market'].fillna(0)
    df_new = df_new.dropna()
    print(f"After removing rows with missing values: {len(df_new)}")
    
    if len(df_new) == 0:
        print("No new listings found after filtering.")
        return None
    
    # Prepare data for prediction
    X_new = df_new[['km', 'car_year', 'model', 'time_on_market', 'zipcode', 'title']]
    
    # Predict prices
    try:
        df_new['predicted_price'] = model.predict(X_new)
        print("Price prediction completed.")
    except Exception as e:
        print(f"Error during price prediction: {e}")
        return None
    
    # Get additional columns from the original dataframe
    other_columns = [col for col in df.columns if col not in df_new.columns]
    temp_df = df[df['is_new'] == 'Yes'][other_columns].copy()
    
    # Merge dataframes
    df_new = pd.concat([df_new, temp_df], axis=1)
    
    # Filter listings from today
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"Filtering listings from today ({today})...")
    
    # Assuming there's a 'date_added' column or similar
    # If there's no direct date column, we might use time_on_market = 0 as a proxy for today
    today_listings = df_new[df_new['time_on_market'] == 0].copy()
    
    print(f"Found {len(today_listings)} listings from today.")
    
    if len(today_listings) == 0:
        print("No listings from today found.")
        return None
    
    # Calculate price difference (predicted - actual)
    today_listings['price_difference'] = today_listings['predicted_price'] - today_listings['price']
    
    # Find the best deals (where actual price is lower than predicted)
    best_deals = today_listings.sort_values(by='price_difference', ascending=False)
    best_deals = best_deals[best_deals['price_difference'] > 0]  # Only positive differences
    
    print(f"Found {len(best_deals)} potential good deals.")
    
    if len(best_deals) == 0:
        print("No good deals found today.")
        return None
    
    # Select relevant columns for the report
    report_columns = ['title', 'km', 'car_year', 'zipcode', 'id', 'price', 'predicted_price', 'price_difference']
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
    formatted_df['price'] = formatted_df['price'].apply(lambda x: f"€{x:,.2f}")
    formatted_df['predicted_price'] = formatted_df['predicted_price'].apply(lambda x: f"€{x:,.2f}")
    formatted_df['price_difference'] = formatted_df['price_difference'].apply(lambda x: f"€{x:,.2f}")
    
    # Create HTML with clickable IDs
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dacia Best Deals Report</title>
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
        <h1>Dacia Best Deals Report</h1>
        <p>Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
    """
    
    # Add the table with clickable IDs
    html_table = formatted_df.to_html(escape=False, formatters={
        'id': lambda id: f'<a href="https://www.kleinanzeigen.de/s-anzeige/{id}" target="_blank">{id}</a>'
    }, classes='deals-table')
    
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
    report_filename = f"dacia_best_deals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    report_path = reports_dir / report_filename
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"HTML report generated: {report_path}")
    
    # Open the report in the default browser
    browser_path = report_path.absolute().as_uri()
    webbrowser.open(browser_path)

if __name__ == "__main__":
    best_deals = find_best_deals()
    if best_deals is not None and not best_deals.empty:
        print(f"Found {len(best_deals)} good deals today!")
        # Display top 5 deals in console
        print("\nTop 5 Best Deals:")
        print(best_deals.head(5)[['title', 'price', 'predicted_price', 'price_difference']])
