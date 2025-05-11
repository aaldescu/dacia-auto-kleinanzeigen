import os
import pymysql
import pandas as pd
import re
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def clean_cars_data():
    # Get database connection details from environment variables
    db_host = os.getenv('DB_HOST')
    db_port = int(os.getenv('DB_PORT'))
    db_name = os.getenv('DB_NAME')
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    
    # Connect to the MySQL database
    try:
        conn = pymysql.connect(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password,
            database=db_name,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        print(f"Connected to MySQL database: {db_name} on {db_host}")
    except Exception as e:
        print(f"Error connecting to MySQL database: {e}")
        return pd.DataFrame()
    
    print("Connected to database. Checking tables...")
    cursor = conn.cursor()
    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()
    print(f"Available tables: {[list(t.values())[0] for t in tables]}")
    
    # Extract data from the original table
    try:
        query = "SELECT * FROM cars"
        print(f"Executing query: {query}")
        df = pd.DataFrame(cursor.fetchall())
        cursor.execute(query)
        rows = cursor.fetchall()
        df = pd.DataFrame(rows)
        print(f"Retrieved {len(df)} rows from cars table")
        
        if len(df) == 0:
            print("WARNING: No data found in the cars table!")
            return pd.DataFrame()
        
        # Check if the first row contains column headers as values
        first_row = df.iloc[0].to_dict()
        if first_row.get('id') == 'id' and first_row.get('car_km') == 'car_km':
            print("First row appears to contain column headers - removing it")
            df = df.iloc[1:].reset_index(drop=True)
            
    except Exception as e:
        print(f"Error retrieving data: {str(e)}")
        return pd.DataFrame()
    
    # Create a cleaned dataframe
    df_clean = pd.DataFrame()
    
    # Copy and clean ID column
    df_clean['id'] = df['id']
    
    # Copy ad_type
    df_clean['ad_type'] = df['ad_type']
    
    # Clean car_km values
    print("Cleaning car_km values...")
    df_clean['km'] = df['car_km'].apply(lambda x: clean_km(x))
    print(f"Successfully extracted {df_clean['km'].notnull().sum()} non-null km values")
    
    # Clean car_year values
    print("Cleaning car_year values...")
    df_clean['car_year'] = df['car_year'].apply(lambda x: clean_year(x))
    print(f"Successfully extracted {df_clean['car_year'].notnull().sum()} non-null car_year values")
    
    # Copy image_src
    df_clean['image_src'] = df['image_src']
    
    # Copy location and zipcode
    df_clean['location'] = df['location']
    df_clean['zipcode'] = df['zipcode']
    
    # Copy date columns
    df_clean['date_posted'] = df['date_posted']
    df_clean['date_scrape'] = df['date_scrape']
    
    # Copy title
    df_clean['title'] = df['title']
    
    # Clean price values
    print("Cleaning price values...")
    df_clean['price'] = df['price'].apply(lambda x: clean_price(x))
    print(f"Successfully extracted {df_clean['price'].notnull().sum()} non-null price values")
    
    # Calculate time_on_market in days
    print("Calculating time on market...")
    df_clean['time_on_market'] = df_clean.apply(
        lambda row: calculate_days_on_market(row['date_posted'], row['date_scrape']), 
        axis=1
    )
    print(f"Successfully calculated {df_clean['time_on_market'].notnull().sum()} non-null time_on_market values")
    
    # Export the cleaned data to CSV
    print("Exporting data to CSV file...")
    export_to_csv(df_clean)
    
    conn.close()
    print("Data cleaning completed successfully!")
    return df_clean

def clean_km(value):
    """Clean and extract kilometer values"""
    if pd.isna(value) or value == '' or value is None:
        return None
    
    # Convert to string and strip whitespace
    value_str = str(value).strip()
    
    # Skip special values
    if value_str in ['car_km', 'Gesuch']:
        return None
    
    # Extract numeric part
    numeric_str = re.sub(r'[^\d]', '', value_str)
    
    try:
        if numeric_str:
            return int(numeric_str)
        return None
    except ValueError:
        return None

def clean_year(value):
    """Clean and extract year values"""
    if pd.isna(value) or value == '' or value is None:
        return None
    
    # Convert to string and strip whitespace
    value_str = str(value).strip()
    
    # Skip special values
    if value_str in ['car_year', '0 km', 'Gesuch']:
        return None
    
    # If it's already a 4-digit year
    if value_str.isdigit() and len(value_str) == 4:
        year = int(value_str)
        if 1900 <= year <= datetime.now().year + 1:
            return year
    
    # Try to extract a 4-digit year
    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', value_str)
    if year_match:
        return int(year_match.group(1))
    
    return None

def clean_price(value):
    """Clean and extract price values"""
    if pd.isna(value) or value == '' or value is None:
        return None
    
    # Convert to string and strip whitespace
    value_str = str(value).strip()
    
    # Skip special values
    if value_str == 'price':
        return None
    
    # Extract numeric part
    numeric_str = re.sub(r'[^\d]', '', value_str)
    
    try:
        if numeric_str:
            return int(numeric_str)
        return None
    except ValueError:
        return None

def calculate_days_on_market(date_posted, date_scrape):
    """Calculate days between posting and scraping"""
    if pd.isna(date_posted) or pd.isna(date_scrape):
        return None
    
    try:
        # Based on our diagnostic, the format is DD/MM/YYYY
        posted_date = datetime.strptime(str(date_posted).strip(), '%d/%m/%Y')
        scrape_date = datetime.strptime(str(date_scrape).strip(), '%d/%m/%Y')
        
        days = (scrape_date - posted_date).days
        if days >= 0:  # Ensure it's not negative
            return days
        return None
    except Exception as e:
        return None

def export_to_csv(df, filename='cars_clean.csv'):
    """Export the cleaned data to a CSV file"""
    try:
        df.to_csv(filename, index=False)
        print(f"Successfully exported data to {filename}")
        return True
    except Exception as e:
        print(f"Error exporting data to CSV: {str(e)}")
        return False

if __name__ == "__main__":
    # Clean the car data using MySQL connection from environment variables
    cleaned_data = clean_cars_data()
    
    # Preview the cleaned data
    if not cleaned_data.empty:
        print("\nPreview of cleaned data:")
        print(cleaned_data.head())
        
        # Data validation summary
        print("\nData validation summary:")
        print(f"Number of records: {len(cleaned_data)}")
        print(f"Non-null km values: {cleaned_data['km'].count()}")
        print(f"Non-null car_year values: {cleaned_data['car_year'].count()}")
        print(f"Non-null price values: {cleaned_data['price'].count()}")
        print(f"Non-null time_on_market values: {cleaned_data['time_on_market'].count()}")
    else:
        print("\nNo data to preview or validate!")
