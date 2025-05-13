import os
import pymysql
import pandas as pd
import re
import numpy as np
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
        
        # Create a copy of the original dataframe to analyze price changes
        df_original = df.copy()
        
        # Convert date_scrape to datetime for proper comparison
        df['date_scrape_dt'] = pd.to_datetime(df['date_scrape'], format='%d/%m/%Y', errors='coerce')
        
        # Get unique ad_ids
        unique_ad_ids = df['id'].unique()
        print(f"Found {len(unique_ad_ids)} unique Ad_ids")
        
        # Create a dataframe to store the results
        df_clean = pd.DataFrame()
        
        # Process each ad_id
        print("Processing each ad_id to calculate time_on_market and price changes...")
        
        # Lists to store data for each ad
        all_ids = []
        all_ad_types = []
        all_kms = []
        all_years = []
        all_image_srcs = []
        all_locations = []
        all_zipcodes = []
        all_date_posted = []
        all_date_scrape = []
        all_titles = []
        all_prices = []
        all_time_on_markets = []
        all_models = []
        all_price_changes = []
        all_price_diff_amounts = []
        all_price_diff_percentages = []
        all_days_tracked = []
        
        for ad_id in unique_ad_ids:
            # Get all rows for this ad_id
            ad_rows = df[df['id'] == ad_id].copy()
            
            # Sort by date (newest first)
            ad_rows = ad_rows.sort_values('date_scrape_dt', ascending=False)
            
            # Get the latest row
            latest_row = ad_rows.iloc[0]
            
            # Calculate time_on_market
            time_on_market = calculate_days_on_market(latest_row['date_posted'], latest_row['date_scrape'])
            
            # Extract Dacia model from title
            model = extract_dacia_model(latest_row['title'])
            
            # Calculate days tracked
            days_tracked = calculate_days_tracked(ad_id, df_original)
                
            # Determine price change
            price_change, price_diff, price_pct = determine_price_change(ad_id, df_original)
            
            # Clean values
            km = clean_km(latest_row['car_km'])
            year = clean_year(latest_row['car_year'])
            price = clean_price(latest_row['price'])
            
            # Append to lists
            all_ids.append(latest_row['id'])
            all_ad_types.append(latest_row['ad_type'])
            all_kms.append(km)
            all_years.append(year)
            all_image_srcs.append(latest_row['image_src'])
            all_locations.append(latest_row['location'])
            all_zipcodes.append(latest_row['zipcode'])
            all_date_posted.append(latest_row['date_posted'])
            all_date_scrape.append(latest_row['date_scrape'])
            all_titles.append(latest_row['title'])
            all_prices.append(price)
            all_time_on_markets.append(time_on_market)
            all_models.append(model)
            all_price_changes.append(price_change)
            all_price_diff_amounts.append(price_diff)
            all_price_diff_percentages.append(price_pct)
            all_days_tracked.append(days_tracked)
        
        # Create the cleaned dataframe
        df_clean['id'] = all_ids
        df_clean['ad_type'] = all_ad_types
        df_clean['km'] = all_kms
        df_clean['car_year'] = all_years
        df_clean['image_src'] = all_image_srcs
        df_clean['location'] = all_locations
        df_clean['zipcode'] = all_zipcodes
        df_clean['date_posted'] = all_date_posted
        df_clean['date_scrape'] = all_date_scrape
        df_clean['title'] = all_titles
        df_clean['price'] = all_prices
        df_clean['time_on_market'] = all_time_on_markets
        df_clean['model'] = all_models
        df_clean['price_change'] = all_price_changes
        df_clean['price_diff'] = all_price_diff_amounts
        df_clean['price_diff_pct'] = all_price_diff_percentages
        df_clean['days_tracked'] = all_days_tracked
        
        
        # Convert numeric columns to integers (removing decimals)
        numeric_columns = ['km', 'car_year', 'price', 'time_on_market', 'price_diff', 'days_tracked']
        for col in numeric_columns:
            # Convert to integer only if the column exists and has values
            if col in df_clean.columns and not df_clean[col].isna().all():
                # Convert to integer, but preserve NaN values
                df_clean[col] = df_clean[col].apply(lambda x: int(x) if pd.notnull(x) else x)
                
        # Convert empty values to empty strings for CSV export
        for col in df_clean.columns:
            if col != 'days_tracked':  # days_tracked should never be empty
                df_clean[col] = df_clean[col].fillna('')
        
        print(f"Successfully processed {len(df_clean)} unique Ad_ids")
        
        # Return the cleaned dataframe with only the latest version of each Ad_id
        return df_clean
            
    except Exception as e:
        print(f"Error retrieving data: {str(e)}")
        return pd.DataFrame()

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

def calculate_days_tracked(ad_id, df):
    """Calculate days tracked by looking at all occurrences of the ad_id in the dataframe"""
    # Get all occurrences of this ad_id from the dataframe
    all_occurrences = df[df['id'] == ad_id].copy()
    
    return len(all_occurrences)

def extract_dacia_model(title):
    """Extract Dacia model from the title"""
    if pd.isna(title) or title == '' or title is None:
        return None
        
    title = title.lower()
    
    # Define Dacia models to look for
    models = {
        'sandero': 'Sandero',
        'duster': 'Duster',
        'logan': 'Logan',
        'spring': 'Spring',
        'jogger': 'Jogger',
        'dokker': 'Dokker',
        'lodgy': 'Lodgy',
        'stepway': 'Stepway'
    }
    
    # Check for each model in the title
    for key, model in models.items():
        if key in title:
            return model
            
    return 'Other'

def determine_price_change(ad_id, df):
    """Determine if price has dropped, increased, or remained the same for an ad_id"""
    # Get all rows for this ad_id
    ad_rows = df[df['id'] == ad_id].copy()
    
    if len(ad_rows) <= 1:
        return 'No Change', None, None
    
    # Convert date_scrape to datetime for sorting
    ad_rows['date_scrape_dt'] = pd.to_datetime(ad_rows['date_scrape'], format='%d/%m/%Y', errors='coerce')
    
    # Sort by date (oldest to newest)
    ad_rows = ad_rows.sort_values('date_scrape_dt')
    
    # Convert price to numeric
    ad_rows['price_num'] = pd.to_numeric(ad_rows['price'], errors='coerce')
    
    # Get first and last price
    first_price = ad_rows['price_num'].iloc[0]
    last_price = ad_rows['price_num'].iloc[-1]
    
    # Calculate price difference
    if pd.isna(first_price) or pd.isna(last_price):
        return 'Unknown', None, None
        
    price_diff = last_price - first_price
    price_pct = (price_diff / first_price) * 100 if first_price > 0 else 0
    
    if price_diff < 0:
        return 'Decreased', abs(price_diff), round(abs(price_pct), 2)
    elif price_diff > 0:
        return 'Increased', price_diff, round(price_pct, 2)
    else:
        return 'No Change', 0, 0

def export_to_csv(df, filename='cars_clean.csv'):
    """Export the cleaned data to a CSV file"""
    try:
        # Export to CSV without index
        df.to_csv(filename, index=False)
        print(f"Successfully exported data to {filename}")
        return True
    except Exception as e:
        print(f"Error exporting data to CSV: {str(e)}")
        return False

if __name__ == "__main__":
    # Clean the car data using MySQL connection from environment variables
    cleaned_data = clean_cars_data()
    
    # Export the cleaned data to CSV
    if not cleaned_data.empty:
        export_to_csv(cleaned_data, 'cars_clean.csv')
    
    # Preview the cleaned data
    if not cleaned_data.empty:
        print("\nPreview of cleaned data:")
        print(cleaned_data.head())
        
        # Data validation summary
        print("\nData validation summary:")
        print(f"Number of unique Ad_ids: {len(cleaned_data)}")
        print(f"Non-null km values: {cleaned_data['km'].count()}")
        print(f"Non-null car_year values: {cleaned_data['car_year'].count()}")
        print(f"Non-null price values: {cleaned_data['price'].count()}")
        print(f"Non-null time_on_market values: {cleaned_data['time_on_market'].count()}")
        
        # Model distribution
        print("\nDacia model distribution:")
        model_counts = cleaned_data['model'].value_counts()
        for model, count in model_counts.items():
            print(f"{model}: {count} ads ({count/len(cleaned_data)*100:.1f}%)")
        
        # Price change summary
        print("\nPrice change summary:")
        price_change_counts = cleaned_data['price_change'].value_counts()
        for change, count in price_change_counts.items():
            print(f"{change}: {count} ads ({count/len(cleaned_data)*100:.1f}%)")
        
        # Average price decrease percentage
        decreased_prices = cleaned_data[cleaned_data['price_change'] == 'Decreased']
        if not decreased_prices.empty:
            avg_decrease_pct = decreased_prices['price_diff_pct'].mean()
            print(f"Average price decrease: {avg_decrease_pct:.2f}%")
        
        # Average price increase percentage
        increased_prices = cleaned_data[cleaned_data['price_change'] == 'Increased']
        if not increased_prices.empty:
            avg_increase_pct = increased_prices['price_diff_pct'].mean()
            print(f"Average price increase: {avg_increase_pct:.2f}%")
        
        # Average days tracked
        avg_days_tracked = cleaned_data['days_tracked'].mean()
        print(f"Average days tracked per ad: {avg_days_tracked:.1f} days")
    else:
        print("\nNo data to preview or validate!")
