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
        # Get all data from the cars table
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
        first_row = df.iloc[0].to_dict() if not df.empty else {}
        if first_row.get('id') == 'id' and first_row.get('car_km') == 'car_km':
            print("First row appears to contain column headers - removing it")
            df = df.iloc[1:].reset_index(drop=True)
        
        # Process all data first
        print("Processing all data...")
        
        # Pre-process data in bulk
        df['km'] = df['car_km'].apply(clean_km)
        df['car_year_clean'] = df['car_year'].apply(clean_year)
        df['price_clean'] = df['price'].apply(clean_price)
        df['model'] = df['title'].apply(extract_dacia_model)
        
        # Count occurrences of each ad_id for days_tracked
        ad_id_counts = df['id'].value_counts().to_dict()
        # Ensure days_tracked is at least 1 even if there's only one occurrence
        df['days_tracked'] = df['id'].map(lambda x: max(1, ad_id_counts.get(x, 1)))
        
        # Convert date_scrape to datetime for sorting
        df['date_scrape_dt'] = pd.to_datetime(df['date_scrape'], format='%d/%m/%Y', errors='coerce')
        
        # Create a new dataframe with the processed data
        df_processed = pd.DataFrame({
            'id': df['id'],
            'ad_type': df['ad_type'],
            'km': df['km'],
            'car_year': df['car_year_clean'],
            'location': df['location'],
            'zipcode': df['zipcode'],
            'date_posted': df['date_posted'],
            'date_scrape': df['date_scrape'],
            'date_scrape_dt': df['date_scrape_dt'],
            'title': df['title'],
            'price': df['price_clean'],
            'model': df['model'],
            'days_tracked': df['days_tracked']
        })
        
        # Calculate price changes
        print("Analyzing price changes and time on market...")
        
        # Group by id to analyze price changes and time on market
        price_changes = {}
        time_on_market_data = {}
        first_seen_dates = {}  # Dictionary to track when each ad was first seen
        
        for ad_id, group in df_processed.groupby('id'):
            if len(group) > 1:
                # Sort by date
                group_sorted = group.sort_values('date_scrape_dt')
                
                # Get first and last scrape dates for time on market calculation
                first_date = group_sorted['date_scrape_dt'].iloc[0]
                last_date = group_sorted['date_scrape_dt'].iloc[-1]
                
                # Store the first seen date for this ad
                first_seen_dates[ad_id] = group_sorted['date_scrape'].iloc[0]
                
                # Calculate time on market directly (in days)
                days_on_market = (last_date - first_date).days
                if days_on_market >= 0:  # Ensure it's not negative
                    time_on_market_data[ad_id] = days_on_market
                else:
                    time_on_market_data[ad_id] = 0
                
                # Get first and last price
                first_price = group_sorted['price'].iloc[0]
                last_price = group_sorted['price'].iloc[-1]
                
                if pd.notnull(first_price) and pd.notnull(last_price):
                    price_diff = last_price - first_price
                    price_pct = (price_diff / first_price) * 100 if first_price > 0 else 0
                    
                    if price_diff < 0:
                        change = 'Decreased'
                        price_changes[ad_id] = (change, abs(price_diff), round(abs(price_pct), 2))
                    elif price_diff > 0:
                        change = 'Increased'
                        price_changes[ad_id] = (change, price_diff, round(price_pct, 2))
                    else:
                        price_changes[ad_id] = ('No Change', 0, 0)
                else:
                    price_changes[ad_id] = ('Unknown', None, None)
            else:
                price_changes[ad_id] = ('No Change', 0, 0)
                # For ads with only one occurrence, use the scrape date as first_seen
                first_seen_dates[ad_id] = group['date_scrape'].iloc[0]
        
        # Add price change information
        df_processed['price_change'] = df_processed['id'].map(lambda x: price_changes.get(x, ('Unknown', None, None))[0])
        df_processed['price_diff'] = df_processed['id'].map(lambda x: price_changes.get(x, ('Unknown', None, None))[1])
        df_processed['price_diff_pct'] = df_processed['id'].map(lambda x: price_changes.get(x, ('Unknown', None, None))[2])
        df_processed['time_on_market'] = df_processed['id'].map(time_on_market_data)
        df_processed['first_seen'] = df_processed['id'].map(first_seen_dates)  # Add first_seen column
        
        # Now keep only the latest version of each Ad_id
        print("Keeping only the latest version of each Ad_id...")
        df_clean = df_processed.sort_values('date_scrape_dt', ascending=False).groupby('id').first().reset_index()
        
        # Drop the temporary datetime column
        df_clean = df_clean.drop('date_scrape_dt', axis=1)
        
        # Set parameters
        current_year = datetime.now().year

        # Calculate derived features
        df_clean['age'] = current_year - df_clean['car_year']
        df_clean['price_per_km'] = df_clean['price'] / df_clean['km']
        df_clean['price_per_year'] = df_clean['price'] / (df_clean['age'] + 1)  # +1 to avoid division by zero
        
        # Add a column to indicate if an ad is new (posted on the date the script is run)
        today = datetime.now().strftime('%d/%m/%Y')
        # Mark listings that were posted today
        df_clean['posted_today'] = df_clean['date_posted'].apply(lambda x: 'Yes' if x == today else 'No')
        
        # Convert numeric columns to integers (removing decimals)
        numeric_columns = ['km', 'car_year', 'price', 'time_on_market', 'price_diff', 'days_tracked', 'age']
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

def export_to_csv(df, filename='cars_clean.csv'):
    """Export the cleaned data to a CSV file"""
    try:
        # Create a copy for export to avoid modifying the original dataframe
        export_df = df.copy()
        
        # First, fill NaN values with appropriate placeholders
        export_df = export_df.fillna('')
        
        # Convert integer columns to integers for CSV export
        integer_columns = ['km', 'car_year', 'price', 'time_on_market', 'price_diff', 'days_tracked', 'age']
        for col in integer_columns:
            if col in export_df.columns:
                # Only convert values that are not empty strings or NaN
                export_df[col] = export_df[col].apply(lambda x: int(float(x)) if x != '' and not pd.isna(x) else '')
        
        # Round floating point columns to 2 decimal places
        float_columns = ['price_per_km', 'price_per_year', 'price_diff_pct']
        for col in float_columns:
            if col in export_df.columns:
                # Only convert values that are not empty strings or NaN
                export_df[col] = export_df[col].apply(lambda x: round(float(x), 2) if x != '' and not pd.isna(x) else '')
        
        # Export to CSV without index
        export_df.to_csv(filename, index=False)
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

        # Average time on market
        # Convert to numeric first to handle any string values
        cleaned_data['time_on_market_num'] = pd.to_numeric(cleaned_data['time_on_market'], errors='coerce')
        avg_time_on_market = cleaned_data['time_on_market_num'].mean()
        print(f"Average time on market per ad: {avg_time_on_market:.1f} days")
        
        # Print some statistics about the data
        total_ads = len(cleaned_data)
        posted_today_count = (cleaned_data['posted_today'] == 'Yes').sum()
        price_change_count = (cleaned_data['price_change'] != 'No Change').sum()
        
        print(f"Total ads: {total_ads}")
        print(f"Ads posted today: {posted_today_count}")
        print(f"Ads with price changes: {price_change_count}")
        
    else:
        print("\nNo data to preview or validate!")
