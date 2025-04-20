import sqlite3
import pandas as pd
import re
from datetime import datetime

def clean_cars_data(db_path):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    
    print("Connected to database. Checking tables...")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"Available tables: {[t[0] for t in tables]}")
    
    # Extract data from the original table
    try:
        query = "SELECT * FROM cars"
        print(f"Executing query: {query}")
        df = pd.read_sql_query(query, conn)
        print(f"Retrieved {len(df)} rows from cars table")
        
        if len(df) == 0:
            print("WARNING: No data found in the cars table!")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error retrieving data: {str(e)}")
        return pd.DataFrame()
    
    # Group by ad_id and find the maximum date_scrape for each ad
    # Convert date_scrape to datetime format - removed deprecated parameter
    print("Converting dates to datetime format...")
    df['date_scrape_dt'] = pd.to_datetime(df['date_scrape'], errors='coerce')
    
    # Get the latest scrape date for each ad
    print("Finding latest scrape date for each ad...")
    latest_scrapes = df.groupby('id')['date_scrape_dt'].max().reset_index()
    latest_scrapes.rename(columns={'date_scrape_dt': 'max_date_scrape'}, inplace=True)
    
    # Merge with original data to get rows with the latest scrape date
    print("Merging with original data...")
    df = pd.merge(df, latest_scrapes, on='id')
    df = df[df['date_scrape_dt'] == df['max_date_scrape']]
    
    print(f"After filtering for latest scrape dates: {len(df)} rows")
    
    # Create a cleaned dataframe
    df_clean = df.copy()
    
    # Clean car_km: Extract numeric values
    print("Cleaning car_km values...")
    df_clean['km'] = df_clean['car_km'].apply(lambda x: extract_numeric(x) if pd.notna(x) else None)
    
    # Clean car_year: Convert to integer
    print("Cleaning car_year values...")
    df_clean['car_year'] = df_clean['car_year'].apply(
        lambda x: clean_year(x) if pd.notna(x) else None
    )
    
    # Clean price: Extract numeric values
    print("Cleaning price values...")
    df_clean['price'] = df_clean['price'].apply(lambda x: extract_numeric(x) if pd.notna(x) else None)
    
    # Calculate time_on_market in days using the latest scrape date
    print("Calculating time on market...")
    df_clean['time_on_market'] = df_clean.apply(
        lambda row: calculate_days_on_market(row['date_posted'], row['max_date_scrape']), 
        axis=1
    )
    
    # Create the new table structure - drop car_registration as requested
    columns_to_keep = [
        'id', 'ad_type', 'km', 'car_year', 'image_src', 'location', 
        'zipcode', 'date_posted', 'date_scrape', 'title', 'price', 'time_on_market'
    ]
    
    df_clean = df_clean[columns_to_keep]
    
    # Create the cleaned cars table
    if len(df_clean) > 0:
        print(f"Creating cars_clean table with {len(df_clean)} rows...")
        create_clean_table(conn, df_clean)
        
        # Export the cleaned data to CSV
        print("Exporting data to CSV file...")
        export_to_csv(df_clean)
    else:
        print("WARNING: No data to write to cars_clean table!")
    
    conn.close()
    print("Data cleaning completed successfully!")
    return df_clean

def clean_year(value):
    """Extract year as an integer"""
    if pd.isna(value):
        return None
        
    # First try to extract 4-digit year
    year_match = re.search(r'(19|20)\d{2}', str(value))
    if year_match:
        return int(year_match.group(0))
        
    # If that fails, try to extract any number
    digits = re.sub(r'\D', '', str(value))
    if digits and len(digits) <= 4:
        year_int = int(digits)
        # Validate reasonable car year
        if 1900 <= year_int <= datetime.now().year:
            return year_int
    
    return None

def extract_numeric(value):
    """Extract numeric values from strings"""
    if pd.isna(value):
        return None
    # Remove non-numeric characters except decimal points
    numeric_str = re.sub(r'[^\d.]', '', str(value))
    try:
        return float(numeric_str) if numeric_str else None
    except ValueError:
        return None

def calculate_days_on_market(date_posted, date_scrape):
    """Calculate days between posting and scraping"""
    if pd.isna(date_posted) or pd.isna(date_scrape):
        return None
    
    try:
        # Try different date formats
        date_formats = [
            '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d',
            '%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S'
        ]
        
        posted_date = None
        scrape_date = None
        
        # If inputs are already datetime objects, use them directly
        if isinstance(date_posted, pd.Timestamp):
            posted_date = date_posted
        else:
            for fmt in date_formats:
                try:
                    posted_date = datetime.strptime(str(date_posted), fmt)
                    break
                except ValueError:
                    continue
        
        if isinstance(date_scrape, pd.Timestamp):
            scrape_date = date_scrape
        else:
            for fmt in date_formats:
                try:
                    scrape_date = datetime.strptime(str(date_scrape), fmt)
                    break
                except ValueError:
                    continue
        
        if posted_date and scrape_date:
            return (scrape_date - posted_date).days
        return None
    except Exception:
        return None

def create_clean_table(conn, df):
    """Create the cleaned cars table"""
    # Drop the table if it exists
    conn.execute("DROP TABLE IF EXISTS cars_clean")
    
    # Create the new table
    conn.execute('''
    CREATE TABLE cars_clean (
        id TEXT PRIMARY KEY,
        ad_type TEXT,
        km NUMERIC,
        car_year INTEGER,
        image_src TEXT,
        location TEXT,
        zipcode TEXT,
        date_posted TEXT,
        date_scrape TEXT,
        title TEXT,
        price NUMERIC,
        time_on_market NUMERIC
    )
    ''')
    
    # Insert the cleaned data
    df.to_sql('cars_clean', conn, if_exists='append', index=False)
    conn.commit()

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
    # Path to your SQLite database
    database_path = 'ads.db'
    if not database_path:
        database_path = input("Enter the path to your SQLite database: ")
    
    # Clean the car data
    cleaned_data = clean_cars_data(database_path)
    
    # Optional: Preview the cleaned data
    if not cleaned_data.empty:
        print("\nPreview of cleaned data:")
        print(cleaned_data.head())
        
        # Data validation check
        print("\nData validation summary:")
        print(f"Number of records: {len(cleaned_data)}")
        print(f"Non-null km values: {cleaned_data['km'].count()}")
        print(f"Non-null car_year values: {cleaned_data['car_year'].count()}")
        print(f"Non-null price values: {cleaned_data['price'].count()}")
        print(f"Non-null time_on_market values: {cleaned_data['time_on_market'].count()}")
    else:
        print("\nNo data to preview or validate!")
