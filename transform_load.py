import os
import json
import pymysql
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Path to the folder containing JSON files
DATA_FOLDER = 'extract_data/'
CATEGORY_DATA_FOLDER = 'extract_data_frontpage/'


# --- Data Transformations ---
def parse_german_date(date_str, scrape_date_str):
    """Parse German date formats like 'Gestern, 12:05' or 'Heute, 15:30'.
    
    Args:
        date_str: The date string to parse (e.g., 'Gestern, 12:05')
        scrape_date_str: The date when the data was scraped (format: 'MM/DD/YYYY')
        
    Returns:
        A formatted date string in 'DD/MM/YYYY' format
    """
    if not date_str or not scrape_date_str:
        return None
        
    # Parse the scrape date
    try:
        scrape_date = datetime.strptime(scrape_date_str, "%m/%d/%Y")
    except ValueError:
        print(f"Invalid scrape date format: {scrape_date_str}")
        return None
    
    # Handle German relative dates
    if "Gestern" in date_str:  # Yesterday
        base_date = scrape_date - timedelta(days=1)
    elif "Heute" in date_str:  # Today
        base_date = scrape_date
    elif "Vorgestern" in date_str:  # Day before yesterday
        base_date = scrape_date - timedelta(days=2)
    elif re.match(r"\d{2}\.\d{2}\.\d{4}", date_str):  # Format: DD.MM.YYYY
        try:
            return datetime.strptime(date_str, "%d.%m.%Y").strftime("%d/%m/%Y")
        except ValueError:
            return None
    else:
        # Try to extract a date like "15.05.2023" from the string
        date_match = re.search(r"(\d{1,2})\.(\d{1,2})\.", date_str)
        if date_match:
            day = int(date_match.group(1))
            month = int(date_match.group(2))
            year = scrape_date.year
            
            # If the extracted date is in the future, it's probably from last year
            extracted_date = datetime(year, month, day)
            if extracted_date > scrape_date:
                extracted_date = datetime(year-1, month, day)
                
            return extracted_date.strftime("%d/%m/%Y")
        return None
    
    # Extract time if available
    time_match = re.search(r"(\d{1,2}):(\d{2})", date_str)
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2))
        base_date = base_date.replace(hour=hour, minute=minute)
    
    return base_date.strftime("%d/%m/%Y")

def transform_ad_data(ad):
    """Transform the ad data."""
    # Extract car year
    ad["car_year"] = ad["car_registration"].strip()[-4:] if ad.get("car_registration") else None

    # Extract zipcode
    match = re.search(r"^\d{5}", ad["location"])
    ad["zipcode"] = match.group(0) if match else None

    # Format date_posted using the new German date parser
    if ad.get("date_posted") and ad.get("date_scrape"):
        ad["date_posted"] = parse_german_date(ad["date_posted"], ad["date_scrape"])
    
    return ad

# --- Database Setup ---
# Get database connection details from environment variables
DB_HOST = os.getenv('DB_HOST')
DB_PORT = int(os.getenv('DB_PORT'))
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

# Function to get database connection
def get_db_connection():
    """Create and return a connection to the MySQL database."""
    try:
        connection = pymysql.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        return connection
    except Exception as e:
        print(f"Error connecting to MySQL database: {e}")
        raise

def create_table():
    """Create the cars table if it does not exist."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cars (
                    id VARCHAR(50),
                    ad_type VARCHAR(20),
                    car_km VARCHAR(20),
                    car_registration VARCHAR(20),
                    car_year VARCHAR(4),
                    image_src TEXT,
                    location VARCHAR(100),
                    zipcode VARCHAR(10),
                    date_posted VARCHAR(20),
                    date_scrape VARCHAR(20),
                    title TEXT,
                    price VARCHAR(20),
                    PRIMARY KEY (id, date_scrape)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
        conn.commit()
        print("Cars table created or already exists.")
    except Exception as e:
        print(f"Error creating cars table: {e}")
    finally:
        conn.close()

def create_category_table():
    """Create the categories table if it does not exist."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS categories (
                    category VARCHAR(100),
                    date_scrape VARCHAR(20),
                    url TEXT,
                    count VARCHAR(20),
                    PRIMARY KEY (category, date_scrape)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
        conn.commit()
        print("Categories table created or already exists.")
    except Exception as e:
        print(f"Error creating categories table: {e}")
    finally:
        conn.close()

def insert_ad(ad):
    """Insert an ad into the database, allowing blanks for missing values."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # Ensure missing values are handled
            ad['image_src'] = ad.get('image_src', None)  # Set to None if missing
            ad['location'] = ad.get('location', None)  # Set to None if missing
            ad['zipcode'] = ad.get('zipcode', None)  # Set to None if missing
            ad['date_posted'] = ad.get('date_posted', None)  # Set to None if missing
            ad['date_scrape'] = ad.get('date_scrape', None)  # Set to None if missing
            ad['title'] = ad.get('title', None)  # Set to None if missing
            ad['price'] = ad.get('price', None)  # Set to None if missing
            ad['car_year'] = ad.get('car_year', None)  # Set to None if missing

            cursor.execute("""
                INSERT INTO cars (id, ad_type, car_km, car_registration, car_year, image_src, location, zipcode, date_posted, date_scrape, title, price)
                VALUES (%(id)s, %(ad_type)s, %(car_km)s, %(car_registration)s, %(car_year)s, %(image_src)s, %(location)s, %(zipcode)s, %(date_posted)s, %(date_scrape)s, %(title)s, %(price)s)
                ON DUPLICATE KEY UPDATE
                ad_type = VALUES(ad_type),
                car_km = VALUES(car_km),
                car_registration = VALUES(car_registration),
                car_year = VALUES(car_year),
                image_src = VALUES(image_src),
                location = VALUES(location),
                zipcode = VALUES(zipcode),
                date_posted = VALUES(date_posted),
                title = VALUES(title),
                price = VALUES(price)
            """, ad)
        conn.commit()
        print(f"Inserted/updated ad {ad['id']} successfully.")
    except pymysql.err.IntegrityError:
        print(f"Ad {ad['id']} already exists for the same scrape time, skipping.")
    except Exception as e:
        print(f"Error inserting ad {ad['id']}: {e}")
    finally:
        conn.close()

def insert_category(ad):
    """Insert a category count into the database, allowing blanks for missing values."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # Ensure missing values are handled
            ad['category'] = ad.get('category', None)  # Set to None if missing
            ad['url'] = ad.get('url', None)  # Set to None if missing
            ad['count'] = ad.get('count', None)  # Set to None if missing
            ad['date_scrape'] = ad.get('date_scrape', None)  # Set to None if missing

            cursor.execute("""
                INSERT INTO categories (category, url, count, date_scrape)
                VALUES (%(category)s, %(url)s, %(count)s, %(date_scrape)s)
                ON DUPLICATE KEY UPDATE
                url = VALUES(url),
                count = VALUES(count)
            """, ad)
        conn.commit()
        print(f"Inserted/updated category {ad['category']} successfully.")
    except pymysql.err.IntegrityError:
        print(f"Category {ad['category']} already exists for the same scrape time, skipping.")
    except Exception as e:
        print(f"Error inserting category {ad['category']}: {e}")
    finally:
        conn.close()

def load_json_files_and_insert():
    """Load all .json files from the folder and insert their data into the database."""
    # List all .json files in the directory
    json_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.json')]
    
    # Loop through each file
    for json_file in json_files:
        file_path = os.path.join(DATA_FOLDER, json_file)
        
        try:
            # Open and load JSON data from file
            with open(file_path, 'r', encoding='utf-8') as f:
                ads = json.load(f)
                
                # Iterate over each ad in the JSON
                for ad in ads:
                    transformed_ad = transform_ad_data(ad)
                    insert_ad(transformed_ad)
                
                print(f"Processed file {json_file} successfully.")
            
            # Delete the JSON file after successful processing
            os.remove(file_path)
            print(f"Deleted file {json_file} after processing.")

        except Exception as e:
            print(f"Failed to process file {json_file}: {e}")

def load_category_files_and_insert():
    """Load all .json files from the folder and insert their data into the database."""
    # List all .json files in the directory
    json_files = [f for f in os.listdir(CATEGORY_DATA_FOLDER) if f.endswith('.json')]
    
    # Loop through each file
    for json_file in json_files:
        file_path = os.path.join(CATEGORY_DATA_FOLDER, json_file)
        
        try:
            # Open and load JSON data from file
            with open(file_path, 'r', encoding='utf-8') as f:
                ads = json.load(f)
                
                # Iterate over each ad in the JSON
                for ad in ads:
                   
                    insert_category(ad)
                
                print(f"Processed file {json_file} successfully.")
            
            # Delete the JSON file after successful processing
            os.remove(file_path)
            print(f"Deleted file {json_file} after processing.")

        except Exception as e:
            print(f"Failed to process file {json_file}: {e}")

# --- Run the database operations ---
create_table()
create_category_table()

load_json_files_and_insert()
load_category_files_and_insert()


