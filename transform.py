import os
import json
import sqlite3
import re
from datetime import datetime

# Path to the folder containing JSON files
DATA_FOLDER = 'extract_data/'
CATEGORY_DATA_FOLDER = 'extract_data_frontpage/'


# --- Data Transformations ---
def transform_ad_data(ad):
    """Transform the ad data."""
    # Extract car year
    ad["car_year"] = ad["car_registration"].strip()[-4:] if ad.get("car_registration") else None

    # Extract zipcode
    match = re.search(r"^\d{5}", ad["location"])
    ad["zipcode"] = match.group(0) if match else None

    # Format date_posted
    if ad.get("date_posted"):
        try:
            date_obj = datetime.strptime(ad["date_posted"], "%d.%m.%Y")  # Parse old format
            ad["date_posted"] = date_obj.strftime("%d/%m/%Y")  # Convert to new format
        except ValueError:
            ad["date_posted"] = None  # Handle invalid dates gracefully
    
    return ad

# --- Database Setup ---
DB_FILE = "ads.db"

def create_table():
    """Create the ads table if it does not exist."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cars (
                id TEXT,
                ad_type TEXT,
                car_km TEXT,
                car_registration TEXT,
                car_year TEXT,
                image_src TEXT,
                location TEXT,
                zipcode TEXT,
                date_posted TEXT,
                date_scrape TEXT,
                title TEXT,
                price TEXT,
                PRIMARY KEY (id, date_scrape)  -- Combination of ad_id and scrape time as unique key
            )
        """)
        conn.commit()

def create_category_table():
    """Create the ads table if it does not exist."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS categories (
                category TEXT,
                date_scrape TEXT,
                url TEXT,
                count TEXT,
                PRIMARY KEY (category, date_scrape)  -- Combination of categotry and data_scrape as unique key
            )
        """)
        conn.commit()

def insert_ad(ad):
    """Insert an ad into the database, allowing blanks for missing values."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        try:
            # Ensure missing values are handled
            ad['image_src'] = ad.get('image_src', None)  # Set to None if missing
            ad['location'] = ad.get('location', None)  # Set to None if missing
            ad['zipcode'] = ad.get('zipcode', None)  # Set to None if missing
            ad['date_posted'] = ad.get('date_posted', None)  # Set to None if missing
            ad['date_scrape'] = ad.get('date_scrape', None)  # Set to None if missing
            ad['title'] = ad.get('title', None)  # Set to None if missing
            ad['price'] = ad.get('price', None)  # Set to None if missing

            cursor.execute("""
                INSERT INTO cars (id, ad_type, car_km, car_registration, car_year, image_src, location, zipcode, date_posted, date_scrape, title, price)
                VALUES (:id, :ad_type, :car_km, :car_registration, :car_year, :image_src, :location, :zipcode, :date_posted, :date_scrape, :title, :price)
            """, ad)
            conn.commit()
            print(f"Inserted ad {ad['id']} successfully.")
        except sqlite3.IntegrityError:
            print(f"Ad {ad['id']} already exists for the same scrape time, skipping.")
        except Exception as e:
            print(f"Error inserting ad {ad['id']}: {e}")

def insert_category(ad):
    """Insert an category count into the database, allowing blanks for missing values."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        try:
            # Ensure missing values are handled
            ad['category'] = ad.get('category', None)  # Set to None if missing
            ad['url'] = ad.get('url', None)  # Set to None if missing
            ad['count'] = ad.get('count', None)  # Set to None if missing
            ad['date_scrape'] = ad.get('date_scrape', None)  # Set to None if missing
            

            cursor.execute("""
                INSERT INTO categories (category, url, count, date_scrape)
                VALUES (:category, :url, :count, :date_scrape)
            """, ad)
            conn.commit()
            print(f"Inserted category {ad['category']} successfully.")
        except sqlite3.IntegrityError:
            print(f"Category {ad['category']} already exists for the same scrape time, skipping.")
        except Exception as e:
            print(f"Error inserting ad {ad['category']}: {e}")

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


