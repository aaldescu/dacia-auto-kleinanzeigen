import os
import json
import pymysql
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Constants
EXTRACT_DATA_FOLDER = "extract_data"
DETAILED_DATA_FOLDER = "extract_data_detailed"
SQL_MIGRATION_FOLDER = "sqlmigration"
CATEGORY_DATA_FOLDER = "extract_data_frontpage"

# Ensure  folder exists
os.makedirs(SQL_MIGRATION_FOLDER, exist_ok=True)
os.makedirs(DETAILED_DATA_FOLDER, exist_ok=True)
os.makedirs(EXTRACT_DATA_FOLDER, exist_ok=True)
os.makedirs(CATEGORY_DATA_FOLDER, exist_ok=True)

# Define mapping from JSON keys to database column names
FIELD_MAPPING = {
    'ad_id': 'ad_id',
    'seller_name': 'seller_name',
    'seller_type': 'seller_type',
    'active_since': 'active_since',
    'Marke': 'marke',
    'Modell': 'modell',
    'Kilometerstand': 'kilometerstand',
    'Fahrzeugzustand': 'fahrzeugzustand',
    'Erstzulassung': 'erstzulassung',
    'Kraftstoffart': 'kraftstoffart',
    'Leistung': 'leistung',
    'Getriebe': 'getriebe',
    'Fahrzeugtyp': 'fahrzeugtyp',
    'Anzahl Türen': 'anzahl_tueren',
    'Umweltplakette': 'umweltplakette',
    'Schadstoffklasse': 'schadstoffklasse',
    'Außenfarbe': 'aussenfarbe',
    'Material Innenausstattung': 'material_innenausstattung',
    'extras': 'extras',
    'description': 'description',
    'title': 'title',
    'price': 'price',
    'date_scrape': 'date_scrape',
    'HU bis': 'hu_bis',
    'HU Monat': 'hu_monat',
    'Art': 'art'
}

# Reverse mapping for SQL migration generation
GERMAN_TO_DB_FIELD_MAPPING = {
    'Marke': 'marke',
    'Modell': 'modell',
    'Kilometerstand': 'kilometerstand',
    'Fahrzeugzustand': 'fahrzeugzustand',
    'Erstzulassung': 'erstzulassung',
    'Kraftstoffart': 'kraftstoffart',
    'Leistung': 'leistung',
    'Getriebe': 'getriebe',
    'Fahrzeugtyp': 'fahrzeugtyp',
    'Anzahl Türen': 'anzahl_tueren',
    'Umweltplakette': 'umweltplakette',
    'Schadstoffklasse': 'schadstoffklasse',
    'Außenfarbe': 'aussenfarbe',
    'Material Innenausstattung': 'material_innenausstattung',
    'HU bis': 'hu_bis',
    'HU Monat': 'hu_monat',
    'Art': 'art',
}

# Load environment variables
load_dotenv()
 


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
        print(f"Connected to MySQL database: {DB_NAME}")
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
                    href TEXT,
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

def create_detailed_table():
    """Create the detailed table if it does not exist."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detailed (
                    ad_id VARCHAR(50),
                    seller_name VARCHAR(100),
                    seller_type VARCHAR(50),
                    active_since VARCHAR(50),
                    marke VARCHAR(50),
                    modell VARCHAR(50),
                    kilometerstand VARCHAR(50),
                    fahrzeugzustand VARCHAR(100),
                    erstzulassung VARCHAR(50),
                    kraftstoffart VARCHAR(50),
                    leistung VARCHAR(50),
                    getriebe VARCHAR(50),
                    fahrzeugtyp VARCHAR(50),
                    anzahl_tueren VARCHAR(20),
                    umweltplakette VARCHAR(50),
                    schadstoffklasse VARCHAR(50),
                    aussenfarbe VARCHAR(50),
                    material_innenausstattung VARCHAR(50),
                    hu_bis VARCHAR(50),
                    extras TEXT,
                    description TEXT,
                    title TEXT,
                    price VARCHAR(50),
                    date_scrape VARCHAR(20),
                    PRIMARY KEY (ad_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
        conn.commit()
        print("Detailed table created or already exists.")
    except Exception as e:
        print(f"Error creating detailed table: {e}")
    finally:
        conn.close()

def insert_detailed_ad(ad):
    """Insert a detailed ad into the database."""
    conn = get_db_connection()
    try:
        # Add current date as scrape date if not present
        if 'date_scrape' not in ad:
            ad['date_scrape'] = datetime.now().strftime("%Y-%m-%d")
            
        # Handle extras field which is a list in JSON
        if 'extras' in ad and isinstance(ad['extras'], list):
            ad['extras'] = ', '.join(ad['extras'])
        
        # Create a new dictionary with mapped keys
        mapped_ad = {
            'ad_id': ad.get('ad_id'),
            'seller_name': ad.get('seller_name'),
            'seller_type': ad.get('seller_type'),
            'active_since': ad.get('active_since'),
            'extras': ad.get('extras'),
            'description': ad.get('description'),
            'title': ad.get('title'),
            'price': ad.get('price'),
            'date_scrape': ad.get('date_scrape')
        }
        
        # Map German field names to database column names
        for german_key, db_key in GERMAN_TO_DB_FIELD_MAPPING.items():
            if german_key in ad:
                mapped_ad[db_key] = ad[german_key]
            else:
                mapped_ad[db_key] = None
                
        # Build the SQL query dynamically based on available fields
        columns = []
        placeholders = []
        update_clauses = []
        values = {}
        
        for key, value in mapped_ad.items():
            if value is not None:
                columns.append(key)
                placeholders.append(f'%({key})s')
                update_clauses.append(f'{key} = VALUES({key})')
                values[key] = value
                
        # Construct the SQL query
        sql = f"""
            INSERT INTO detailed ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            ON DUPLICATE KEY UPDATE
            {', '.join(update_clauses)}
        """
        
        with conn.cursor() as cursor:
            cursor.execute(sql, values)
            
        conn.commit()
        print(f"Inserted/updated detailed ad {ad.get('ad_id')} successfully.")
        conn.close()
        return True
    except Exception as e:
        print(f"Error inserting detailed ad: {e}")
        conn.close()
        return False


def insert_ad(ad):
    """Insert an ad into the database, allowing blanks for missing values."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # Ensure missing values are handled
            ad['href'] = ad.get('href', None)  # Set to None if missing
            ad['image_src'] = ad.get('image_src', None)  # Set to None if missing
            ad['location'] = ad.get('location', None)  # Set to None if missing
            ad['zipcode'] = ad.get('zipcode', None)  # Set to None if missing
            ad['date_posted'] = ad.get('date_posted', None)  # Set to None if missing
            ad['date_scrape'] = ad.get('date_scrape', None)  # Set to None if missing
            ad['title'] = ad.get('title', None)  # Set to None if missing
            ad['price'] = ad.get('price', None)  # Set to None if missing
            ad['car_year'] = ad.get('car_year', None)  # Set to None if missing

            cursor.execute("""
                INSERT INTO cars (id, ad_type, car_km, car_registration, car_year, image_src, location, zipcode, date_posted, date_scrape, title, price, href)
                VALUES (%(id)s, %(ad_type)s, %(car_km)s, %(car_registration)s, %(car_year)s, %(image_src)s, %(location)s, %(zipcode)s, %(date_posted)s, %(date_scrape)s, %(title)s, %(price)s, %(href)s)
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
                price = VALUES(price),
                href = VALUES(href)
            """, ad)
        conn.commit()
        print(f"Inserted/updated ad {ad['id']} successfully.")
        conn.close()
        return True
    except pymysql.err.IntegrityError:
        print(f"Ad {ad['id']} already exists for the same scrape time, skipping.")
        conn.close()
        return False
    except Exception as e:
        print(f"Error inserting ad {ad['id']}: {e}")
        conn.close()
        return False

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

def load_ads_json_files_and_insert():
    """Load all .json files from the folder and insert their data into the database."""
    # List all .json files in the directory
    json_files = [f for f in os.listdir(EXTRACT_DATA_FOLDER) if f.endswith('.json')]
    
    # Loop through each file
    for json_file in json_files:
        file_path = os.path.join(EXTRACT_DATA_FOLDER, json_file)
        
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
                
                print(f"Processed category file {json_file} successfully.")
            
            # Delete the JSON file after successful processing
            os.remove(file_path)
            print(f"Deleted file {json_file} after processing.")

        except Exception as e:
            print(f"Failed to process category file {json_file}: {e}")

def get_detailed_table_columns():
    """Get the column names from the detailed table in the database."""
    conn = get_db_connection()
    columns = []
    
    # Try method 1: SHOW COLUMNS
    try:
        with conn.cursor() as cursor:
            cursor.execute("SHOW COLUMNS FROM detailed")
            
            raw_results = cursor.fetchall()
            
            # Extract column names from the 'Field' key in each dictionary
            columns = [row['Field'] for row in raw_results]
            
            if columns:
                print(f"Successfully retrieved {len(columns)} columns using SHOW COLUMNS")
                conn.close()
                return columns
    except Exception as e:
        print(f"Error with SHOW COLUMNS method: {e}")
        conn.close()
        return columns

def generate_sql_migration(new_columns):
    """Generate SQL migration file for new columns.
    
    Args:
        new_columns: List of new column names found in JSON data
    """
    if not new_columns:
        return
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    migration_filename = f"{timestamp}_add_columns_to_detailed.sql"
    migration_path = os.path.join(SQL_MIGRATION_FOLDER, migration_filename)
    
    # Generate SQL statements
    sql_statements = []
    for column in new_columns:
        # Convert the JSON key to a database column name
        if column in GERMAN_TO_DB_FIELD_MAPPING:
            db_column = GERMAN_TO_DB_FIELD_MAPPING[column]
        else:
            # Convert to snake_case if not in mapping
            db_column = column.lower().replace(' ', '_')
        
        # Add to field mapping for future use
        if column not in FIELD_MAPPING:
            FIELD_MAPPING[column] = db_column
        
        if column not in GERMAN_TO_DB_FIELD_MAPPING and column[0].isupper():
            GERMAN_TO_DB_FIELD_MAPPING[column] = db_column
            
        # Default to VARCHAR(100) for new columns
        sql_statements.append(f"ALTER TABLE detailed ADD COLUMN {db_column} VARCHAR(100);")
    
    # Write SQL statements to file
    with open(migration_path, 'w') as f:
        f.write("-- Auto-generated migration for new columns in detailed table\n")
        f.write(f"-- Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("--\n")
        f.write("-- IMPORTANT: After executing this SQL, please also update the mapping constants in transform_load.py:\n")
        f.write("--\n")
        
        # Add mapping suggestions for each new column
        f.write("-- 1. Add to FIELD_MAPPING:\n")
        for column in new_columns:
            db_column = column.lower().replace(' ', '_') if column not in GERMAN_TO_DB_FIELD_MAPPING else GERMAN_TO_DB_FIELD_MAPPING[column]
            f.write(f"--    '{column}': '{db_column}',\n")
        
        f.write("--\n")
        f.write("-- 2. Add to GERMAN_TO_DB_FIELD_MAPPING:\n")
        for column in new_columns:
            if column[0].isupper():  # Only suggest German mappings for capitalized fields
                db_column = column.lower().replace(' ', '_') if column not in GERMAN_TO_DB_FIELD_MAPPING else GERMAN_TO_DB_FIELD_MAPPING[column]
                f.write(f"--    '{column}': '{db_column}',\n")
        
        f.write("\n")
        f.write("\n".join(sql_statements))
    
    print(f"Generated SQL migration file: {migration_path}")
    print("Please review and execute this SQL file to update your database schema.")

def check_json_structure_compatibility(ad_data, db_columns):
    """Check if the JSON structure is compatible with the database table.
    
    Args:
        ad_data: The ad data dictionary from JSON
        db_columns: List of column names in the database table
        
    Returns:
        tuple: (is_compatible, new_columns)
            is_compatible: True if compatible, False if new columns found
            new_columns: List of new columns found in the JSON data
    """
    # Check for new columns in the JSON data
    new_columns = []
    for key in ad_data.keys():
        # Skip standard fields that we already know about
        if key in FIELD_MAPPING or key == 'date_scrape':
            continue
            
        # Check if this key maps to a column in the database
        db_column = FIELD_MAPPING.get(key)
        if db_column is None or db_column not in db_columns:
            new_columns.append(key)
    
    # Generate SQL migration file if new columns are found
    if new_columns:
        generate_sql_migration(new_columns)
    
    return len(new_columns) == 0, new_columns

def load_detailed_files_and_insert():
    """Load all detailed ad .json files and insert them into the database."""
    # Get the column names from the database table
    db_columns = get_detailed_table_columns()
    if not db_columns:
        print("Could not retrieve database columns. Aborting.")
        return
        
    print(f"Database columns: {db_columns}")
    
    # List all .json files in the detailed data directory
    json_files = [f for f in os.listdir(DETAILED_DATA_FOLDER) if f.endswith('.json')]
    
    if not json_files:
        print("No JSON files found in the detailed data folder.")
        return
    
    processed_count = 0
    skipped_count = 0
    structure_issues = []
    
    # Loop through each file
    for json_file in json_files:
        file_path = os.path.join(DETAILED_DATA_FOLDER, json_file)
        
        try:
            # Open and load JSON data from file
            with open(file_path, 'r', encoding='utf-8') as f:
                ad = json.load(f)
                
            if not isinstance(ad, dict):
                print(f"Warning: Expected a dictionary in {file_path}, but got {type(ad)}")
                continue
            
            # Check if the JSON structure is compatible with the database table
            is_compatible, new_columns = check_json_structure_compatibility(ad, db_columns)
            if not is_compatible:
                print(f"Skipping {json_file} due to incompatible structure. New columns: {new_columns}")
                structure_issues.append((json_file, new_columns))
                skipped_count += 1
                continue
                
            # Add current date as scrape date if not present
            if 'date_scrape' not in ad:
                ad['date_scrape'] = datetime.now().strftime("%Y-%m-%d")
                
            # Insert into database
            if insert_detailed_ad(ad):
                processed_count += 1
                print(f"Processed detailed ad {json_file} successfully.")
            
                # Delete the JSON file after successful processing
                os.remove(file_path)
                print(f"Deleted file {json_file} after processing.")
            else:
                #skip file 
                continue

        except Exception as e:
            print(f"Failed to process detailed ad file {json_file}: {e}")
            
    print(f"Processed {processed_count} detailed ads. Skipped {skipped_count} due to structure issues.")
    
    if structure_issues:
        print("\nFiles skipped due to structure issues:")
        for file_name, new_cols in structure_issues:
            print(f"- {file_name}: New columns: {new_cols}")
        print("\nYou may need to update the database schema to include these new columns.")



# --- Run the database operations ---
create_table()
create_category_table()
create_detailed_table()

load_ads_json_files_and_insert()
load_category_files_and_insert()
load_detailed_files_and_insert()
