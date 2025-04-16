import sqlite3
import pandas as pd
import os
import time
import re
from openai import OpenAI
import json

def extract_car_details_with_gpt(db_path, api_key):
    """
    Extract structured car details from titles using GPT and add to cars_clean table
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    
    # Check if cars_clean table exists
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='cars_clean';")
    if not cursor.fetchone():
        print("Error: cars_clean table does not exist in the database.")
        conn.close()
        return pd.DataFrame()
    
    # Check if cars_extended table exists and get processed ids
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='cars_extended';")
    processed_ids = set()
    if cursor.fetchone():
        # First check if the columns exist
        cursor.execute("PRAGMA table_info(cars_extended)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'brand' in columns and 'model' in columns:
            query = "SELECT id FROM cars_extended WHERE brand IS NOT NULL OR model IS NOT NULL"
            processed_ids = set(pd.read_sql_query(query, conn)['id'].tolist())
        else:
            print("Note: cars_extended table exists but doesn't have all required columns yet.")
    
    # Load data from the cleaned cars table, excluding processed ids
    if processed_ids:
        placeholders = ','.join(["'{}'".format(id) for id in processed_ids])
        query = f"SELECT * FROM cars_clean WHERE id NOT IN ({placeholders})"
    else:
        query = "SELECT * FROM cars_clean"
    
    df = pd.read_sql_query(query, conn)
    
    print(f"Loaded {len(df)} unprocessed rows from cars_clean table")
    print(f"Skipping {len(processed_ids)} already processed ads")
    
    if len(df) == 0:
        print("No new records to process.")
        conn.close()
        return pd.DataFrame()
    
    # Ensure cars_extended table exists with correct schema
    print("Setting up cars_extended table...")
    create_extended_table(conn, df.head(0), pd.DataFrame(columns=['id']).head(0))
    
    # Prepare batch processing
    batch_size = 25  # Adjust based on API rate limits
    total_processed = 0
    
    print(f"Processing {len(df)} car titles in batches of {batch_size}...")
    
    # Process in batches to avoid API rate limits
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        batch_results = []
        print(f"Processing batch {i//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size}")
        
        for _, row in batch.iterrows():
            title = row['title']
            car_id = row['id']
            
            # Skip if title is empty
            if pd.isna(title) or title.strip() == '':
                batch_results.append({
                    'id': car_id,
                    'brand': None,
                    'model': None,
                    'trim': None,
                    'generation': None,
                    'engine_size': None,
                    'fuel_type': None,
                    'power': None,
                    'drivetrain': None,
                    'transmission': None,
                    'features': None,
                    'condition': None,
                    'extracted_year': None,
                    'emissions': None
                })
                continue
            
            # Extract information using GPT
            extracted_info = extract_with_gpt(client, title)
            
            # Add car_id to the extracted information
            extracted_info['id'] = car_id
            batch_results.append(extracted_info)
        
        # Convert batch results to DataFrame
        batch_df = pd.DataFrame(batch_results)
        
        # Ensure we don't have a conflict with the 'year' column
        if 'year' in batch_df.columns:
            batch_df.rename(columns={'year': 'extracted_year'}, inplace=True)
        
        # Insert batch results into cars_extended table
        print(f"Inserting batch of {len(batch_df)} rows into cars_extended table...")
        batch_with_original = pd.merge(batch, batch_df, on='id', how='left')
        batch_with_original.to_sql('cars_extended', conn, if_exists='append', index=False)
        conn.commit()
        
        total_processed += len(batch_df)
        print(f"Progress: {total_processed}/{len(df)} records processed")
        
        # Avoid hitting API rate limits
        time.sleep(1)
    
    print(f"Completed processing all {total_processed} records!")
    
    conn.close()
    print("GPT extraction completed and data stored in cars_extended table!")
    
    # Create a final dataframe by joining the two dataframes for return value
    # This is just for reference - we've already saved to the database
    final_df = pd.merge(df, extracted_df, on='id', how='left')
    return final_df

def extract_with_gpt(client, title):
    """
    Use GPT to extract structured information from a car title
    """
    try:
        # Define system prompt
        system_prompt = """
        You are a car listing data extractor that takes a car advertisement title and extracts structured information.
        Extract the following fields (return null if not found):
        - brand: car manufacturer
        - model: specific car model
        - trim: variant/trim level
        - generation: generation number of the model
        - engine_size: engine displacement in liters
        - fuel_type: gasoline, diesel, LPG, electric, hybrid, etc.
        - power: engine power in HP/PS/kW
        - drivetrain: 2WD, 4WD, FWD, RWD, AWD, 4x2, 4x4, etc.
        - transmission: automatic, manual, etc.
        - features: list of key features mentioned (comma separated)
        - condition: any mentioned condition like mileage, ownership, inspection
        - year: production year if mentioned
        - emissions: emissions standard if mentioned
        
        Respond with ONLY a JSON object containing these fields, nothing else.
        """
        
        # User message with the title
        user_message = f"Extract information from this car ad title: {title}"
        
        # Call the API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",  # Use appropriate model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"}
        )
        
        # Parse the response
        extracted_json = json.loads(response.choices[0].message.content)
        
        return extracted_json
    
    except Exception as e:
        print(f"Error extracting info from '{title}': {str(e)}")
        # Return empty structure in case of error
        return {
            'brand': None,
            'model': None,
            'trim': None,
            'generation': None,
            'engine_size': None,
            'fuel_type': None,
            'power': None,
            'drivetrain': None,
            'transmission': None,
            'features': None,
            'condition': None,
            'year': None,
            'emissions': None
        }

def create_extended_table(conn, original_df, extracted_df):
    """Create extended cars table with extracted information"""
    # Drop the table if it exists
    conn.execute("DROP TABLE IF EXISTS cars_extended")
    
    # Define all required columns including the extracted ones
    extracted_columns = [
        'brand', 'model', 'trim', 'generation', 'engine_size', 'fuel_type',
        'power', 'drivetrain', 'transmission', 'features', 'condition',
        'extracted_year', 'emissions'
    ]
    
    # Get original columns
    original_cols = list(original_df.columns)
    
    # Create the SQL for the new table with proper column definitions
    columns_sql = []
    
    # Add original columns with their types
    for col in original_cols:
        if col == 'id':
            columns_sql.append(f"{col} TEXT")
        elif col in ['km', 'price', 'time_on_market']:
            columns_sql.append(f"{col} NUMERIC")
        elif col == 'car_year':
            columns_sql.append(f"{col} INTEGER")
        else:
            columns_sql.append(f"{col} TEXT")
    
    # Add extracted columns (all as TEXT for simplicity)
    for col in extracted_columns:
        columns_sql.append(f"{col} TEXT")
    
    # Build the CREATE TABLE statement
    create_table_sql = f'''
    CREATE TABLE cars_extended (
        {', '.join(columns_sql)},
        PRIMARY KEY (id)
    )
    '''
    
    # Execute the create table statement
    conn.execute(create_table_sql)
    conn.commit()
    
    # If we have data to insert, do it
    if len(original_df) > 0:
        # Perform merge on 'id' column if we have extracted data
        if len(extracted_df) > 0:
            merged_df = pd.merge(original_df, extracted_df, on='id', how='left')
        else:
            merged_df = original_df
            # Add empty columns for extracted fields
            for col in extracted_columns:
                if col not in merged_df.columns:
                    merged_df[col] = None
        
        # Insert the merged data into the new table
        print(f"Inserting {len(merged_df)} rows into cars_extended table...")
        merged_df.to_sql('cars_extended', conn, if_exists='append', index=False)
        conn.commit()

def normalize_features(df):
    """
    Normalize the comma-separated features into a more structured format
    This is an optional enhancement that could be added
    """
    # Get unique features across all listings
    all_features = set()
    for features_str in df['features'].dropna():
        if isinstance(features_str, str):
            features = [f.strip() for f in features_str.split(',')]
            all_features.update(features)
    
    # Create binary columns for common features
    for feature in all_features:
        if len(feature) > 0:
            df[f'has_{feature.lower().replace(" ", "_")}'] = df['features'].apply(
                lambda x: 1 if isinstance(x, str) and feature.lower() in x.lower() else 0
            )
    
    return df

if __name__ == "__main__":
    # Path to your SQLite database
    database_path = input("Enter the path to your SQLite database: ")
    
    # OpenAI API key - you'll need to set this
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Enter your OpenAI API key: ")
    
    # Extract car details and extend the table
    extended_data = extract_car_details_with_gpt(database_path, api_key)
    
    if not extended_data.empty:
        # Print summary
        print("\nExtraction summary:")
        print(f"Total records processed: {len(extended_data)}")
        
        # Show extraction stats
        non_null_columns = ['brand', 'model', 'trim', 'engine_size', 'fuel_type', 'transmission', 'features']
        non_null_counts = {col: extended_data[col].count() if col in extended_data.columns else 0 
                          for col in non_null_columns}
        
        for col, count in non_null_counts.items():
            if col in extended_data.columns:
                percent = (count / len(extended_data)) * 100
                print(f"{col}: {count} values ({percent:.1f}%)")
        
        # Optional: Normalize features into individual columns
        # extended_data = normalize_features(extended_data)
        
        # Show sample data
        print("\nSample of extracted data:")
        sample_cols = ['id', 'title']
        for col in non_null_columns:
            if col in extended_data.columns:
                sample_cols.append(col)
        
        if len(extended_data) > 0:
            print(extended_data[sample_cols].head(3).to_string())
    else:
        print("No data was processed.")
