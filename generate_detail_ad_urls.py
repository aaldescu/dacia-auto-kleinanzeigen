#!/usr/bin/env python3
import os
import sys
import pymysql
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_db_connection():
    """Connect to the MySQL database using environment variables."""
    return pymysql.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        user=os.getenv('DB_USER', 'root'),
        password=os.getenv('DB_PASSWORD', ''),
        db=os.getenv('DB_NAME', 'dacia_auto'),
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

def get_urls(filter_type='today'):
    """Get URLs from the cars table based on the filter type.
    
    Args:
        filter_type: 'today' to get URLs posted today, 'all' to get all URLs
        
    Returns:
        List of URLs with the base URL prepended
    """
    conn = get_db_connection()
    urls = []
    
    try:
        with conn.cursor() as cursor:
            if filter_type.lower() == 'today':
                # Get today's date in the format used in the database
                today = datetime.now().strftime("%d/%m/%Y")
                query = "SELECT href FROM cars WHERE date_posted = %s AND href IS NOT NULL"
                cursor.execute(query, (today,))
            else:  # 'all'
                query = "SELECT href FROM cars WHERE href IS NOT NULL"
                cursor.execute(query)
                
            results = cursor.fetchall()
            
            # Process the results
            base_url = "https://www.kleinanzeigen.de"
            for row in results:
                href = row.get('href')
                if href:
                    # Check if href already starts with the base URL
                    if not href.startswith(base_url):
                        full_url = f"{base_url}{href}"
                    else:
                        full_url = href
                    urls.append(full_url)
                    
    except Exception as e:
        print(f"Error retrieving URLs: {e}")
    finally:
        conn.close()
        
    return urls

def save_urls_to_file(urls, filter_type):
    """Save the URLs to a text file.
    
    Args:
        urls: List of URLs to save
        filter_type: The filter type used ('today' or 'all')
    """
    # Create a filename with the current date and filter type
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #filename = f"urls_{filter_type}_{timestamp}.txt"
    filename = f"urls_detailed_ad.txt"

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            for url in urls:
                f.write(f"{url}\n")
        print(f"Successfully saved {len(urls)} URLs to {filename}")
    except Exception as e:
        print(f"Error saving URLs to file: {e}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate a text file with URLs from the cars table')
    parser.add_argument('filter_type', nargs='?', default='today', choices=['today', 'all'],
                        help='Filter type: "today" for URLs posted today, "all" for all URLs (default: today)')
    
    # Parse arguments
    args = parser.parse_args()
    filter_type = args.filter_type
    
    print(f"Retrieving URLs with filter: {filter_type}")
    
    # Get the URLs
    urls = get_urls(filter_type)
    
    if not urls:
        print(f"No URLs found with filter: {filter_type}")
        return
    
    print(f"Found {len(urls)} URLs")
    
    # Save the URLs to a file
    save_urls_to_file(urls, filter_type)

if __name__ == "__main__":
    main()
