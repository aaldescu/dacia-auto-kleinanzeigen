import os
import sqlite3
import pandas as pd
import openai
import time
import random
import logging
import concurrent.futures
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"tokenize_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try to get OpenAI API key from environment, otherwise ask for it
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    logger.warning("OpenAI API key not found in environment variables. Prompting for input.")
    openai.api_key = input("Please enter your OpenAI API key: ").strip()
    if not openai.api_key:
        logger.error("No API key provided.")
        raise ValueError("OpenAI API key is required to run this script.")

class RateLimiter:
    """Rate limiter with exponential backoff for API calls"""
    def __init__(self, base_delay=0.05, max_delay=30.0, jitter=0.1):
        self.base_delay = base_delay  # Base delay between requests in seconds
        self.max_delay = max_delay    # Maximum delay between requests
        self.jitter = jitter          # Random jitter factor to add
        self.current_delay = base_delay
        self.last_request_time = 0
    
    def wait(self):
        """Wait the appropriate amount of time"""
        now = time.time()
        elapsed = now - self.last_request_time
        
        # If enough time has passed, reset the delay
        if elapsed > self.current_delay * 5:
            self.current_delay = self.base_delay
        
        # Calculate wait time with jitter
        wait_time = max(0, self.current_delay - elapsed)
        if wait_time > 0:
            # Add random jitter
            jitter_amount = random.uniform(-self.jitter, self.jitter) * wait_time
            wait_time += jitter_amount
            wait_time = max(0, wait_time)  # Ensure non-negative
            time.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def backoff(self):
        """Increase delay after a rate limit error"""
        self.current_delay = min(self.current_delay * 1.5, self.max_delay)
        logger.warning(f"Rate limit hit. Backing off. New delay: {self.current_delay:.2f}s")

def gpt_tokenize_title(title, rate_limiter, retry_count=3):
    """Tokenize a title using GPT with rate limiting and retries"""
    prompt = (
        "Split the following car ad title into individual tokens (words). "
        "Return the tokens as a Python list of strings.\n"
        f"Title: {title}\nTokens:"
    )
    
    for attempt in range(retry_count):
        try:
            # Wait according to rate limiter
            rate_limiter.wait()
            
            # Using the new OpenAI client format
            client = openai.OpenAI(api_key=openai.api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # or "gpt-4" if available
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0,
            )
            
            text = response.choices[0].message.content
            try:
                tokens = eval(text, {"__builtins__": {}})
                if isinstance(tokens, list):
                    return tokens
                else:
                    logger.warning(f"Non-list response for title: {title}. Got: {text}")
                    return []
            except Exception as e:
                logger.error(f"Error parsing response for title: {title}. Response: {text}. Error: {e}")
                return []
                
        except Exception as e:
            error_msg = str(e).lower()
            logger.error(f"Error processing title: {title}. Error: {str(e)}")
            
            # Check for rate limit errors in the newer OpenAI client
            if "rate limit" in error_msg or "too many requests" in error_msg:
                rate_limiter.backoff()
                if attempt < retry_count - 1:
                    logger.warning(f"Rate limit hit, retrying ({attempt+1}/{retry_count})")
                    continue
                else:
                    logger.error(f"Rate limit hit after {retry_count} attempts. Skipping title: {title}")
                    return []
            
            # Handle connection issues
            elif "timeout" in error_msg or "connection" in error_msg:
                rate_limiter.backoff()
                if attempt < retry_count - 1:
                    logger.warning(f"Connection issue, retrying ({attempt+1}/{retry_count})")
                    continue
            
            return []

def process_batch(batch_data, rate_limiter):
    """Process a batch of titles with a single rate limiter"""
    batch_results = []
    for row_id, title in batch_data:
        tokens = gpt_tokenize_title(title, rate_limiter)
        for token in tokens:
            batch_results.append({"id": row_id, "token": token})
    return batch_results, [row_id for row_id, _ in batch_data]

def process_in_batches(df, batch_size=20, checkpoint_interval=100, max_workers=5):
    """Process dataframe in parallel batches with checkpointing"""
    # Connect to the SQLite database
    conn = sqlite3.connect("ads.db")
    
    # Create the table if it doesn't exist
    conn.execute("""
    CREATE TABLE IF NOT EXISTS cars_title_tokens (
        id TEXT,
        token TEXT
    )
    """)
    
    # Check if we have a checkpoint file
    checkpoint_file = "tokenize_checkpoint.txt"
    processed_ids = set()
    
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            processed_ids = set(f.read().splitlines())
        logger.info(f"Resuming from checkpoint. {len(processed_ids)} IDs already processed.")
    
    # Filter out already processed IDs
    df_to_process = df[~df['id'].isin(processed_ids)]
    logger.info(f"Processing {len(df_to_process)} out of {len(df)} total records")
    
    # Prepare batches for parallel processing
    all_batches = []
    for i in range(0, len(df_to_process), batch_size):
        batch = df_to_process.iloc[i:i+batch_size]
        batch_data = [(row['id'], row['title']) for _, row in batch.iterrows()]
        all_batches.append(batch_data)
    
    logger.info(f"Split data into {len(all_batches)} batches for parallel processing")
    
    # Initialize shared variables
    token_rows = []
    processed_count = 0
    total_batches = len(all_batches)
    
    # Process batches in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a rate limiter for each worker
        rate_limiters = [RateLimiter(base_delay=0.05, max_delay=30.0) for _ in range(max_workers)]
        
        # Submit all batches for processing
        future_to_batch = {}
        for i, batch_data in enumerate(all_batches):
            # Use a different rate limiter for each worker to avoid conflicts
            limiter_index = i % max_workers
            future = executor.submit(process_batch, batch_data, rate_limiters[limiter_index])
            future_to_batch[future] = i
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_batch):
            batch_index = future_to_batch[future]
            try:
                batch_results, batch_ids = future.result()
                
                # Add results to our collection
                token_rows.extend(batch_results)
                processed_ids.update(batch_ids)
                processed_count += len(batch_ids)
                
                # Log progress
                logger.info(f"Completed batch {batch_index + 1}/{total_batches} - {processed_count} records processed")
                
                # Save checkpoint at intervals
                if processed_count % checkpoint_interval == 0 or len(token_rows) > 5000:
                    # Save tokens to database
                    if token_rows:
                        tokens_df = pd.DataFrame(token_rows)
                        tokens_df.to_sql("cars_title_tokens", conn, if_exists="append", index=False)
                        token_rows = []  # Clear after saving
                    
                    # Update checkpoint file
                    with open(checkpoint_file, "w") as f:
                        f.write("\n".join(processed_ids))
                    
                    logger.info(f"Checkpoint saved. Processed {processed_count} records.")
                    
            except Exception as e:
                logger.error(f"Error processing batch {batch_index}: {str(e)}")
    
    # Save any remaining tokens
    if token_rows:
        tokens_df = pd.DataFrame(token_rows)
        tokens_df.to_sql("cars_title_tokens", conn, if_exists="append", index=False)
    
    # Update final checkpoint
    with open(checkpoint_file, "w") as f:
        f.write("\n".join(processed_ids))
    
    logger.info(f"Processing complete. Total records processed: {processed_count}")
    conn.close()

if __name__ == "__main__":
    # Connect to the SQLite database
    conn = sqlite3.connect("ads.db")
    
    # Check if the cars_clean table exists
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='cars_clean'")
    if not cursor.fetchone():
        logger.error("cars_clean table not found in the database")
        conn.close()
        exit(1)
    
    # Read the cars_clean table
    logger.info("Reading cars_clean table from database")
    cars_clean = pd.read_sql_query("SELECT id, title FROM cars_clean", conn)
    conn.close()
    
    logger.info(f"Found {len(cars_clean)} records to process")
    
    # Process the data in batches with parallel processing
    # Adjust these parameters for optimal performance on your system:
    # - batch_size: Number of titles in each batch
    # - checkpoint_interval: How often to save progress
    # - max_workers: Number of parallel threads (don't set too high to avoid rate limits)
    process_in_batches(
        cars_clean, 
        batch_size=20, 
        checkpoint_interval=100,
        max_workers=5  # Adjust based on your API rate limits and CPU cores
    )

