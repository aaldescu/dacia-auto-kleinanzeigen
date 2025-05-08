#!/usr/bin/env python3
import os
import sqlite3
import pymysql
import pandas as pd
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('migration.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# SQLite database path
SQLITE_DB_PATH = 'ads.db'

# MySQL connection details
MYSQL_HOST = os.getenv('DB_HOST', 'andreialdescu.com')
MYSQL_PORT = int(os.getenv('DB_PORT', 3306))
MYSQL_DB = os.getenv('DB_NAME', 'otzbgdpw_dacia')
MYSQL_USER = os.getenv('DB_USER', 'otzbgdpw_dacia')
MYSQL_PASSWORD = os.getenv('DB_PASSWORD', '')

# SQLite to MySQL type mapping
TYPE_MAPPING = {
    'TEXT': 'TEXT',
    'INTEGER': 'INT',
    'REAL': 'FLOAT',
    'NUMERIC': 'FLOAT',
    'BLOB': 'BLOB',
    '': 'TEXT'  # Default type
}

def get_sqlite_connection():
    """Connect to SQLite database"""
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        logger.info(f"Connected to SQLite database: {SQLITE_DB_PATH}")
        return conn
    except Exception as e:
        logger.error(f"Error connecting to SQLite database: {e}")
        raise

def get_mysql_connection():
    """Connect to MySQL database"""
    try:
        # Log connection details (without password)
        logger.info(f"Connecting to MySQL database: {MYSQL_DB} on {MYSQL_HOST}:{MYSQL_PORT} as {MYSQL_USER}")
        
        # Check if password is empty
        if not MYSQL_PASSWORD:
            logger.error("MySQL password is empty. Check your .env file.")
            raise ValueError("MySQL password cannot be empty")
        
        conn = pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=10
        )
        logger.info(f"Successfully connected to MySQL database: {MYSQL_DB} on {MYSQL_HOST}")
        return conn
    except pymysql.err.OperationalError as e:
        logger.error(f"MySQL connection error: {e}")
        logger.error(f"Check your MySQL credentials and ensure the server is accessible from this machine.")
        raise
    except Exception as e:
        logger.error(f"Error connecting to MySQL database: {e}")
        raise

def get_sqlite_tables(sqlite_conn):
    """Get list of tables in SQLite database"""
    cursor = sqlite_conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [table[0] for table in cursor.fetchall() if table[0] != 'sqlite_sequence']
    logger.info(f"Found {len(tables)} tables in SQLite database: {tables}")
    return tables

def get_sqlite_table_schema(sqlite_conn, table_name):
    """Get schema for a SQLite table"""
    cursor = sqlite_conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    return columns

def create_mysql_table(mysql_conn, table_name, sqlite_schema):
    """Create table in MySQL based on SQLite schema"""
    cursor = mysql_conn.cursor()
    
    # Drop table if it exists
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    
    # Build CREATE TABLE statement
    create_table_sql = f"CREATE TABLE {table_name} (\n"
    
    # Process columns
    columns = []
    primary_keys = []
    
    for col in sqlite_schema:
        col_id, col_name, col_type, not_null, default_value, is_pk = col
        
        # Map SQLite type to MySQL type
        mysql_type = TYPE_MAPPING.get(col_type.upper(), 'TEXT')
        
        # For VARCHAR types, add a reasonable length
        if mysql_type == 'TEXT' and col_name not in ['image_src', 'title', 'url']:
            mysql_type = 'VARCHAR(255)'
        
        # Build column definition
        column_def = f"    {col_name} {mysql_type}"
        
        if not_null:
            column_def += " NOT NULL"
            
        if default_value is not None:
            column_def += f" DEFAULT '{default_value}'"
            
        if is_pk:
            primary_keys.append(col_name)
            
        columns.append(column_def)
    
    # Add primary key constraint if any
    if primary_keys:
        primary_key_clause = f"    PRIMARY KEY ({', '.join(primary_keys)})"
        columns.append(primary_key_clause)
    
    create_table_sql += ",\n".join(columns)
    create_table_sql += "\n) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci"
    
    # Create the table
    try:
        cursor.execute(create_table_sql)
        mysql_conn.commit()
        logger.info(f"Created table {table_name} in MySQL")
        return True
    except Exception as e:
        logger.error(f"Error creating table {table_name} in MySQL: {e}")
        logger.error(f"SQL: {create_table_sql}")
        return False

def migrate_table_data(sqlite_conn, mysql_conn, table_name):
    """Migrate data from SQLite table to MySQL table"""
    try:
        # Read data from SQLite
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", sqlite_conn)
        logger.info(f"Read {len(df)} rows from SQLite table {table_name}")
        
        if len(df) == 0:
            logger.warning(f"No data found in SQLite table {table_name}")
            return True
        
        # Insert data into MySQL
        cursor = mysql_conn.cursor()
        
        # Special handling for cars_title_tokens table
        if table_name == 'cars_title_tokens':
            # Batch insert for better performance and error handling
            return batch_insert_tokens(mysql_conn, df, table_name)
        
        # Process each row
        rows_inserted = 0
        errors = 0
        for _, row in df.iterrows():
            # Replace NaN values with None for SQL compatibility
            row_dict = row.where(pd.notnull(row), None).to_dict()
            
            # Create placeholders for the SQL query
            placeholders = ", ".join(["%s"] * len(row_dict))
            columns = ", ".join(row_dict.keys())
            
            # Create the SQL query
            sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
            
            try:
                # Execute the query
                cursor.execute(sql, list(row_dict.values()))
                rows_inserted += 1
                
                # Commit in smaller batches to avoid large transactions
                if rows_inserted % 1000 == 0:
                    mysql_conn.commit()
                    logger.info(f"Inserted {rows_inserted} rows so far into {table_name}")
            except pymysql.err.IntegrityError as e:
                # Handle duplicate key errors gracefully
                logger.warning(f"Duplicate key error for {table_name}: {e}")
                errors += 1
                continue
            except Exception as e:
                logger.error(f"Error inserting row into MySQL table {table_name}: {e}")
                logger.error(f"Row data: {row_dict}")
                errors += 1
                continue
        
        # Commit the final batch
        mysql_conn.commit()
        logger.info(f"Inserted {rows_inserted} rows into MySQL table {table_name} with {errors} errors")
        return True
    except Exception as e:
        logger.error(f"Error migrating data for table {table_name}: {e}")
        return False


def batch_insert_tokens(mysql_conn, df, table_name):
    """Special batch insert function for the cars_title_tokens table"""
    try:
        cursor = mysql_conn.cursor()
        
        # Create a temporary table for bulk loading
        temp_table = f"{table_name}_temp"
        cursor.execute(f"DROP TABLE IF EXISTS {temp_table}")
        cursor.execute(f"CREATE TABLE {temp_table} LIKE {table_name}")
        
        # Prepare data for batch insert
        batch_size = 1000
        total_rows = len(df)
        rows_inserted = 0
        errors = 0
        
        for i in range(0, total_rows, batch_size):
            batch_df = df.iloc[i:i+batch_size]
            
            if len(batch_df) == 0:
                continue
                
            # Create batch insert SQL
            values_list = []
            for _, row in batch_df.iterrows():
                # Replace NaN values with None
                row_dict = row.where(pd.notnull(row), None).to_dict()
                
                # Escape string values
                values = []
                for val in row_dict.values():
                    if val is None:
                        values.append("NULL")
                    elif isinstance(val, str):
                        # Double escape single quotes for MySQL
                        escaped_val = val.replace("'", "''")
                        values.append(f"'{escaped_val}'")
                    else:
                        values.append(str(val))
                        
                values_str = f"({', '.join(values)})"
                values_list.append(values_str)
            
            # Execute batch insert
            if values_list:
                columns = ", ".join(batch_df.columns)
                values_sql = ", ".join(values_list)
                sql = f"INSERT INTO {temp_table} ({columns}) VALUES {values_sql}"
                
                try:
                    cursor.execute(sql)
                    mysql_conn.commit()
                    rows_inserted += len(batch_df)
                    logger.info(f"Batch inserted {len(batch_df)} rows into {temp_table}, total: {rows_inserted}/{total_rows}")
                except Exception as e:
                    logger.error(f"Error batch inserting into {temp_table}: {e}")
                    errors += len(batch_df)
                    
                    # Fall back to row-by-row insert for this batch
                    row_by_row_inserted = 0
                    for _, row in batch_df.iterrows():
                        row_dict = row.where(pd.notnull(row), None).to_dict()
                        placeholders = ", ".join(["%s"] * len(row_dict))
                        columns = ", ".join(row_dict.keys())
                        sql = f"INSERT INTO {temp_table} ({columns}) VALUES ({placeholders})"
                        
                        try:
                            cursor.execute(sql, list(row_dict.values()))
                            row_by_row_inserted += 1
                        except Exception as e2:
                            logger.error(f"Row-by-row insert error: {e2}")
                            logger.error(f"Problem row: {row_dict}")
                    
                    mysql_conn.commit()
                    rows_inserted += row_by_row_inserted
                    errors -= row_by_row_inserted
                    logger.info(f"Row-by-row fallback inserted {row_by_row_inserted} rows")
        
        # Replace the original table with the temp table
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        cursor.execute(f"RENAME TABLE {temp_table} TO {table_name}")
        mysql_conn.commit()
        
        logger.info(f"Successfully migrated {rows_inserted} tokens to {table_name} with {errors} errors")
        return True
    except Exception as e:
        logger.error(f"Error in batch token migration: {e}")
        return False

def main():
    """Main migration function"""
    logger.info("Starting migration from SQLite to MySQL")
    
    try:
        # Connect to databases
        sqlite_conn = get_sqlite_connection()
        
        # Test MySQL connection before proceeding
        try:
            mysql_conn = get_mysql_connection()
            # Test the connection with a simple query
            cursor = mysql_conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            if not result or result.get('1') != 1:
                raise Exception("MySQL connection test failed")
            logger.info("MySQL connection test successful")
        except Exception as e:
            logger.error(f"MySQL connection test failed: {e}")
            logger.error("Please check your MySQL credentials and server availability")
            return
        
        # Get list of tables
        tables = get_sqlite_tables(sqlite_conn)
        
        # Allow specifying specific tables to migrate
        import sys
        specific_tables = []
        if len(sys.argv) > 1:
            specific_tables = sys.argv[1:]
            logger.info(f"Only migrating specified tables: {specific_tables}")
            tables = [t for t in tables if t in specific_tables]
        
        # Process each table
        for table_name in tables:
            logger.info(f"Processing table: {table_name}")
            
            # Get table schema
            schema = get_sqlite_table_schema(sqlite_conn, table_name)
            
            # Create table in MySQL
            if create_mysql_table(mysql_conn, table_name, schema):
                # Migrate data
                success = migrate_table_data(sqlite_conn, mysql_conn, table_name)
                if not success:
                    logger.error(f"Failed to migrate data for table {table_name}")
                    continue
            else:
                logger.error(f"Failed to create table {table_name} in MySQL")
        
        logger.info("Migration completed successfully")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Close connections
        if 'sqlite_conn' in locals():
            sqlite_conn.close()
        if 'mysql_conn' in locals():
            mysql_conn.close()

if __name__ == "__main__":
    main()
