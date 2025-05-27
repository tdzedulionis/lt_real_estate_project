"""Database manager for handling SQL database operations with real estate data."""

import sys
import os
import pyodbc
import pandas as pd
import math
import time
import unidecode
from datetime import datetime
from ..config.settings import DATABASE_CONFIG

def get_available_driver():
    """Get the first available SQL Server ODBC driver."""
    drivers = pyodbc.drivers()
    
    # Preferred order of drivers
    preferred_drivers = [
        "ODBC Driver 18 for SQL Server",
        "ODBC Driver 17 for SQL Server", 
        "ODBC Driver 13 for SQL Server",
        "SQL Server Native Client 11.0",
        "SQL Server"
    ]
    
    for driver in preferred_drivers:
        if driver in drivers:
            return driver
    
    raise Exception(f"No suitable SQL Server ODBC driver found. Available drivers: {drivers}")

def get_connection_string():
    """Get the appropriate connection string based on available drivers."""
    
    try:
        driver = get_available_driver()
        print(f"Using ODBC driver: {driver}")
    except Exception as e:
        print(f"Driver detection error: {e}")
        raise
    
    connection_string = (
        f"Driver={{{driver}}};"
        f"Server={st.secrets['server']};"
        f"Database={st.secrets['database']};"
        f"Uid={st.secrets['username']};"
        f"Pwd={st.secrets['password']};"
        #"Encrypt=yes;"
        #"TrustServerCertificate=yes;"
        #"Connection Timeout=60;"
        #"Login Timeout=60;"
        #f"Command Timeout=300;"
    )
    
    return connection_string

def get_db_connection():
    """Connect to Azure SQL database with retry logic, returns Connection object or None."""
    max_attempts = 8
    attempt = 0
    

    try:
        conn_str = get_connection_string()
    except Exception as e:
        print(f"Failed to get connection string: {e}")
        return None
    
    while attempt < max_attempts:
        try:
            attempt += 1
            print(f"Connection attempt {attempt}/{max_attempts}...")
            conn = pyodbc.connect(conn_str)
            print("Database connection established successfully.")
            return conn
        except pyodbc.Error as ex:
            sqlstate = ex.args[0]
            print(f"Database connection failed (attempt {attempt}/{max_attempts}): SQL State: {sqlstate} - {ex}")
            if attempt < max_attempts:
                print("Retrying connection in 5 seconds...")
                time.sleep(5)
            else:
                print("Maximum retry attempts reached. Could not establish connection.")
                return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None
        
def create_table(df, table_name = "butai"):
    """Create database table based on DataFrame schema if it doesn't exist."""
    conn = get_db_connection()
    if conn is None:
        return

    columns_sql = []
    for col_name, dtype in df.dtypes.items():
        if col_name == "url":
            sql_type = "NVARCHAR(500) PRIMARY KEY"
        elif col_name == "scrape_date":
            sql_type = "NVARCHAR(255)"
        elif "datetime" in str(dtype).lower():
            sql_type = "DATETIME"
        elif "int" in str(dtype).lower():
            sql_type = "INT"
        elif "float" in str(dtype).lower():
            sql_type = "FLOAT"
        elif "object" in str(dtype).lower():
            if col_name in ("ypatybes", "papildomos_patalpos", "papildoma_iranga", "apsauga"):
                sql_type = "NVARCHAR(MAX)"
            else:
                sql_type = "NVARCHAR(255)"
        else:
            sql_type = "NVARCHAR(255)"
            print(f"Warning: Unrecognized data type for '{col_name}': {dtype}. Using NVARCHAR(255).")

        columns_sql.append(f"[{col_name}] {sql_type}")

    create_table_query = f"""
        IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = '{table_name}')
        BEGIN
            CREATE TABLE {table_name} (
                {', '.join(columns_sql)}
            );
        END
    """

    try:
        cursor = conn.cursor()
        cursor.execute(create_table_query)
        conn.commit()
        print(f"Table '{table_name}' checked/created successfully.")
    except pyodbc.Error as e:
        print(f"Error during table creation: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

def process_value(value, value_type='string'):
    """Convert value to specified type (int/float/string), handling edge cases and invalid values."""
    if value is None or pd.isna(value) or str(value).lower() in ('nan', 'null', '', '-', 'infinity', '-infinity'):
        return None

    try:
        if value_type == 'int':
            cleaned_value = ''.join(c for c in str(value) if c.isdigit())
            return int(cleaned_value) if cleaned_value else None
        elif value_type == 'float':
            if isinstance(value, (int, float)):
                return float(value) if not math.isnan(value) else None
            
            str_value = str(value).strip()
            cleaned_value = ''.join(c for c in str_value if c.isdigit() or c in '.-')
            
            if cleaned_value and not all(c in '.-' for c in cleaned_value):
                if cleaned_value.count('.') > 1:
                    cleaned_value = cleaned_value.replace('.', '', cleaned_value.count('.') - 1)
                float_value = float(cleaned_value)
                return float_value if not math.isnan(float_value) else None
            return None
        else:  # string
            return str(value) if value is not None else None
    except (ValueError, TypeError):
        return None

def add_new_rows(df, table_name = "butai"):
    """Insert new DataFrame rows into database table, skipping existing records."""
    conn = get_db_connection()
    if conn is None:
        return

    cursor = conn.cursor()
    successful_inserts = 0
    skipped_rows = 0
    failed_inserts = 0

    int_columns = [
        'price', 'namo_numeris', 'kambariu_sk',
        'aukstas', 'aukstu_sk', 'metai', 'buto_numeris'
    ]
    float_columns = [
        'plotas', 'latitude', 'longitude', 'distance_to_darzeliai',
        'distance_to_mokyklos', 'distance_to_stoteles', 'distance_to_parduotuves'
    ]

    data = df.to_dict('records')
    columns = [
        'url', 'price', 'namo_numeris', 'plotas', 'kambariu_sk', 'aukstas',
        'aukstu_sk', 'metai', 'pastato_tipas', 'sildymas', 'irengimas',
        'langu_orientacija', 'ypatybes', 'papildomos_patalpos',
        'papildoma_iranga', 'apsauga', 'latitude', 'longitude', 'city',
        'distance_to_darzeliai', 'distance_to_mokyklos', 'distance_to_stoteles',
        'distance_to_parduotuves', 'pastato_energijos_suvartojimo_klase',
        'buto_numeris', 'scrape_date'
    ]

    placeholders = ', '.join('?' * len(columns))
    insert_query = f"""
        INSERT INTO {table_name} (
            {', '.join(columns)}
        ) VALUES ({placeholders})
    """

    print("\nProcessing database records...")
    for i, row in enumerate(data):
        try:
            # Check if URL already exists
            cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE url = ?", (row["url"],))
            exists = cursor.fetchone()[0]

            sys.stdout.write(
                f"\rRow {i+1}/{len(data)} - Success: {successful_inserts}, "
                f"Skipped: {skipped_rows}, Failed: {failed_inserts}"
            )
            sys.stdout.flush()
            
            if exists == 0:
                processed_row = {}
                for key, value in row.items():
                    value_type = 'int' if key in int_columns else 'float' if key in float_columns else 'string'
                    processed_row[key] = process_value(value, value_type)

                values = [processed_row.get(col, None) for col in columns]
                cursor.execute(insert_query, values)
                successful_inserts += 1
            else:
                skipped_rows += 1

        except pyodbc.Error as e:
            print(f"\nError inserting row (URL: {row.get('url', 'N/A')}): {e}")
            failed_inserts += 1
        except Exception as e:
            print(f"\nUnexpected error processing row: {e}")
            failed_inserts += 1

    conn.commit()
    conn.close()
    
    print("\n\nDatabase Operations Summary:")
    print(f"Successfully inserted: {successful_inserts} records")
    print(f"Skipped (duplicates): {skipped_rows} records")
    print(f"Failed to insert: {failed_inserts} records")
    print("\nDatabase connection closed.")

def get_data(query = None, params = None, table_name = "butai"):
    """Fetch data from database using custom query or default SELECT, returns DataFrame."""
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()

    try:
        if query is None:
            query = f"SELECT * FROM {table_name}"
            params = None
        
        df = pd.read_sql(query, conn, params=params)
        print(f"Successfully retrieved {len(df)} rows from database.")
        return df

    except pyodbc.Error as e:
        print(f"Database error: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Unexpected error: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")
            
def prepare_dataframe(df):
    """
    Cleans a Pandas DataFrame containing real estate listing data.

    This function performs the following cleaning steps:
        1. Replaces empty strings and various representations of null/NaN with None.
        2. Standardizes column names: converts to lowercase, replaces spaces
           with underscores, and transliterates Unicode characters to ASCII.
        3. Converts specified columns to appropriate data types (integer, float).
        4. Removes dots from column names.

    Args:
        df (pd.DataFrame): The input DataFrame to be cleaned.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    df = df.replace({"": None, "nan": None, "NaN": None, "NULL": None, "null": None})

    new_columns = []
    for col in df.columns:
        new_col = col.lower().replace(" ", "_")
        new_col = unidecode.unidecode(new_col)
        new_columns.append(new_col)
    df.columns = new_columns

    int_columns = [
        'price', 'namo_numeris', 'kambariu_sk',
        'aukstas', 'aukstu_sk', 'metai', 'buto_numeris'
    ]

    if 'plotas' in df.columns:
        df['plotas'] = df['plotas'].apply(
            lambda x: None if pd.isna(x) or x == '' else
            float(''.join(c for c in str(x).replace('m²', '').replace(',', '.').strip()
                         if c.isdigit() or c == '.'))
            if isinstance(x, (str, int, float)) else None
        )

    for col in int_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    float_columns = [
        'plotas', 'latitude', 'longitude', 'distance_to_darzeliai',
        'distance_to_mokyklos', 'distance_to_stoteles', 'distance_to_parduotuves'
    ]
    for col in float_columns:
        if col in df.columns:
            # Convert to numeric and replace NaN with None
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].where(pd.notna(df[col]), None)

    df.columns = df.columns.str.replace('.', '', regex=False)

    return df
