"""Database manager for handling SQL database operations with real estate data."""

import os
import time
import math
import pyodbc
import pandas as pd
import unidecode
from urllib.parse import quote_plus
from dotenv import load_dotenv

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

def get_available_driver():
    """Get the most suitable available SQL Server ODBC driver."""
    drivers = pyodbc.drivers()
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

def get_connection_config():
    """Get database configuration from Streamlit secrets or environment variables."""
    if STREAMLIT_AVAILABLE:
        try:
            return {
                'server': st.secrets["database"]["DB_SERVER"],
                'database': st.secrets["database"]["DB_NAME"],
                'username': st.secrets["database"]["DB_USER"],
                'password': st.secrets["database"]["DB_PASSWORD"]
            }
        except:
            pass
    
    load_dotenv()
    return {
        'server': os.getenv('DB_SERVER'),
        'database': os.getenv('DB_NAME'),
        'username': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD')
    }

def get_connection_string():
    """Build connection string for database connection."""
    config = get_connection_config()
    driver = get_available_driver()
    
    # Add port if not present
    if ',' not in config['server'] and ':' not in config['server']:
        config['server'] = f"{config['server']},1433"
    
    params = {
        'DRIVER': f'{{{driver}}}',
        'SERVER': config['server'],
        'DATABASE': config['database'],
        'UID': config['username'],
        'PWD': config['password'],
        'Encrypt': 'yes',
        'TrustServerCertificate': 'yes'
    }
    
    conn_str = ';'.join([f'{k}={v}' for k, v in params.items()])
    return quote_plus(conn_str)

def get_db_connection():
    """Connect to Azure SQL database using Streamlit connection management when available."""
    if STREAMLIT_AVAILABLE:
        try:
            return st.connection(
                "azure_sql",
                type="sql",
                url=f"mssql+pyodbc:///?odbc_connect={get_connection_string()}",
                ttl=600  # Cache connection for 10 minutes
            )
        except Exception as e:
            st.error(f"Failed to create Streamlit connection: {e}")
    
    max_attempts = 3
    retry_delay = 5
    
    try:
        conn_str = get_connection_string()
        for attempt in range(max_attempts):
            try:
                return pyodbc.connect(conn_str)
            except pyodbc.Error as ex:
                if attempt < max_attempts - 1:
                    time.sleep(retry_delay)
                else:
                    raise
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

def create_table(df, table_name="butai"):
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
            sql_type = "NVARCHAR(MAX)" if col_name in ("ypatybes", "papildomos_patalpos", "papildoma_iranga", "apsauga") else "NVARCHAR(255)"
        else:
            sql_type = "NVARCHAR(255)"

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
    except Exception as e:
        print(f"Error during table creation: {e}")
    finally:
        if conn:
            conn.close()

def process_value(value, value_type='string'):
    """Convert value to specified type (int/float/string), handling edge cases."""
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

def add_new_rows(df, table_name="butai"):
    """Insert new DataFrame rows into database table, skipping existing records."""
    conn = get_db_connection()
    if conn is None:
        return

    cursor = conn.cursor()
    stats = {'inserted': 0, 'skipped': 0, 'failed': 0}
    
    int_columns = ['price', 'namo_numeris', 'kambariu_sk', 'aukstas', 'aukstu_sk', 'metai', 'buto_numeris']
    float_columns = ['plotas', 'latitude', 'longitude', 'distance_to_darzeliai',
                    'distance_to_mokyklos', 'distance_to_stoteles', 'distance_to_parduotuves']

    columns = ['url', 'price', 'namo_numeris', 'plotas', 'kambariu_sk', 'aukstas',
               'aukstu_sk', 'metai', 'pastato_tipas', 'sildymas', 'irengimas',
               'langu_orientacija', 'ypatybes', 'papildomos_patalpos',
               'papildoma_iranga', 'apsauga', 'latitude', 'longitude', 'city',
               'distance_to_darzeliai', 'distance_to_mokyklos', 'distance_to_stoteles',
               'distance_to_parduotuves', 'pastato_energijos_suvartojimo_klase',
               'buto_numeris', 'scrape_date']

    placeholders = ', '.join('?' * len(columns))
    insert_query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"

    for row in df.to_dict('records'):
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE url = ?", (row["url"],))
            if cursor.fetchone()[0] == 0:
                processed_row = {}
                for key, value in row.items():
                    value_type = 'int' if key in int_columns else 'float' if key in float_columns else 'string'
                    processed_row[key] = process_value(value, value_type)

                values = [processed_row.get(col, None) for col in columns]
                cursor.execute(insert_query, values)
                stats['inserted'] += 1
            else:
                stats['skipped'] += 1
        except Exception as e:
            print(f"Error processing row (URL: {row.get('url', 'N/A')}): {e}")
            stats['failed'] += 1

    conn.commit()
    conn.close()
    print(f"\nDatabase Operations Summary:")
    print(f"Successfully inserted: {stats['inserted']} records")
    print(f"Skipped (duplicates): {stats['skipped']} records")
    print(f"Failed to insert: {stats['failed']} records")

def get_data(query=None, params=None, table_name="butai"):
    """Fetch data from database using custom query or default SELECT."""
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()

    try:
        if query is None:
            query = f"SELECT * FROM {table_name}"
        return pd.read_sql(query, conn, params=params)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

def prepare_dataframe(df):
    """Clean DataFrame containing real estate listing data."""
    df = df.replace({"": None, "nan": None, "NaN": None, "NULL": None, "null": None})

    # Standardize column names
    df.columns = [unidecode.unidecode(col.lower().replace(" ", "_")) for col in df.columns]

    # Process numeric columns
    if 'plotas' in df.columns:
        df['plotas'] = df['plotas'].apply(
            lambda x: None if pd.isna(x) or x == '' else
            float(''.join(c for c in str(x).replace('m²', '').replace(',', '.').strip()
                         if c.isdigit() or c == '.'))
            if isinstance(x, (str, int, float)) else None
        )

    int_columns = ['price', 'namo_numeris', 'kambariu_sk', 'aukstas', 'aukstu_sk', 'metai', 'buto_numeris']
    for col in int_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    float_columns = ['plotas', 'latitude', 'longitude', 'distance_to_darzeliai',
                    'distance_to_mokyklos', 'distance_to_stoteles', 'distance_to_parduotuves']
    for col in float_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].where(pd.notna(df[col]), None)

    df.columns = df.columns.str.replace('.', '', regex=False)
    return df
