"""Global settings and configuration for the aruodas_scraper package."""

from dotenv import load_dotenv
import os

# Try to import streamlit for cloud deployment
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


# Property type configurations
PROPERTY_TYPES = {
    'butai': {
        'table_name': 'butai',
        'output_dir': 'model_output/selling',
        'model_dir': 'model_output/selling/models',
        'viz_dir': 'model_output/selling/visualizations',
        'shap_dir': 'model_output/selling/shap_analysis'
    },
    'butu-nuoma': {
        'table_name': 'butai_rent',
        'output_dir': 'model_output/rental',
        'model_dir': 'model_output/rental/models',
        'viz_dir': 'model_output/rental/visualizations',
        'shap_dir': 'model_output/rental/shap_analysis'
    }
}

# Scraper settings
SCRAPER_CONFIG = {
    'max_pages': 100,
    'base_url': 'https://www.aruodas.lt',
    'category': 'butu-nuoma',  # 'butai' for selling, 'butu-nuoma' for rental
    'location': '',
    'max_retries': 4,
    'timeout': 10,
    'wait_time': 2  # seconds between requests
}

# Pipeline configuration
PIPELINE_CONFIG = {
    'scrape_data': True,
    'retrain_model': False  # Whether to retrain models after scraping
}

# Load environment variables
load_dotenv(override=True)

# Database settings
def get_database_config():
    if STREAMLIT_AVAILABLE:
        try:
            # For Streamlit Cloud deployment
            return {
                'server': st.secrets["database"]["DB_SERVER"],
                'database': st.secrets["database"]["DB_NAME"],
                'username': st.secrets["database"]["DB_USER"],
                'password': st.secrets["database"]["DB_PASSWORD"]
            }
        except:
            pass
    
    # For local development or fallback
    return {
        'server': os.getenv('DB_SERVER'),
        'database': os.getenv('DB_NAME'),
        'username': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD')
    }

DATABASE_CONFIG = get_database_config()

# Azure blob settings
BLOB_CONFIG = {
    'connection_string': os.getenv('BLOB_CONNECTION_STRING'),
    'container_name': os.getenv('BLOB_CONTAINER_NAME')
    }

