#!/usr/bin/env python3
"""Main script for running the Aruodas.lt scraper and data processing pipeline."""

import os
import pandas as pd
from aruodas_scraper.scraper.aruodas_scraper import AruodasScraper
from aruodas_scraper.database.database_manager import create_table, add_new_rows, prepare_dataframe, get_data
from aruodas_scraper.preprocessing.data_manipulation import clean_data, engineer_features, convert_numeric_if_possible
from aruodas_scraper.training.model_builder import final_ml_analysis
from aruodas_scraper.config.settings import SCRAPER_CONFIG, PIPELINE_CONFIG
from aruodas_scraper.config.utils import setup_directories, get_property_config, upload_models_to_blob, setup_logging

def main():
    """Main function to run the scraping and optional model training pipeline."""
    
    # Set up logging and get property configuration
    setup_logging()
    property_config = get_property_config()
    setup_directories(property_config)
    
    # Convert pages to int if not 'all'
    max_pages = SCRAPER_CONFIG['max_pages']
    if max_pages != 'all':
        max_pages = int(max_pages)
    
    # Database table
    table_name = property_config['table_name']
    
    try:
        print("\n=== Pipeline Start ===")
        print(f"Property Type: {'Rental' if SCRAPER_CONFIG['category'] == 'butu-nuoma' else 'Selling'}")
        print(f"Mode: {'Full Pipeline' if PIPELINE_CONFIG['scrape_data'] else 'Retrain models only'}")
        
        if PIPELINE_CONFIG['scrape_data']:
            # 1. Data Scraping (if enabled)
            print("\n=== 1. Data Collection ===")
            scraper = AruodasScraper(
                category=SCRAPER_CONFIG['category'],
                where=SCRAPER_CONFIG['location']
            )
            
            print(f"Starting scraper for category '{SCRAPER_CONFIG['category']}' in location '{SCRAPER_CONFIG['location']}'")
            print(f"Will scrape {SCRAPER_CONFIG['max_pages']} pages")
            
            scraper.scrape_data(max_pages=max_pages)
            scraped_data = scraper.get_data()
            scraper.close()
            
            if not scraped_data:
                print("No data was scraped. Exiting.")
                return
            
            # 2. Data Processing and Storage
            print("\n=== 2. Data Storage ===")
            df = pd.DataFrame(scraped_data)
            print(f"Successfully collected {len(df)} listings!")
            
            print("Preparing scraped data for database...")
            df = prepare_dataframe(df)
            
            # Create table if it doesn't exist
            print(f"Creating/checking table '{table_name}'...")
            create_table(df, table_name)
            
            print("Inserting new records into database...")
            add_new_rows(df, table_name)
            print("\nScraping and data storage completed successfully!")
        
        # 3. Model Training (if enabled)
        if PIPELINE_CONFIG['retrain_model']:
            print("\n=== Starting Model Training ===")
            print("Fetching all data from database...")
            
            # Get all data from database
            df = get_data(table_name = table_name)
            print(f"Retrieved {len(df)} records for training")
            
            print("\nCleaning data...")
            df_cleaned = clean_data(df)
            df_cleaned = df_cleaned.replace({"Not Specified": None})
            for column_name in df_cleaned.columns:
                df_cleaned = convert_numeric_if_possible(df_cleaned, column_name)
            
            print("Engineering features...")
            df_features = engineer_features(df_cleaned)
            
            print("\nEnsuring output directories exist...")
            setup_directories(property_config)
            
            print("\nTraining and evaluating models...")
            try:
                final_ml_analysis(
                    df_features,
                    output_dir=property_config['output_dir'],
                    log=True
                )
                print(f"\nModel training completed successfully!")
                
                # Verify models were created
                models_dir = property_config['model_dir']
                if not os.path.exists(models_dir):
                    raise FileNotFoundError(f"Model directory not found after training: {models_dir}")
                
                model_files = os.listdir(models_dir)
                expected_models = ['ensemble.pkl', 'gradient_boosting.pkl', 'lightgbm.pkl', 'xgboost.pkl']
                missing_models = [m for m in expected_models if m not in model_files]
                if missing_models:
                    raise FileNotFoundError(f"Missing expected model files: {', '.join(missing_models)}")
                print(f"Found all required model files in: {models_dir}")
                
                print("\nUploading models to Azure Blob Storage...")
                current_category = SCRAPER_CONFIG['category']
                property_type = 'rental' if current_category == 'butu-nuoma' else 'selling'
                upload_models_to_blob(property_type)
                print(f"\nModel uploading completed successfully!")
            except Exception as e:
                print(f"\nError during model training and upload: {e}")
                raise
            
        print("\n=== Pipeline Completed Successfully! ===")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
