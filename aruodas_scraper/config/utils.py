"""Utility functions for directory setup, configuration management, model uploading, and logging."""

import os
import logging
from azure.storage.blob import BlobServiceClient
from aruodas_scraper.config.settings import SCRAPER_CONFIG, PROPERTY_TYPES, BLOB_CONFIG

def setup_directories(property_config):
    """Create output, model, visualization, and SHAP analysis directories if they don't exist."""
    for dir_path in [
        property_config['output_dir'],
        property_config['model_dir'],
        property_config['viz_dir'],
        property_config['shap_dir']
    ]:
        os.makedirs(dir_path, exist_ok=True)
        
def get_property_config():
    """Return property-specific configuration dict based on current scraper category."""
    category = SCRAPER_CONFIG['category']
    if category not in PROPERTY_TYPES:
        raise ValueError(f"Invalid category: {category}. Must be one of: {', '.join(PROPERTY_TYPES.keys())}")
    return PROPERTY_TYPES[category]

def upload_models_to_blob(property_type):
    """Upload models to Azure Blob Storage with chunked upload for large files and progress tracking."""
    try:
        connection_string = BLOB_CONFIG['connection_string']
        container_name = BLOB_CONFIG['container_name']
        models_directory = f"model_output/{property_type}/models/"
        
        # Verify directory exists
        if not os.path.exists(models_directory):
            raise FileNotFoundError(f"Models directory not found: {models_directory}")
            
        # Verify we have models to upload
        model_files = os.listdir(models_directory)
        if not model_files:
            raise FileNotFoundError(f"No model files found in {models_directory}")
            
        # Initialize blob client
        client = BlobServiceClient.from_connection_string(connection_string)
        container_client = client.get_container_client(container_name)
        
        # Delete existing models for this property type
        print(f"Cleaning up existing {property_type} models in blob storage...")
        blob_list = container_client.list_blobs(name_starts_with=f"aruodas-models/{property_type}/")
        for blob in blob_list:
            container_client.delete_blob(blob.name)
        
        # Upload each model with progress tracking
        for model_name in model_files:
            model_path = os.path.join(models_directory, model_name)
            blob_path = f"{property_type}/{model_name}"
            file_size = os.path.getsize(model_path)
            
            print(f"Uploading {model_name} to {blob_path} (size: {file_size/1024/1024:.2f} MB)...")
            
            blob_client = container_client.get_blob_client(blob_path)
            
            # Use chunked upload for large files
            chunk_size = 4 * 1024 * 1024  # 4MB chunks
            if file_size > chunk_size:
                # Create block blob for chunked upload
                block_list = []
                
                with open(model_path, "rb") as data:
                    for i in range(0, file_size, chunk_size):
                        chunk = data.read(chunk_size)
                        block_id = f"{i:032d}"  # Create unique block ID
                        blob_client.stage_block(block_id, chunk)
                        block_list.append(block_id)
                        print(f"Uploaded chunk {len(block_list)}/{(file_size + chunk_size - 1)//chunk_size} "
                              f"({min(i + chunk_size, file_size)/file_size*100:.1f}%)")
                
                # Commit the blocks
                blob_client.commit_block_list(block_list)
            else:
                # Use simple upload for small files
                with open(model_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
            
            # Verify upload
            properties = blob_client.get_blob_properties()
            if properties.size != file_size:
                raise Exception(f"Upload verification failed for {model_name}: "
                              f"Expected size {file_size}, got {properties.size}")
                
            print(f"Successfully uploaded {model_name}")
        
        print(f"\nSuccessfully uploaded all {property_type} models to blob storage")
        
    except Exception as e:
        print(f"Error uploading models to blob storage: {str(e)}")
        raise
        
# Logging setup
def setup_logging():
    """Set up root logger with INFO level and console output."""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
