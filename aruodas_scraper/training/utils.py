"""Utility functions for model training and evaluation."""

import os

def get_property_config(output_dir, property_types):
    """Find and return property configuration dictionary matching the output directory path."""
    property_config = None
    for config in property_types.values():
        if config['output_dir'] == output_dir:
            property_config = config
            break
    
    if property_config is None:
        raise ValueError(f"Could not find property configuration for output directory: {output_dir}")
    
    return property_config

def create_model_directories(property_config):
    """Create necessary directories for model outputs."""
    required_dirs = {
        'output_dir': property_config['output_dir'],
        'model_dir': property_config['model_dir'],
        'viz_dir': property_config['viz_dir'],
        'shap_dir': property_config['shap_dir']
    }
    
    for dir_name, dir_path in required_dirs.items():
        try:
            print(f"Creating {dir_name} at: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
            
            # Verify directory was created
            if not os.path.exists(dir_path):
                raise OSError(f"Failed to create {dir_name} directory at: {dir_path}")
            else:
                print(f"Successfully created/verified {dir_name} directory")
                
        except Exception as e:
            print(f"Error creating {dir_name} directory: {str(e)}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Directory contents: {os.listdir(os.path.dirname(dir_path))}")
            raise OSError(f"Failed to create/verify {dir_name} directory: {str(e)}")

def prepare_features(df, target='price'):
    """Split DataFrame columns into numeric and categorical features, removing price-related columns."""
    # Drop unnecessary price columns
    cols_to_drop = [col for col in df.columns if 'price' in col.lower() and col.lower() != target]
    cols_to_drop.append('kaina_men')
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    df = df.dropna(axis=1, how='all')
    
    # Split features by type
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    if target in numeric_features:
        numeric_features.remove(target)
    
    return df, numeric_features, categorical_features
