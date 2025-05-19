import sys
import os
import json
import time
import tempfile
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from azure.storage.blob import BlobServiceClient, ContainerClient
from datetime import datetime
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Azure Blob Storage connection
# Try to get from environment first (for Azure App Service)
BLOB_CONNECTION_STRING = os.environ.get('BLOB_CONNECTION_STRING')
BLOB_CONTAINER_NAME = os.environ.get('BLOB_CONTAINER_NAME')

# Fall back to .env file if not found (for local development)
if not BLOB_CONNECTION_STRING or not BLOB_CONTAINER_NAME:
    from dotenv import load_dotenv
    load_dotenv()
    BLOB_CONNECTION_STRING = os.environ.get('BLOB_CONNECTION_STRING')
    BLOB_CONTAINER_NAME = os.environ.get('BLOB_CONTAINER_NAME')

app = FastAPI(
    title="Real Estate Price Prediction API",
    description="API for predicting real estate prices using machine learning models from Azure Blob Storage",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model cache
model_cache = {
    "rental": {
        "latest": {"model": None, "last_updated": None, "required_columns": [], "metrics": {}},
        "specific": {}  # Will hold specific models by name
    },
    "selling": {
        "latest": {"model": None, "last_updated": None, "required_columns": [], "metrics": {}},
        "specific": {}  # Will hold specific models by name
    }
}

class PropertyFeatures(BaseModel):
    """Base input features for property prediction"""
    city: str = Field(..., description="City name")
    plotas: float = Field(..., description="Area in square meters")
    kambariu_sk: int = Field(..., description="Number of rooms")
    aukstas: int = Field(..., description="Floor number")
    aukstu_sk: int = Field(..., description="Total number of floors")
    metai: Optional[int] = Field(None, description="Year built")
    pastato_tipas: Optional[str] = Field(None, description="Building type")
    sildymas: Optional[str] = Field(None, description="Heating type")
    irengimas: Optional[str] = Field(None, description="Interior finishing")
    pastato_energijos_suvartojimo_klase: Optional[str] = Field(None, description="Energy efficiency class")
    latitude: Optional[float] = Field(None, description="Property latitude")
    longitude: Optional[float] = Field(None, description="Property longitude")
    distance_to_darzeliai: Optional[float] = Field(None, description="Distance to kindergartens in km")
    distance_to_mokyklos: Optional[float] = Field(None, description="Distance to schools in km")
    distance_to_stoteles: Optional[float] = Field(None, description="Distance to public transport in km")
    distance_to_parduotuves: Optional[float] = Field(None, description="Distance to stores in km")
    
    # Binary features with default values (0 means feature not present)
    atskiras_iejimas: Optional[int] = Field(0, description="Separate entrance")
    aukstos_lubos: Optional[int] = Field(0, description="High ceilings")
    butas_palepeje: Optional[int] = Field(0, description="Attic apartment")
    butas_per_kelis_aukstus: Optional[int] = Field(0, description="Multi-level apartment")
    internetas: Optional[int] = Field(0, description="Internet")
    kabeline_televizija: Optional[int] = Field(0, description="Cable TV")
    nauja_elektros_instaliacija: Optional[int] = Field(0, description="New electrical installation")
    nauja_kanalizacija: Optional[int] = Field(0, description="New sewage system")
    renovuotas_namas: Optional[int] = Field(0, description="Renovated building")
    tualetas_ir_vonia_atskirai: Optional[int] = Field(0, description="Separate toilet and bathroom")
    uzdaras_kiemas: Optional[int] = Field(0, description="Enclosed yard")
    virtuve_sujungta_su_kambariu: Optional[int] = Field(0, description="Kitchen connected with room")
    yra_liftas: Optional[int] = Field(0, description="Has elevator")
    balkonas: Optional[int] = Field(0, description="Has balcony")
    drabuzine: Optional[int] = Field(0, description="Has wardrobe")
    pirtis: Optional[int] = Field(0, description="Has sauna")
    sandeliukas: Optional[int] = Field(0, description="Has storage room")
    terasa: Optional[int] = Field(0, description="Has terrace")
    vieta_automobiliui: Optional[int] = Field(0, description="Has parking space")
    yra_palepe: Optional[int] = Field(0, description="Has attic")
    duso_kabina: Optional[int] = Field(0, description="Has shower cabin")
    indaplove: Optional[int] = Field(0, description="Has dishwasher")
    kondicionierius: Optional[int] = Field(0, description="Has air conditioning")
    plastikiniai_vamzdziai: Optional[int] = Field(0, description="Has plastic pipes")
    rekuperacine_sistema: Optional[int] = Field(0, description="Has recovery system")
    skalbimo_masina: Optional[int] = Field(0, description="Has washing machine")
    su_baldais: Optional[int] = Field(0, description="Furnished")
    virtuves_komplektas: Optional[int] = Field(0, description="Has kitchen set")
    virykle: Optional[int] = Field(0, description="Has stove")
    vonia: Optional[int] = Field(0, description="Has bathtub")
    budintis_sargas: Optional[int] = Field(0, description="Has security guard")
    kodine_laiptines_spyna: Optional[int] = Field(0, description="Has code lock")
    signalizacija: Optional[int] = Field(0, description="Has alarm")
    vaizdo_kameros: Optional[int] = Field(0, description="Has video cameras")

    # Rental-specific fields
    galima_deklaruoti_gyvenam: Optional[int] = Field(0, description="Can declare residence")
    galima_su_gyv: Optional[int] = Field(0, description="Can have pets")

class PredictionResponse(BaseModel):
    """Response model for price predictions"""
    predicted_price: float
    price_per_sqm: float
    model_info: Dict[str, Any]

def get_blob_client():
    """Initialize and return Azure Blob Storage container client or raise HTTPException."""
    try:
        blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)
        return container_client
    except Exception as e:
        logger.error(f"Failed to connect to Azure Blob Storage: {e}")
        raise HTTPException(status_code=500, detail="Could not connect to model storage")

def is_cache_valid(model_type, model_name=None):
    """Check if cached model is less than 1 hour old and still valid."""
    if model_name:
        # Check specific model cache
        if model_name not in model_cache[model_type]["specific"]:
            return False
            
        cache_entry = model_cache[model_type]["specific"][model_name]
    else:
        # Check latest model cache
        cache_entry = model_cache[model_type]["latest"]
    
    if (cache_entry.get("model") is None or cache_entry.get("last_updated") is None):
        return False
    
    # Check if model is less than 1 hour old
    time_diff = (datetime.now() - cache_entry["last_updated"]).total_seconds()
    return time_diff < 3600

def normalize(series):
    """Scale numeric values to 0-1 range using min-max normalization."""
    series = pd.Series(series)
    min_val = series.min()
    max_val = series.max()
    return [(x - min_val) / (max_val - min_val) if pd.notnull(x) and max_val != min_val else 0.5 
            for x in series]

def get_city_center(city_name, country='Lithuania'):
    """Get latitude and longitude for city center using geocoding, with country-specific search."""
    geolocator = Nominatim(user_agent="real_estate_feature_engineering")
    
    try:
        # Try to geocode the city name with country for better precision
        location = geolocator.geocode(f"{city_name}, {country}", exactly_one=True)
        
        # If unsuccessful, try just the city name
        if location is None:
            location = geolocator.geocode(city_name, exactly_one=True)
        
        if location:
            return (location.latitude, location.longitude)
        else:
            print(f"Could not find coordinates for {city_name}, {country}")
            return None
        
    except (GeocoderTimedOut, GeocoderUnavailable) as e:
        print(f"Geocoding error for {city_name}: {e}")
        return None

def download_specific_model(model_type, model_name):
    """Download and cache a specific model from blob storage with required features and metrics."""
    import pickle
    # Check if model is in cache and valid
    if is_cache_valid(model_type, model_name):
        logger.info(f"Using cached specific model: {model_name}")
        return model_cache[model_type]["specific"][model_name]["model"]
    
    # Get a temporary file path
    temp_file_fd, temp_file_path = tempfile.mkstemp(suffix='.pkl')
    os.close(temp_file_fd)  # Close file descriptor
    
    try:
        # Get blob client
        blob_client = get_blob_client()
        
        # Log which model we're trying to download
        blob_path = f"{model_type}/{model_name}"
        logger.info(f"Attempting to download specific model: {blob_path}")
        
        # Check if the blob exists
        try:
            # Download the model
            with open(temp_file_path, 'wb') as file:
                blob_data = blob_client.download_blob(blob_path).readall()
                file.write(blob_data)
        except Exception as e:
            logger.error(f"Failed to download model {model_name}: {str(e)}")
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        # Load the model with custom unpickler to handle random state
        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "numpy.random._mt19937" and name == "MT19937":
                    return np.random.MT19937
                return super().find_class(module, name)
                
        with open(temp_file_path, 'rb') as f:
            model = CustomUnpickler(f).load()
        
        # Extract model metrics
        metrics = {}
        if 'metrics' in model:
            metrics = model['metrics']
        elif 'eval_metrics' in model:
            metrics = model['eval_metrics']
        else:
            logger.warning(f"No metrics found in model {model_name}")
        
        # Extract required columns from the model
        required_columns = []
        try:
            # Handle ensemble models
            if isinstance(model, dict) and 'base_models' in model:
                base_model = list(model['base_models'].values())[0]  # Get any base model
                if hasattr(base_model, 'named_steps') and 'preprocessor' in base_model.named_steps:
                    preprocessor = base_model.named_steps['preprocessor']
                    if hasattr(preprocessor, 'transformers_'):
                        # Get required columns from the column transformer
                        for _, _, cols in preprocessor.transformers_:
                            if cols:
                                required_columns.extend([col for col in cols if not isinstance(col, (int, float))])
            # Handle individual models with preprocessor
            elif hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
                preprocessor = model.named_steps['preprocessor']
                if hasattr(preprocessor, 'transformers_'):
                    for _, _, cols in preprocessor.transformers_:
                        if cols:
                            required_columns.extend([col for col in cols if not isinstance(col, (int, float))])
        except Exception as e:
            logger.warning(f"Could not extract required columns from model: {e}")
        
        # Cache the specific model
        model_cache[model_type]["specific"][model_name] = {
            "model": model,
            "last_updated": datetime.now(),
            "required_columns": required_columns,
            "metrics": metrics
        }
        
        return model
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading specific model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load model {model_name}: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_file_path}: {e}")

def download_latest_model(model_type):
    """Download and cache most recent model from blob storage based on modified date."""
    import pickle
    # Use cached model if valid
    if is_cache_valid(model_type):
        logger.info(f"Using cached {model_type} model")
        return model_cache[model_type]["latest"]["model"]
    
    # Get a temporary file path
    temp_file_fd, temp_file_path = tempfile.mkstemp(suffix='.pkl')
    os.close(temp_file_fd)  # Close file descriptor
    
    try:
        # Get blob client
        blob_client = get_blob_client()
        
        # List all blobs in the container with the model type prefix
        blob_list = list(blob_client.list_blobs(name_starts_with=f"{model_type}/"))
        
        if not blob_list:
            raise HTTPException(status_code=404, detail=f"No {model_type} models found in storage")
        
        # Find the ensemble model
        ensemble_blobs = [blob for blob in blob_list if blob.name.endswith('ensemble.pkl')]
        if not ensemble_blobs:
            ensemble_blobs = [blob for blob in blob_list if blob.name.endswith('.pkl')]  # Fallback to any model
            
        if not ensemble_blobs:
            raise HTTPException(status_code=404, detail=f"No ensemble model found for {model_type}")
        
        # Get the latest model by modified time
        latest_model = max(ensemble_blobs, key=lambda x: x.last_modified)
        logger.info(f"Found latest {model_type} model: {latest_model.name}, last modified: {latest_model.last_modified}")
        
        # Download the model using the direct blob name from Azure (already includes full path)
        logger.info(f"Downloading latest model: {latest_model.name}")
        with open(temp_file_path, 'wb') as file:
            blob_data = blob_client.download_blob(latest_model.name).readall()
            file.write(blob_data)
        
        # Load the model with custom unpickler to handle random state
        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "numpy.random._mt19937" and name == "MT19937":
                    return np.random.MT19937
                return super().find_class(module, name)
                
        with open(temp_file_path, 'rb') as f:
            model = CustomUnpickler(f).load()
        
        # Extract model metrics
        metrics = {}
        if 'metrics' in model:
            metrics = model['metrics']
        elif 'eval_metrics' in model:
            metrics = model['eval_metrics']
        else:
            logger.warning(f"No metrics found in the {model_type} model")
        
        # Extract required columns from the model
        required_columns = []
        try:
            base_model = list(model['base_models'].values())[0]  # Get any base model
            if hasattr(base_model, 'named_steps') and 'preprocessor' in base_model.named_steps:
                preprocessor = base_model.named_steps['preprocessor']
                if hasattr(preprocessor, 'transformers_'):
                    # Get required columns from the column transformer
                    for _, _, cols in preprocessor.transformers_:
                        if cols:
                            required_columns.extend([col for col in cols if not isinstance(col, (int, float))])
        except Exception as e:
            logger.warning(f"Could not extract required columns from model: {e}")
        
        # Update cache
        model_cache[model_type]["latest"] = {
            "model": model,
            "last_updated": datetime.now(),
            "required_columns": required_columns,
            "metrics": metrics
        }
        
        return model
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load {model_type} model: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_file_path}: {e}")
                
# Feature Engineering Functions
def engineer_features(df, cache_file='city_coordinates_cache.json', predict = False):
    """
    Apply feature engineering to real estate DataFrame
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input DataFrame containing real estate data
    cache_file : str, default='city_coordinates_cache.json'
        Path to the file where city coordinates are cached
    
    Returns:
    --------
    df : pandas DataFrame
        DataFrame with added engineered features
    """
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # ===== PRICE AND BASIC METRICS =====
    
    if predict == False:
        # Calculate price per square meter
        df['price_per_sqm'] = [price / area if area > 0 else None 
                              for price, area in zip(df['price'], df['plotas'])]
    
    # Room size average
    df['avg_room_size'] = df['plotas'] / df['kambariu_sk'].replace(0, 1)
                          
    # Calculate building age
    current_year = datetime.today().year
    df['building_age'] = [
        current_year - year if year != 'Not Specified' and isinstance(year, (int, float))
        else None for year in df['metai']
    ]
    
    # ===== FLOOR-RELATED FEATURES =====
    
    # Relative floor position
    df['relative_floor_position'] = df['aukstas'] / df['aukstu_sk']
    
    # Binary feature for ground floor
    df['is_ground_floor'] = (df['aukstas'] == 1).astype(int)
    
    # Binary feature for top floor
    df['is_top_floor'] = (df['aukstas'] == df['aukstu_sk']).astype(int)
    
    # Binary feature for middle floors
    df['is_middle_floor'] = ((df['aukstas'] != 1) & 
                             (df['aukstas'] != df['aukstu_sk'])).astype(int)
    
    # ===== AGE-RELATED FEATURES =====
    
    # Create age bins
    age_bins = [0, 5, 10, 20, 30, 50, 100, float('inf')]
    age_labels = ['<5 years', '5-10 years', '10-20 years', '20-30 years', 
                  '30-50 years', '50-100 years', '100+ years']
    
    df['building_age_category'] = pd.cut(df['building_age'], bins=age_bins, 
                                          labels=age_labels, right=False)
    
    # ===== AMENITY SCORES =====
    
    # Calculate amenity score
    amenity_features = [
        'balkonas', 'sandeliukas', 'terasa', 'vieta_automobiliui', 
        'duso_kabina', 'indaplove', 'kondicionierius', 'virtuves_komplektas',
        'skalbimo_masina', 'vonia'
    ]
    
    df['amenity_score'] = np.nan
    for i in range(len(df)):
        count = sum(1 for feature in amenity_features 
                   if feature in df and df.loc[i, feature] == 1.0)
        total = sum(1 for feature in amenity_features 
                   if feature in df and df.loc[i, feature] is not None)
        
        score = (count / total * 10) if total > 0 else None
        df.at[i, 'amenity_score'] = round(score, 1) if score is not None else None
    
    # Calculate renovation index
    renovation_features = [
        'nauja_elektros_instaliacija', 'nauja_kanalizacija', 
        'renovuotas_namas', 'plastikiniai_vamzdziai'
    ]
    
    df['renovation_index'] = np.nan
    for i in range(len(df)):
        count = sum(1 for feature in renovation_features 
                   if feature in df and df.loc[i, feature] == 1.0)
        total = sum(1 for feature in renovation_features 
                   if feature in df and df.loc[i, feature] is not None)
        
        score = (count / total * 10) if total > 0 else None
        df.at[i, 'renovation_index'] = round(score, 1) if score is not None else None
    
    # ===== LOCATION-BASED FEATURES =====
    
    # Load and manage city coordinates cache
    city_center_cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                loaded_cache = json.load(f)
                # Convert string tuples back to actual tuples
                for city, coords in loaded_cache.items():
                    if coords is not None:
                        city_center_cache[city] = tuple(coords)
                    else:
                        city_center_cache[city] = None
            print(f"Loaded {len(city_center_cache)} city coordinates from cache.")
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading cache: {e}")
    
    # Get city center coordinates for each unique city
    unique_cities = df['city'].dropna().unique()
    cache_updated = False
    
    for city in unique_cities:
        # Use cache if available, otherwise fetch coordinates
        if city not in city_center_cache:
            center_coords = get_city_center(city)
            if center_coords:
                city_center_cache[city] = center_coords
                cache_updated = True
                # Add a delay to avoid hitting API rate limits
                time.sleep(1)
            else:
                # Use a placeholder if geocoding fails
                print(f"Using default coordinates for {city}")
                city_center_cache[city] = None
    
    # Calculate distance to city center
    def calculate_distance_to_center(row):
        if pd.isnull(row['city']) or pd.isnull(row['latitude']) or pd.isnull(row['longitude']):
            return np.nan
        
        city_center = city_center_cache.get(row['city'])
        if city_center is None:
            return np.nan
        
        # Calculate distance using geodesic (more accurate than Haversine)
        property_coords = (row['latitude'], row['longitude'])
        distance_km = geodesic(property_coords, city_center).kilometers
        
        return distance_km
    
    # Apply the function to each row to calculate distance to city center
    df['distance_to_city_center'] = df.apply(calculate_distance_to_center, axis=1)
    
    # ===== CONVENIENCE SCORE =====
    
    # Calculate convenience score based on distances to amenities
    distance_columns = ['distance_to_darzeliai', 'distance_to_mokyklos', 
                        'distance_to_stoteles', 'distance_to_parduotuves']
    
    if all(col in df.columns for col in distance_columns):
        df['convenience_score'] = -1 * (
            df['distance_to_darzeliai'] + 
            df['distance_to_mokyklos'] + 
            df['distance_to_stoteles'] + 
            df['distance_to_parduotuves']
        )
    
    
    # ===== VALUE SCORE =====
    
    # Calculate value score if necessary columns exist
    if all(col in df.columns for col in ['amenity_score', 'convenience_score']):
        normalized_amenity = normalize(df['amenity_score'])
        normalized_accessibility = normalize(df['convenience_score'])
        
        df['value_score'] = [(a + b ) / 3 * 10 
                            for a, b in zip(normalized_amenity, normalized_accessibility)]
        df['value_score'] = df['value_score'].round(1)
    
    return df

def prepare_prediction_data(property_data, model_type, model_name=None):
    """Transform property features into model-ready DataFrame with engineered features."""
    # Convert to DataFrame
    df = pd.DataFrame([property_data.dict()])
    
    # Apply feature engineering
    df = engineer_features(df, predict=True)
    
    # Get required columns from the appropriate cache entry
    if model_name and model_name in model_cache[model_type]["specific"]:
        required_columns = model_cache[model_type]["specific"][model_name].get("required_columns", [])
    else:
        required_columns = model_cache[model_type]["latest"].get("required_columns", [])
    
    # Add missing columns required by the model
    for col in required_columns:
        if col not in df.columns:
            df[col] = None
            
    return df

def generate_prediction(model, df, property_area, model_type):
    """Make price prediction using model, supporting both ensemble and individual models."""
    try:
        # Check if this is an ensemble model or individual model
        is_ensemble = isinstance(model, dict) and 'base_models' in model and 'meta_model' in model
        
        if is_ensemble:
            # Handle ensemble model (existing logic)
            base_models = model['base_models']
            meta_model = model['meta_model']
            
            # Get predictions from base models
            meta_features = np.column_stack([
                base_model.predict(df) for base_model in base_models.values()
            ])
            
            # Get final prediction from meta-model
            prediction_log = meta_model.predict(meta_features)[0]
            
            # Transform back from log if needed
            if model.get('target_info', {}).get('log_transform', False):
                prediction = np.expm1(prediction_log)
            else:
                prediction = prediction_log
                
            # For ensemble models, get info about base models
            model_info = {
                "base_models": list(base_models.keys()),
                "features_used": len(df.columns),
                "model_type": "ensemble"
            }
        else:
            # Handle individual model (direct prediction)
            prediction = model.predict(df)[0]
            
            prediction = np.expm1(prediction)

            # For individual models, get info about the model type
            model_info = {
                "features_used": len(df.columns),
                "model_type": "individual"
            }
        
        # Calculate price per square meter
        price_per_sqm = prediction / property_area if property_area > 0 else 0
        
        # Get model metrics
        metrics = model_cache[model_type].get("metrics", {})
        
        # Add metrics to the model info
        model_info["accuracy_metrics"] = {
            "r2_score": metrics.get("r2", metrics.get("r2_score", None)),
            "rmse": metrics.get("rmse", None),
            "mae": metrics.get("mae", None),
            "mape": metrics.get("mape", None)
        }
        
        return {
            "predicted_price": round(prediction, 2),
            "price_per_sqm": round(price_per_sqm, 2),
            "model_info": model_info
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/", tags=["Health"])
async def root():
    """API health check endpoint"""
    return {"status": "online", "message": "Real Estate Price Prediction API is running"}

@app.get("/models", tags=["Models"])
async def list_available_models():
    """List available models in Azure Blob Storage"""
    try:
        # Get blob client
        blob_client = get_blob_client()
        
        # List all model blob files
        rental_blobs = list(blob_client.list_blobs(name_starts_with="rental/"))
        selling_blobs = list(blob_client.list_blobs(name_starts_with="selling/"))
        
        rental_models = [{"name": blob.name, "last_modified": blob.last_modified} 
                         for blob in rental_blobs if blob.name.endswith('.pkl')]
        selling_models = [{"name": blob.name, "last_modified": blob.last_modified} 
                          for blob in selling_blobs if blob.name.endswith('.pkl')]
        
        return {
            "rental_models": sorted(rental_models, key=lambda x: x["last_modified"], reverse=True),
            "selling_models": sorted(selling_models, key=lambda x: x["last_modified"], reverse=True)
        }
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@app.post("/predict/{model_type}", response_model=PredictionResponse, tags=["Predictions"])
async def predict_price(
    model_type: str, 
    property_data: PropertyFeatures,
    model_name: Optional[str] = None
):
    # Validate model type
    if model_type not in ["rental", "selling"]:
        raise HTTPException(status_code=400, detail="Model type must be 'rental' or 'selling'")
    
    # Get specific model if name provided, otherwise use latest
    if model_name:
        model = download_specific_model(model_type, model_name)
    else:
        model = download_latest_model(model_type)
    
    # Prepare features
    df = prepare_prediction_data(property_data, model_type)
    
    # Generate prediction
    result = generate_prediction(model, df, property_data.plotas, model_type)
    result["model_info"]["type"] = model_type
    
    return PredictionResponse(**result)

@app.get("/refresh-model/{model_type}", tags=["Models"])
async def refresh_model_cache(model_type: str):
    """Force a refresh of the model cache"""
    if model_type not in ["rental", "selling"]:
        raise HTTPException(status_code=400, detail="Model type must be 'rental' or 'selling'")
    
    # Reset the cache entry
    model_cache[model_type] = {"model": None, "last_updated": None, "required_columns": [], "metrics": {}}
    
    # Download latest model
    model = download_latest_model(model_type)
    
    return {
        "status": "success",
        "message": f"{model_type.capitalize()} model cache refreshed successfully",
        "model_info": {
            "base_models": list(model['base_models'].keys()) if model else [],
            "accuracy_metrics": model_cache[model_type].get("metrics", {})
        }
    }

@app.get("/model-requirements/{model_type}", tags=["Models"])
async def get_model_requirements(model_type: str):
    """Get the required features for a specific model type"""
    if model_type not in ["rental", "selling"]:
        raise HTTPException(status_code=400, detail="Model type must be 'rental' or 'selling'")
    
    # Ensure model is loaded
    if not is_cache_valid(model_type):
        download_latest_model(model_type)
    
    return {
        "model_type": model_type,
        "required_columns": model_cache[model_type].get("required_columns", []),
        "accuracy_metrics": model_cache[model_type].get("metrics", {})
    }

# Simple direct prediction endpoints for backward compatibility
@app.post("/predict/rental", response_model=PredictionResponse, tags=["Predictions"])
async def predict_rental(property_data: PropertyFeatures):
    """Predict rental price for a property"""
    return await predict_price("rental", property_data)

@app.post("/predict/selling", response_model=PredictionResponse, tags=["Predictions"])
async def predict_selling(property_data: PropertyFeatures):
    """Predict selling price for a property"""
    return await predict_price("selling", property_data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
