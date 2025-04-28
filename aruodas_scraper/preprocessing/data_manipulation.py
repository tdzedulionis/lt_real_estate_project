"""Module for data cleaning, preprocessing and feature engineering of real estate data."""

import os
import re
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
from unidecode import unidecode
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

# Utility Functions
def normalize(series):
    """Normalize numeric values to a 0-1 range using min-max scaling."""
    series = pd.Series(series)
    min_val = series.min()
    max_val = series.max()
    return [(x - min_val) / (max_val - min_val) if pd.notnull(x) and max_val != min_val else 0.5 
            for x in series]

def convert_numeric_if_possible(df, column_name):
    """Convert DataFrame column to numeric type if less than 70% values would be NaN."""
    try:
        converted_column = pd.to_numeric(df[column_name], errors='coerce')
        nan_percentage = converted_column.isna().sum() / len(converted_column)
        if nan_percentage < 0.7:  # Less than 70% NaNs
            df[column_name] = converted_column
            return df
        else:
            print(f"Column '{column_name}' could not be meaningfully converted to numeric (too many NaNs).")
            return df

    except (ValueError, TypeError) as e:
        print(f"Column '{column_name}' could not be converted to numeric: {e}")
        return df
    except Exception as e:
        print(f"An unexpected error occurred while converting '{column_name}': {e}")
        return df

def create_features_columns(df, columns_to_process):
    """Create binary columns for each unique feature found in text columns."""
    def extract_unique_features(df, column):
        unique_features = set()
        for item in df[f'{column}']:
            if isinstance(item, str) and item != 'nan':
                features = re.findall(r'[A-Z][a-zėųčšžį]+(?:\s+[a-zėųčšžį/]+)*', item)
                for feature in features:
                    unique_features.add(feature)
        return list(sorted(unique_features))

    for column in columns_to_process:
        unique_features = extract_unique_features(df, column)
        for feature in unique_features:
            df[feature] = df[column].apply(
                lambda x: 1 if isinstance(x, str) and feature in x else np.nan if pd.isna(x) else 0
            )
    return df

def get_city_center(city_name, country='Lithuania'):
    """Get latitude and longitude coordinates for a city center using geocoding."""
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

# Data Cleaning Functions
def clean_data(df):
    """Clean real estate data by standardizing values, removing outliers, and dropping unnecessary columns."""
    # Define columns to process
    columns_to_process = ['ypatybes', 'papildomos_patalpos', 'papildoma_iranga', 'apsauga']
    
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Create binary feature columns from text columns
    df = create_features_columns(df, columns_to_process)
    
    # Standardize column names
    new_columns = []
    for col in df.columns:
        new_col = col.lower().replace(" ", "_")
        new_col = unidecode(new_col)
        new_columns.append(new_col)
    df.columns = new_columns
    
    # Remove price outliers
    prices = df['price'].copy()
    lower_bound = np.percentile(prices, 1)
    upper_bound = np.percentile(prices, 99.9)
    print(f"Removing prices below {lower_bound} and above {upper_bound}")
    print(f"This will remove {sum((prices < lower_bound) | (prices > upper_bound))} out of {len(prices)} records")
    df = df[(prices >= lower_bound) & (prices <= upper_bound)]
    df = df.reset_index(drop=True)
    
    # Clean energy efficiency values
    df['pastato_energijos_suvartojimo_klase'] = df['pastato_energijos_suvartojimo_klase'].apply(
        lambda x: x if x in ['A', 'A+', 'A++', 'B', 'Not Specified'] else 'Lower than B'
    )
    
    # Fill missing values with 'Not Specified'
    df = df.fillna('Not Specified')
    
    # Clean up invalid values
    df = df.replace({"": None, "nan": None, "NaN": None, "NULL": None, "null": None})
    
    # Drop unnecessary columns
    columns_to_drop = [
        'url', 'namo_numeris', 'langu_orientacija', 'buto_numeris', 
        'scrape_date', 'unikalus_daikto_numeris_(rc_numeris)', 'varzytynes'
    ] + columns_to_process
    
    df = df.drop([col for col in columns_to_drop if col in df.columns], axis=1)
    
    return df

# Feature Engineering Functions
def engineer_features(df, cache_file='city_coordinates_cache.json', predict=False):
    """Generate advanced features from raw real estate data including price metrics, location scores, and amenity indices."""
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
