import streamlit as st
import pandas as pd
import numpy as np

# Import components
from components.ui import (
    set_page_config, apply_custom_css, display_header, 
    display_about_section, display_how_to_use, display_footer,
    display_prediction_result
)
from components.api import (
    predict_price, get_available_models, get_model_requirements,
    add_to_history, format_price
)
from components.geocoding import (
    location_input_section, distance_inputs
)
from components.prediction import (
    basic_info_inputs, building_characteristics_inputs,
    property_features_inputs, prepare_property_data,
    create_prediction_button
)
from components.market_analysis import display_market_analysis

# Set page configuration and apply styling
set_page_config()
apply_custom_css()

# Display page header
display_header()

# Initialize session state for property data if not exists
if "property_data" not in st.session_state:
    st.session_state.property_data = {
        "city": "Vilnius",
        "plotas": 50.0,
        "kambariu_sk": 2,
        "aukstas": 3,
        "aukstu_sk": 5,
        "metai": 2000
    }

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # Model type selection
    model_type = st.radio(
        "Choose prediction type",
        ["selling", "rental"],
        index=0,  # Default to "Selling Price"
        format_func=lambda x: "Rental Price" if x == "rental" else "Selling Price",
        help="Select whether you want to predict rental or selling price"
    )
    
    # Display model info
    st.subheader("Model Information")
    models = get_available_models()
    model_requirements = get_model_requirements(model_type)
    
    # Model selector
    available_models = []
    selected_model = None
    
    if model_type == "rental" and models["rental_models"]:
        # Create a dictionary mapping display names to actual model names
        from components.ui import format_model_name
        model_options = {format_model_name(model["name"].split('/')[-1]): model["name"].split('/')[-1] 
                        for model in models["rental_models"]}
        
        # Get display names in a list
        display_names = list(model_options.keys())
        # Find the index of 'Ensemble' in display names (case-insensitive)
        default_model = next((i for i, name in enumerate(display_names) if 'ensemble' in name.lower()), 0)
        
        # Show the user-friendly names in the dropdown
        selected_display_name = st.selectbox(
            "Select rental model",
            display_names,
            index=default_model,
            help="Choose from available rental price prediction models"
        )
        
        # Get the actual model name
        selected_model = model_options[selected_display_name]
        
        # Get the selected model info for display
        selected_model_info = next((model for model in models["rental_models"] 
                                if model["name"].split('/')[-1] == selected_model), None)
        
        if selected_model_info:
            last_updated = pd.to_datetime(selected_model_info["last_modified"]).strftime("%Y-%m-%d %H:%M")
            st.success(f"Using: {selected_display_name}")
            st.info(f"Last updated: {last_updated}")
    elif model_type == "selling" and models["selling_models"]:
        # Create a dictionary mapping display names to actual model names
        from components.ui import format_model_name
        model_options = {format_model_name(model["name"].split('/')[-1]): model["name"].split('/')[-1] 
                        for model in models["selling_models"]}
        
        # Get display names in a list
        display_names = list(model_options.keys())
        # Find the index of 'Ensemble' in display names (case-insensitive)
        default_model = next((i for i, name in enumerate(display_names) if 'ensemble' in name.lower()), 0)
        
        # Show the user-friendly names in the dropdown
        selected_display_name = st.selectbox(
            "Select selling model",
            display_names,
            index=default_model,
            help="Choose from available selling price prediction models"
        )
        
        # Get the actual model name
        selected_model = model_options[selected_display_name]
        
        # Get the selected model info for display
        selected_model_info = next((model for model in models["selling_models"] 
                                if model["name"].split('/')[-1] == selected_model), None)
        
        if selected_model_info:
            last_updated = pd.to_datetime(selected_model_info["last_modified"]).strftime("%Y-%m-%d %H:%M")
            st.success(f"Using: {selected_display_name}")
            st.info(f"Last updated: {last_updated}")
    else:
        st.warning(f"No {model_type} models found")

# Main form tabs
tab1, tab2, tab3, tab4 = st.tabs(["📋 Basic Details", "🗺️ Location & Distances", "✅ Property Features", "📊 Market Analysis"])

with tab1:
    # Get basic property information
    basic_info = basic_info_inputs()
    
    # Get building characteristics
    building_chars = building_characteristics_inputs()

with tab2:
    st.subheader("Location")
    
    # Get location information
    location = location_input_section(basic_info["city"])
    
    # Get distance inputs
    distances = distance_inputs()

with tab3:
    # Get property features
    property_features = property_features_inputs(model_type)

with tab4:
    # Display market analysis
    display_market_analysis()

# Create prediction button
predict_button = create_prediction_button()

# Processing prediction
if predict_button:
    # Prepare complete property data
    property_data = prepare_property_data(
        basic_info, 
        building_chars, 
        property_features, 
        location, 
        distances
    )
    
    # Update session state
    st.session_state.property_data = property_data
    
    # Call predict_price API
    with st.spinner(f"Predicting {model_type} price..."):
        # Clear st.cache_data for predict_price to ensure fresh results
        st.cache_data.clear()  # Updated cache clearing method
    
        if selected_model:
            # Get just the filename from the full path
            result = predict_price(property_data, model_type, selected_model)
        else:
            # No model selected, use default
            result = predict_price(property_data, model_type)
    
    # Show results
    if result:
        # Add to history
        add_to_history(property_data, result, model_type)
        
        # Display prediction results
        display_prediction_result(result, property_data, model_type, selected_model)

# Display information sections
display_about_section()
display_how_to_use()
display_footer()
