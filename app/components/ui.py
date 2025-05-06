import streamlit as st

def set_page_config():
    """Configure the Streamlit page with title, icon, and layout settings"""
    st.set_page_config(
        page_title="Aruodas Real Estate Price Predictor",
        page_icon="🏠",
        layout="wide"
    )

def apply_custom_css():
    """Apply custom CSS styling to the Streamlit app"""
    st.markdown("""
    <style>
        .main .block-container {
            padding-top: 2rem;
        }
        .prediction-box {
            background-color: #f8f9fa;
            border-radius: 0.5rem;
            padding: 1.5rem;
            box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
        }
        .stTabs [data-baseweb="tab"] {
            height: 3rem;
            white-space: pre-wrap;
            border-radius: 0.25rem 0.25rem 0 0;
        }
        .stTabs [aria-selected="true"] {
            background-color: #f0f2f6;
            font-weight: bold;
        }
        .feature-group {
            border: 1px solid #eee;
            border-radius: 0.5rem;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .feature-header {
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)

def display_header():
    """Display the app header and description"""
    st.title("🏠 Real Estate Price Prediction")
    st.markdown("Enter property details to predict rental or selling price in Lithuania based on advanced machine learning models")

def format_model_name(raw_name):
    """Convert internal model names to user-friendly display names with proper capitalization."""
    if "ensemble" in raw_name.lower():
        return "Ensemble Model"
    elif "random_forest" in raw_name.lower():
        return "Random Forest Model"
    elif "xgboost" in raw_name.lower():
        return "XGBoost Model"
    elif "lightgbm" in raw_name.lower():
        return "LightGBM Model"
    elif "gradient_boosting" in raw_name.lower():
        return "Gradient Boosting Model"
    else:
        # For any other models, try to beautify the name
        return " ".join([word.capitalize() for word in raw_name.replace('.pkl', '').split('_')])

def display_about_section():
    """Display the About the Model section"""
    st.markdown("---")
    with st.expander("📊 About the Model"):
        st.markdown("""
        ### Advanced Machine Learning Pipeline for Real Estate Price Prediction
        
        This application uses an ensemble of machine learning models to predict real estate prices in Lithuania. The model has been trained on data from Aruodas.lt and includes the following components:
        
        - **Ensemble Model Architecture**: Combines LightGBM, XGBoost, Random Forest, and Gradient Boosting with a meta-model
        - **Advanced Feature Engineering**: Processing of numerical and categorical features with domain-specific transformations
        - **High Accuracy**: Fine-tuned model parameters optimize for low error rates and high predictive power
        
        The model takes into account location, property features, and amenities to generate accurate price predictions for both rental and selling scenarios.
        """)

def display_how_to_use():
    """Display the How to Use section"""
    with st.expander("⚙️ How to Use This App"):
        st.markdown("""
        ### Instructions
        
        1. **Choose Model Type**: Select whether you want to predict rental or selling price in the sidebar
        2. **Enter Property Details**: Fill in the basic information in the "Basic Details" tab
        3. **Add Location**: Optionally set the exact location and distances in the "Location & Distances" tab
        4. **Select Features**: Check all features that apply to your property in the "Property Features" tab
        5. **Get Prediction**: Click the "PREDICT PRICE" button to generate a price estimate
        
        You can view your prediction history in the sidebar. The most recent predictions are stored for quick reference.
        """)

def display_footer():
    """Display the app footer"""
    st.markdown("---")
    st.markdown(
        """<div style='text-align: center; color: #888;'>
        Powered by machine learning models trained on real estate data from Aruodas.lt<br>
        © 2025 Real Estate Price Predictor
        © Trained by Tomas Dzedulionis
        </div>""", 
        unsafe_allow_html=True
    )

def display_prediction_result(result, property_data, model_type, selected_model):
    """Display the prediction results in a formatted box"""
    st.markdown("---")
    st.subheader("🎯 Prediction Results")
    
    with st.container():
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        
        # Main price
        if model_type == "rental":
            price_text = f"€{result['predicted_price']:,.2f} / month"
            price_per_sqm_text = f"€{result['price_per_sqm']:,.2f} / m² / month"
            price_label = "Predicted Monthly Rent"
        else:
            price_text = f"€{result['predicted_price']:,.2f}"
            price_per_sqm_text = f"€{result['price_per_sqm']:,.2f} / m²"
            price_label = "Predicted Property Value"
        
        st.markdown(f"<h2 style='text-align: center; margin-bottom: 0.5rem;'>{price_label}</h2>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center; color: #1e88e5;'>{price_text}</h1>", unsafe_allow_html=True)
        st.metric("Price per m²", price_per_sqm_text)
        
        # Property summary
        st.markdown("### Property Summary")
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        with summary_col1:
            st.markdown(f"**City:** {property_data['city']}")
            st.markdown(f"**Area:** {property_data['plotas']} m²")
        with summary_col2:
            st.markdown(f"**Rooms:** {property_data['kambariu_sk']}")
            st.markdown(f"**Floor:** {property_data['aukstas']}/{property_data['aukstu_sk']}")
        with summary_col3:
            st.markdown(f"**Year built:** {property_data['metai']}")
            heating = property_data['sildymas'] if property_data['sildymas'] else "Not specified"
            st.markdown(f"**Heating:** {heating}")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Display model information
    st.markdown("### Model Details")
    with st.container():
        st.markdown('<div class="prediction-box" style="margin-top: 20px;">', unsafe_allow_html=True)
        
        # Model type and name
        st.markdown(f"**Model type:** {model_type.capitalize()}")
        
        # Display model name and info
        if "ensemble" in selected_model.lower():
            st.markdown("**Model used:** Ensemble Model")
            st.markdown("**Base algorithms:** LightGBM, XGBoost, Random Forest, Gradient Boosting")
        elif "random_forest" in selected_model.lower():
            st.markdown("**Model used:** Random Forest")
            st.markdown("**Algorithm type:** Tree-based ensemble using bootstrap aggregating")
        elif "xgboost" in selected_model.lower():
            st.markdown("**Model used:** XGBoost")
            st.markdown("**Algorithm type:** Gradient boosted decision trees with regularization")
        elif "lightgbm" in selected_model.lower():
            st.markdown("**Model used:** LightGBM")
            st.markdown("**Algorithm type:** Gradient boosting framework using tree-based learning")
        elif "gradient_boosting" in selected_model.lower():
            st.markdown("**Model used:** Gradient Boosting")
            st.markdown("**Algorithm type:** Sequential ensemble of weak learners")
        else:
            st.markdown(f"**Model used:** {format_model_name(selected_model)}")
        
        # Additional model info
        if "model_info" in result and "features_used" in result["model_info"]:
            st.markdown(f"**Number of features used:** {result['model_info']['features_used']}")
        if "model_info" in result and "performance" in result["model_info"]:
            st.markdown(f"**Model accuracy:** {result['model_info']['performance']}%")
        
        import pandas as pd
        st.markdown(f"**Prediction timestamp:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.markdown('</div>', unsafe_allow_html=True)