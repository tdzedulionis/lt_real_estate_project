import os
import streamlit as st
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_URL = os.getenv('API_URL', 'https://real-estate-fast-api-dxafaefkg3hefpfs.polandcentral-01.azurewebsites.net/')
PREDICTION_TIMEOUT = int(os.getenv('PREDICTION_TIMEOUT', '60'))
MODEL_INFO_TIMEOUT = int(os.getenv('MODEL_INFO_TIMEOUT', '30'))
CACHE_TTL_PREDICTIONS = int(os.getenv('CACHE_TTL_PREDICTIONS', '60'))
CACHE_TTL_MODELS = int(os.getenv('CACHE_TTL_MODELS', '600'))
MAX_RETRIES = int(os.getenv('MAX_RETRIES', '5'))
MIN_RETRY_WAIT = int(os.getenv('MIN_RETRY_WAIT', '4'))
MAX_RETRY_WAIT = int(os.getenv('MAX_RETRY_WAIT', '30'))

@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=2, min=MIN_RETRY_WAIT, max=MAX_RETRY_WAIT)
)
@st.cache_data(ttl=CACHE_TTL_PREDICTIONS)
def predict_price(property_data, model_type, model_name=None):
    """Request price prediction from API with retry logic and error handling."""
    try:
        endpoint = f"{API_URL}/predict/{model_type}"
        
        # Include the model name in the parameters if provided
        if model_name:
            endpoint += f"?model_name={model_name}"
            
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
            
        response = requests.post(
            endpoint,
            json=property_data,
            headers=headers,
            timeout=PREDICTION_TIMEOUT
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            error_detail = f"Status Code: {response.status_code}"
            try:
                error_json = response.json()
                if "detail" in error_json:
                    error_detail += f" - {error_json['detail']}"
            except:
                error_detail += f" - {response.text}"
            
            if response.status_code == 502:
                st.error("""
                API Error: Bad Gateway (502). This may occur with ensemble models due to their complexity.
                Possible solutions:
                - Try again in a few moments
                - Use a simpler model (e.g., LightGBM or XGBoost instead of ensemble)
                - Reduce the number of features if possible
                """)
            else:
                st.error(f"API Error: {error_detail}")
            return None
    except Exception as e:
        st.error(f"Error calling prediction API: {str(e)}")
        return None

@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=MIN_RETRY_WAIT, max=MAX_RETRY_WAIT)
)
@st.cache_data(ttl=CACHE_TTL_MODELS)
def get_available_models():
    """Fetch list of available rental and selling models from API with caching."""
    try:
        headers = {
            'Accept': 'application/json'
        }
        response = requests.get(
            f"{API_URL}/models",
            headers=headers,
            timeout=MODEL_INFO_TIMEOUT
        )
        if response.status_code == 200:
            return response.json()
        else:
            error_detail = f"Failed to fetch models (Status Code: {response.status_code})"
            st.warning(error_detail)
            return {"rental_models": [], "selling_models": []}
    except requests.exceptions.Timeout:
        st.warning("Request timed out while fetching models")
        return {"rental_models": [], "selling_models": []}
    except requests.exceptions.ConnectionError:
        st.warning("Connection error while fetching models")
        return {"rental_models": [], "selling_models": []}
    except Exception as e:
        st.warning(f"Error fetching models: {str(e)}")
        return {"rental_models": [], "selling_models": []}

@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=MIN_RETRY_WAIT, max=MAX_RETRY_WAIT)
)
@st.cache_data(ttl=CACHE_TTL_MODELS)
def get_model_requirements(model_type):
    """Fetch required features and metrics for specified model type."""
    try:
        headers = {
            'Accept': 'application/json'
        }
        response = requests.get(
            f"{API_URL}/model-requirements/{model_type}",
            headers=headers,
            timeout=MODEL_INFO_TIMEOUT
        )
        if response.status_code == 200:
            return response.json()
        else:
            error_detail = f"Failed to fetch model requirements (Status Code: {response.status_code})"
            st.warning(error_detail)
            return {"required_columns": []}
    except requests.exceptions.Timeout:
        st.warning("Request timed out while fetching model requirements")
        return {"required_columns": []}
    except requests.exceptions.ConnectionError:
        st.warning("Connection error while fetching model requirements")
        return {"required_columns": []}
    except Exception as e:
        st.warning(f"Error fetching model requirements: {str(e)}")
        return {"required_columns": []}

def format_price(price, model_type):
    """Format price with euro symbol and add /month suffix for rentals."""
    if model_type == "rental":
        return f"€{price:,.2f} / month"
    else:
        return f"€{price:,.2f}"

def add_to_history(property_data, result, model_type):
    """Store prediction in session state history, keeping last 10 predictions."""
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    
    # Create history item
    import pandas as pd
    history_item = {
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "model_type": model_type,
        "property_data": property_data,
        "result": result
    }
    
    # Add to history
    st.session_state.prediction_history.append(history_item)
    
    # Keep only last 10 items
    if len(st.session_state.prediction_history) > 10:
        st.session_state.prediction_history = st.session_state.prediction_history[-10:]
