# Aruodas Real Estate Analytics

A comprehensive real estate analytics platform that combines web scraping, machine learning, and data visualization to provide insights into the Lithuanian real estate market through Aruodas.lt data.

## Live Demo

🔗 [Try the App](https://real-estate-lt.streamlit.app/) - Explore Lithuanian real estate market analysis and price predictions

## Features

- **Data Collection**: Automated scraping of real estate listings from Aruodas.lt
- **Machine Learning**: Price prediction models for both rental and sales properties
- **Market Analysis**: Interactive tools for analyzing real estate market trends
- **Automated Pipeline**: Weekly data collection and model retraining via GitHub Actions
- **Azure Integration**: Seamless integration with Azure SQL Database and Blob Storage

## Architecture & Data Flow

```
Data Collection & Storage
🏠 Home Network
  └── 🤖 Scraper
      └── 📊 Aruodas.lt Data
          └── 💾 Azure SQL Database

Model Training Pipeline (Weekly)
⏰ Monday Midnight (GitHub Actions)
  ├── 📥 Pull Data from Azure SQL
  ├── 🧮 Train Models
  └── 📤 Save to Azure Blob Storage

Services
📡 Azure App Service
  └── ⚡ FastAPI
      ├── 📥 Load Models from Blob
      └── 🎯 Serve Predictions

📱 Streamlit App
  ├── 📊 Market Analysis
  ├── 💰 Price Predictions
  └── 🗺️ Property Comparison

Data Flow: 
Aruodas.lt 🔄 Scraper ➡️ SQL DB ➡️ Training ➡️ Blob Storage ⬅️ API ⬅️ Streamlit
```

## Project Structure

```
project/
├── api/                      # FastAPI service for predictions
│   ├── main.py              # API endpoints and model serving
│   └── requirements.txt      # API-specific dependencies
├── app/                      # Streamlit web application
│   ├── streamlit_app.py     # Main app file
│   └── components/          # UI and functionality components
├── aruodas_scraper/         # Core package
│   ├── config/             # Configuration and settings
│   │   ├── settings.py    # Main configuration file
│   │   └── utils.py       # Utility functions
│   ├── scraper/           # Web scraping using Selenium
│   ├── database/          # Azure SQL Database operations
│   ├── preprocessing/     # Data cleaning and feature engineering
│   └── training/         # Model training and evaluation
├── model_output/          # Trained ML models and artifacts
│   ├── rental/           # Rental price models
│   └── selling/          # Sales price models
```

## Prerequisites

- Python 3.11 or higher
- Chrome/Chromium browser
- ODBC Driver 18 for SQL Server
- Azure SQL Database
- Azure Blob Storage

## Getting Started

### Quick Start
1. 🚀 [Try the live app](https://real-estate-lt.streamlit.app/)
2. 📖 Check the [Configuration](#configuration) section for local setup
3. 🛠️ Follow the [Usage](#usage) guide to run the pipeline

### Configuration

1. Clone and setup:
```bash
git clone https://github.com/tdzedulionis/lt_real_estate_project
cd aruodas_scraper
cp .env.example .env
cp app/.env.example app/.env
```

2. Configure environment variables:

Main configuration (.env):
```
DB_SERVER=your_server_name
DB_NAME=your_database_name
DB_USER=your_username
DB_PASSWORD=your_password
HOME_PROXY=your_proxy_server  # Optional
BLOB_CONNECTION_STRING=your_connection_string
BLOB_CONTAINER_NAME=your_container_name
```

Streamlit app configuration (app/.env):
```
API_URL=https://your-fastapi-service.azurewebsites.net/
CACHE_TTL_PREDICTIONS=60
CACHE_TTL_MODELS=600
PREDICTION_TIMEOUT=60
MODEL_INFO_TIMEOUT=30
MAX_RETRIES=5
MIN_RETRY_WAIT=4
MAX_RETRY_WAIT=30
```

## Usage

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the pipeline:
```bash
python main.py
```

This will scrape data, train models, and upload them to Azure Blob Storage.

### GitHub Actions Pipeline

The automated weekly pipeline:
1. Runs Monday at midnight UTC
2. Scrapes property data
3. Trains and updates models
4. Uploads to Azure Blob Storage

To configure:
1. Fork this repository
2. Add repository secrets:
   - `DB_SERVER`
   - `DB_NAME`
   - `DB_USER`
   - `DB_PASSWORD`
   - `HOME_PROXY` (optional)
   - `BLOB_CONNECTION_STRING`
   - `BLOB_CONTAINER_NAME`
   - `SCHEDULED_MAX_PAGES`

## Contributing

1. Fork the repository
2. Create your feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.
