# Aruodas Real Estate Analytics

A comprehensive real estate analytics platform that combines web scraping, machine learning, and data visualization to provide insights into the Lithuanian real estate market through Aruodas.lt data.

## Project Components

```
project/
├── api/                      # FastAPI service for predictions
│   ├── main.py              # API endpoints
│   └── requirements.txt      # API-specific dependencies
├── app/                      # Streamlit web application
│   ├── streamlit_app.py     # Main app file
│   └── components/          # UI and functionality components
├── aruodas_scraper/         # Core package
│   ├── config/             # Configuration settings
│   ├── scraper/            # Web scraping logic
│   ├── database/           # Database operations
│   ├── preprocessing/      # Data preprocessing
│   └── training/          # Model training and analysis
├── model_output/           # Trained ML models
│   ├── rental/            # Rental price models
│   └── selling/           # Sales price models
```

## Features

- **Data Collection**: Automated scraping of real estate listings from Aruodas.lt
- **Machine Learning**: Price prediction models for both rental and sales properties
- **Automated Pipeline**: Daily data collection and model retraining via GitHub Actions
- **Azure Integration**: Seamless integration with Azure SQL Database and Blob Storage
- **Market Analysis**: Tools for analyzing real estate market trends

## Prerequisites

- Python 3.11 or higher
- Chrome/Chromium browser
- ODBC Driver 18 for SQL Server
- Azure SQL Database
- Azure Blob Storage

## Configuration

1. Clone the repository:
```bash
git clone https://github.com/tdzedulionis/lt_real_estate_project
cd aruodas_scraper
```

2. Copy .env.example to .env and configure:
```bash
cp .env.example .env
```

Required environment variables:
```
# Database Configuration
DB_SERVER=your_server_name
DB_NAME=your_database_name
DB_USER=your_username
DB_PASSWORD=your_password

# Azure Blob Storage
BLOB_CONNECTION_STRING=your_connection_string
BLOB_CONTAINER_NAME=your_container_name
```

3. Configure scraping parameters in `aruodas_scraper/config/settings.py`:
```python
SCRAPER_CONFIG = {
    'max_pages': '1',  # number of pages or 'all'
    'category': 'butai',  # 'butai' for selling, 'butu-nuoma' for rental
    'location': '',  # city name or empty for all
    'wait_time': 2  # seconds between requests
}
```

## Usage

### Local Execution

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the scraper and training pipeline:
```bash
python main.py
```

The script will:
- Scrape property listings from Aruodas.lt
- Store data in Azure SQL Database
- Train machine learning models
- Upload models to Azure Blob Storage

### GitHub Actions Pipeline

The project includes an automated CI/CD pipeline that:
1. Runs daily at midnight UTC to collect new data
2. Executes in sequence:
   - Scrapes selling properties
   - Trains and updates selling models
   - Scrapes rental properties
   - Trains and updates rental models
3. Uploads all updated models to Azure Blob Storage

To configure the pipeline:
1. Fork this repository
2. Add the following secrets to your GitHub repository settings:
   - `DB_SERVER`
   - `DB_NAME`
   - `DB_USER`
   - `DB_PASSWORD`
   - `BLOB_CONNECTION_STRING`
   - `BLOB_CONTAINER_NAME`
   - `SCHEDULED_MAX_PAGES` (number of pages to scrape in scheduled runs)

3. Manual pipeline execution:
   - Go to Actions tab in GitHub
   - Select "ML Pipeline"
   - Click "Run workflow"
   - Enter the number of pages to scrape (or 'all')

## Project Structure Details

### Core Package (aruodas_scraper/)
- `config/`: Global configuration and settings
  - `settings.py`: Main configuration file
  - `utils.py`: Utility functions
- `scraper/`: Web scraping implementation using Selenium
- `database/`: Azure SQL Database operations
- `preprocessing/`: Data cleaning and feature engineering
- `training/`: Model training and evaluation

### API Service (api/)
- FastAPI implementation for real-time predictions
- Model loading and management
- Azure Blob Storage integration

### Web Application (app/)
- Streamlit dashboard
- Market analysis visualizations
- Interactive price predictions
- Property comparison tools

### Model Output (model_output/)
- Trained model files
- Performance metrics
- Feature importance analysis
- Model artifacts

## Contributing

1. Fork the repository
2. Create your feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.
