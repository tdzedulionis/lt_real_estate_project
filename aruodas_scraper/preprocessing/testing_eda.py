"""Exploratory data analysis module for real estate data visualization and insights."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap, MarkerCluster
import plotly.express as px

from aruodas_scraper.database.database_manager import get_data
from aruodas_scraper.preprocessing.data_manipulation import clean_data, convert_numeric_if_possible, engineer_features

df = get_data()
df = clean_data(df)
df = df.replace({"Not Specified": None})
for column_name in df.columns:
    df = convert_numeric_if_possible(df, column_name)
df = engineer_features(df)

# Display basic information
print("Dataset shape:", df.shape)
print("\nData types:")
print(df.dtypes)

# Basic statistics summary
print("\nBasic statistics for numerical features:")
print(df.describe())

# Check for missing values
print("\nMissing values per column:")
print(df.isna().sum())

def price_vs_area_plot(city=None):
    """Plot property prices against area with regression line, optionally filtered by city."""
    plt.figure(figsize=(12, 7))
    
    if city:
        # Filter data for specific city
        city_data = df[df['city'] == city]
        
        # Check if we have enough data
        if len(city_data) < 5:
            print(f"Warning: Only {len(city_data)} data points for {city}")
            return
        
        # Create scatter plot with regression line for single city
        sns.regplot(
            x='plotas', 
            y='price', 
            data=city_data,
            scatter_kws={'alpha': 0.6},
            line_kws={'color': 'red'}
        )
        plt.title(f'Price vs. Area in {city} (n={len(city_data)})')
        
    else:
        # Use all data but color by city
        # Get top 5 cities by count to avoid too many colors
        top_cities = df['city'].value_counts().nlargest(5).index
        plot_data = df[df['city'].isin(top_cities)]
        
        # Plot each city with different color
        for city_name in top_cities:
            city_data = plot_data[plot_data['city'] == city_name]
            plt.scatter(
                city_data['plotas'], 
                city_data['price'], 
                alpha=0.6, 
                label=f"{city_name} (n={len(city_data)})"
            )
            
        # Add overall regression line
        sns.regplot(
            x='plotas', 
            y='price', 
            data=plot_data,
            scatter=False,  # Don't show points again
            line_kws={'color': 'black', 'linestyle': '--'}
        )
        
        plt.title('Price vs. Area by City')
        plt.legend()
    
    # Common elements
    plt.xlabel('Area (sq.m)')
    plt.ylabel('Price (€)')
    plt.grid(True, alpha=0.3)
    
    return plt.show()

def price_distribution_plot():
    """Create histogram of property prices with mean and median markers."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df['price'], kde=True, bins=30)
    
    mean_price = df['price'].mean()
    median_price = df['price'].median()
    
    plt.title('Distribution of Property Prices')
    plt.xlabel('Price (€)')
    plt.axvline(mean_price, color='r', linestyle='--', 
                label=f'Mean: {mean_price:,.0f} €')
    plt.axvline(median_price, color='g', linestyle='--', 
                label=f'Median: {median_price:,.0f} €')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt.show()


def price_per_sqm_plot():
    """Create histogram of price per square meter with mean and median markers."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df['price_per_sqm'], kde=True, bins=30)
    
    mean_price = df['price_per_sqm'].mean()
    median_price = df['price_per_sqm'].median()
    
    plt.title('Distribution of Price per Square Meter')
    plt.xlabel('Price per sq.m (€/m²)')
    plt.axvline(mean_price, color='r', linestyle='--', 
                label=f'Mean: {mean_price:,.0f} €/m²')
    plt.axvline(median_price, color='g', linestyle='--', 
                label=f'Median: {median_price:,.0f} €/m²')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt.show()


def rooms_vs_price_plot():
    """Create box plot showing price distribution by number of rooms."""
    plt.figure(figsize=(12, 6))
    
    # Convert kambariu_sk to string for better x-axis labels
    room_data = df.copy()
    room_data['kambariu_sk'] = room_data['kambariu_sk'].astype(str)
    
    # Create the box plot
    sns.boxplot(x='kambariu_sk', y='price', data=room_data)
    
    # Add count labels above each box
    for i, room_count in enumerate(sorted(room_data['kambariu_sk'].unique())):
        count = len(room_data[room_data['kambariu_sk'] == room_count])
        median = room_data[room_data['kambariu_sk'] == room_count]['price'].median()
        plt.text(i, median, f'n={count}', ha='center', va='bottom')
    
    plt.title('Price Distribution by Number of Rooms')
    plt.xlabel('Number of Rooms')
    plt.ylabel('Price (€)')
    plt.grid(True, alpha=0.3)
    
    return plt.show()


def create_property_map(cluster=True):
    """Create interactive map of property locations with clustered markers or heatmap visualization."""
    # Filter rows with valid coordinates
    map_data = df.copy()
    
    # Handle non-numeric values in latitude/longitude
    map_data['latitude'] = pd.to_numeric(map_data['latitude'], errors='coerce')
    map_data['longitude'] = pd.to_numeric(map_data['longitude'], errors='coerce')
    
    # Drop rows with missing coordinates
    map_data = map_data.dropna(subset=['latitude', 'longitude'])
    
    # Create a base map centered around the mean coordinates
    base_map = folium.Map(
        location=[map_data['latitude'].mean(), map_data['longitude'].mean()],
        zoom_start=13,
        tiles='CartoDB positron'  # cleaner map style
    )
    
    if cluster:
        # Add clustered markers for each property
        marker_cluster = MarkerCluster().add_to(base_map)
        
        for idx, row in map_data.iterrows():
            # Convert price and other values to appropriate types
            try:
                price = f"{int(row['price']):,} €" if pd.notnull(row['price']) else "N/A"
                area = f"{float(row['plotas']):.1f} m²" if pd.notnull(row['plotas']) else "N/A"
                rooms = str(row['kambariu_sk']) if pd.notnull(row['kambariu_sk']) and row['kambariu_sk'] != 'Not Specified' else "N/A"
                
                # Calculate price per sqm safely
                if pd.notnull(row['price_per_sqm']):
                    price_per_sqm = f"{float(row['price_per_sqm']):,.0f} €/m²"
                elif pd.notnull(row['price']) and pd.notnull(row['plotas']) and float(row['plotas']) > 0:
                    price_per_sqm = f"{float(row['price'])/float(row['plotas']):,.0f} €/m²"
                else:
                    price_per_sqm = "N/A"
                
                popup_text = f"""
                <b>Price:</b> {price}<br>
                <b>Area:</b> {area}<br>
                <b>Rooms:</b> {rooms}<br>
                <b>Price/m²:</b> {price_per_sqm}
                """
                
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup=folium.Popup(popup_text, max_width=300),
                    icon=folium.Icon(color='blue', icon='home')
                ).add_to(marker_cluster)
            except (ValueError, TypeError) as e:
                print(f"Skipping property with invalid data: {e}")
        
        filename = 'property_locations.html'
    else:
        # Create a heatmap based on property prices
        heat_data = []
        
        for idx, row in map_data.iterrows():
            try:
                # Only include rows with valid price data
                if pd.notnull(row['price']) and row['price'] != 'Not Specified':
                    price = float(row['price'])
                    heat_data.append([row['latitude'], row['longitude'], price])
            except (ValueError, TypeError):
                # Skip rows with invalid data
                continue
        
        HeatMap(heat_data).add_to(base_map)
        filename = 'price_heatmap.html'
    
    # Save the map
    base_map.save(filename)
    print(f"Map saved as {filename}")
    
    return base_map

def city_price_comparison_plot():
    """Create bar plot comparing median property prices and price per sqm across cities."""
    plt.figure(figsize=(14, 8))
    
    # Get top 10 cities by count
    top_cities = df['city'].value_counts().head(10).index
    city_data = df[df['city'].isin(top_cities)]
    
    # Calculate median prices by city
    city_stats = city_data.groupby('city').agg({
        'price': 'median',
        'price_per_sqm': 'median',
        'city': 'size'
    }).rename(columns={'city': 'count'}).sort_values('price', ascending=False)
    
    # Create subplot with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Plot median price bars
    x = np.arange(len(city_stats.index))
    width = 0.4
    bars1 = ax1.bar(x - width/2, city_stats['price'], width, color='steelblue', label='Median Price')
    ax1.set_ylabel('Median Price (€)', color='steelblue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='steelblue')
    
    # Create second y-axis for price per sqm
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, city_stats['price_per_sqm'], width, color='darkred', label='Median Price/m²')
    ax2.set_ylabel('Median Price/m² (€/m²)', color='darkred', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='darkred')
    
    # Add labels and formatting
    ax1.set_xticks(x)
    ax1.set_xticklabels(city_stats.index, rotation=45, ha='right')
    plt.title('Median Property Prices by City', fontsize=14, fontweight='bold')
    
    # Add count labels on top of bars
    for i, (city, row) in enumerate(city_stats.iterrows()):
        ax1.text(i - width/2, row['price'], f'{int(row["price"]):,}€', 
                 ha='center', va='bottom', fontsize=9)
        ax2.text(i + width/2, row['price_per_sqm'], f'{int(row["price_per_sqm"]):,}€/m²', 
                 ha='center', va='bottom', fontsize=9)
        plt.text(i, 0, f'n={row["count"]}', ha='center', va='bottom')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    return plt.show()


def price_heatmap_by_features():
    """Create correlation heatmap between property price and other numeric features."""
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Identify columns to drop (contain 'price' but are not exactly 'price')
    cols_to_drop = [col for col in numeric_df.columns if 'price' in col.lower() and col.lower() != 'price']

    # Drop the identified columns
    numeric_df = numeric_df.drop(columns=cols_to_drop)
    
    # Calculate correlations
    corr = numeric_df.corr()
    
    # Sort columns by correlation with price
    corr_with_price = corr['price'].sort_values(ascending=False)
    sorted_columns = corr_with_price.index.tolist()
    
    # Create a more focused correlation matrix
    focused_corr = corr.loc[sorted_columns[:10], sorted_columns[:10]]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(focused_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", linewidths=0.5)
    plt.title('Correlation Between Price and Property Features', fontsize=14)
    plt.tight_layout()
    return plt.show()

# Basic EDA plots
price_distribution_plot()
price_per_sqm_plot()
rooms_vs_price_plot()
price_heatmap_by_features()

# City comparison plot
price_vs_area_plot()  # All cities
price_vs_area_plot(city='Vilnius')  # Single city example
city_price_comparison_plot() 

# Maps
create_property_map(cluster=True)  # Clustered markers
create_property_map(cluster=False)  # Heatmap
