import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from aruodas_scraper.database.database_manager import get_data, get_db_connection

@st.cache_data(ttl=600)  # Cache for 10 minutes
def load_market_data_streamlit(listing_type="selling", start_date=None, end_date=None):
    """Load market data using Streamlit connection management."""
    try:
        conn = get_streamlit_connection()
        if conn is None:
            return pd.DataFrame()
        
        table_name = "butai" if listing_type == "selling" else "butai_rent"
        
        # Build date filter
        date_filter = ""
        if start_date and end_date:
            date_filter = f"AND scrape_date BETWEEN '{start_date}' AND '{end_date}'"
        
        query = f"""
        SELECT 
            price,
            plotas,
            kambariu_sk,
            city,
            pastato_tipas,
            metai,
            scrape_date,
            aukstas,
            aukstu_sk
        FROM {table_name}
        WHERE price IS NOT NULL 
        AND plotas IS NOT NULL
        AND price > 0
        AND plotas > 0
        {date_filter}
        """
        
        df = conn.query(query, ttl=300)  # Cache query results for 5 minutes
        return df
        
    except Exception as e:
        st.error(f"Failed to load market data: {e}")
        # Fallback to original method
        return load_market_data_fallback(listing_type, start_date, end_date)

def load_market_data_fallback(listing_type="selling", start_date=None, end_date=None):
    """Fallback method using original database connection."""
    try:
        from aruodas_scraper.database.database_manager import get_data, get_db_connection
        
        table_name = "butai_rent" if listing_type == "rental" else "butai"
        
        conn = get_db_connection()
        if conn is None:
            return pd.DataFrame()

        # Build date filter
        date_filter = ""
        if start_date and end_date:
            date_filter = f"AND scrape_date BETWEEN '{start_date}' AND '{end_date}'"

        query = f"""
        SELECT 
            price,
            plotas,
            kambariu_sk,
            city,
            pastato_tipas,
            metai,
            scrape_date,
            aukstas,
            aukstu_sk
        FROM {table_name}
        WHERE price IS NOT NULL 
        AND plotas IS NOT NULL
        AND price > 0
        AND plotas > 0
        {date_filter}
        """

        df = get_data(query=query, table_name=table_name)
        return df

    except Exception as e:
        print(f"Error in fallback data loading: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_market_data(listing_type="selling", start_date=None, end_date=None):
    """Load market data from database with caching."""
    # Get all data from database
    table_name = "butai_rent" if listing_type == "rental" else "butai"
    
    # First check if table exists and get its structure
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()

    try:
        # Get column names from the table
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT COLUMN_NAME 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME = '{table_name}'
        """)
        columns = [row[0] for row in cursor.fetchall()]
        
        # Define required columns and their alternatives
        required_columns = {
            'miestas': ['miestas', 'city'],
            'kaina': ['kaina', 'price'],
            'plotas': ['plotas', 'plotas'],
            'namo_tipas': ['namo_tipas', 'pastato_tipas'],
            'kambariu_sk': ['kambariu_sk', 'kambariu_sk'],
            'scrape_date': ['scrape_date', 'scrape_date'],
            'metai': ['metai', 'metai', 'construction_year']
        }
        
        # Map actual column names to our expected names
        column_mapping = {}
        for expected, alternatives in required_columns.items():
            found = next((col for col in alternatives if col in columns), None)
            if found:
                column_mapping[found] = expected

        # Build the query using the actual column names
        actual_cols = [col for col in columns if col in list(column_mapping.keys())]
        select_parts = [f"{col} as {column_mapping[col]}" for col in actual_cols]
        
        # Add date filter if dates are provided
        date_filter = ""
        if start_date and end_date:
            date_filter = f"AND {column_mapping.get('scrape_date', 'scrape_date')} BETWEEN '{start_date}' AND '{end_date}'"
        
        query = f"""
        SELECT {', '.join(select_parts)}
        FROM {table_name}
        WHERE {column_mapping.get('kaina', 'price')} IS NOT NULL 
        AND {column_mapping.get('plotas', 'plotas')} IS NOT NULL
        {date_filter}
        """

        df = get_data(query=query, table_name=table_name)
        
        # Rename columns for consistency
        df = df.rename(columns={
            'miestas': 'city',
            'kaina': 'price',
            'namo_tipas': 'pastato_tipas'
        })
        
        return df

    except Exception as e:
        print(f"Error loading market data: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

@st.cache_data(ttl=3600)
def get_cities(listing_type="selling"):
    """Get list of available cities."""
    df = load_market_data(listing_type)
    if 'city' in df.columns and not df.empty:
        cities = sorted(df['city'].unique())
        return ['All Lithuania'] + list(cities)
    return ['All Lithuania']  # Return default if no data

@st.cache_data(ttl=3600)
def get_cities_streamlit(listing_type="selling"):
    """Get list of available cities using Streamlit connection."""
    try:
        conn = get_streamlit_connection()
        if conn is None:
            return ['All Lithuania']
        
        table_name = "butai" if listing_type == "selling" else "butai_rent"
        
        query = f"""
        SELECT DISTINCT city 
        FROM {table_name} 
        WHERE city IS NOT NULL 
        ORDER BY city
        """
        
        df = conn.query(query, ttl=3600)
        cities = df['city'].tolist()
        return ['All Lithuania'] + cities
        
    except Exception as e:
        st.error(f"Failed to load cities: {e}")
        return ['All Lithuania']

def filter_by_location(df, location):
    """Filter dataframe by selected location."""
    if location != 'All Lithuania':
        return df[df['city'] == location]
    return df

def calculate_basic_stats(df):
    """Calculate basic market statistics."""
    stats = {
        'total_listings': len(df),
        'avg_price': df['price'].mean(),
        'median_price': df['price'].median(),
        'avg_price_per_m2': (df['price'] / df['plotas']).mean(),
        'avg_area': df['plotas'].mean(),
    }
    return stats

def calculate_time_trends(df, listing_type):
    """Calculate price per square meter trends over time."""
    df['scrape_date'] = pd.to_datetime(df['scrape_date'])
    df['price_per_m2'] = df['price'] / df['plotas']
    
    # Get earliest and latest dates
    earliest_date = df['scrape_date'].min()
    latest_date = df['scrape_date'].max()
    
    # Calculate metrics for first and last day
    first_day = df[df['scrape_date'] == earliest_date]
    last_day = df[df['scrape_date'] == latest_date]
    
    first_avg_m2 = first_day['price_per_m2'].mean()
    last_avg_m2 = last_day['price_per_m2'].mean()
    
    price_change_m2 = last_avg_m2 - first_avg_m2
    price_change_pct = (price_change_m2 / first_avg_m2 * 100) if first_avg_m2 else 0
    
    label = "Rent/m²" if listing_type == "rental" else "Price/m²"
    return {
        'period': f"{earliest_date.strftime('%Y-%m-%d')} to {latest_date.strftime('%Y-%m-%d')}",
        'total_days': (latest_date - earliest_date).days,
        'price_change': price_change_m2,
        'price_change_pct': price_change_pct,
        'first_avg_m2': first_avg_m2,
        'last_avg_m2': last_avg_m2,
        'label': label
    }

def create_price_distribution_plot(df, listing_type):
    """Create box plot for price distribution by property type."""
    title = 'Monthly Rent Distribution by Property Type' if listing_type == 'rental' else 'Price Distribution by Property Type'
    yaxis_title = 'Monthly Rent (€)' if listing_type == 'rental' else 'Price (€)'
    
    fig = px.box(
        df,
        x='pastato_tipas',
        y='price',
        title=title
    )
    fig.update_layout(
        xaxis_title='Property Type',
        yaxis_title=yaxis_title
    )
    return fig

def create_time_series_plot(df, listing_type):
    """Create time series plot with dual y-axes."""
    df['scrape_date'] = pd.to_datetime(df['scrape_date'])
    df['price_per_m2'] = df['price'] / df['plotas']
    
    # Calculate daily averages
    daily_avg = df.groupby('scrape_date').agg({
        'price': 'mean',
        'price_per_m2': 'mean'
    }).reset_index()
    
    # Create plot
    title = 'Price Trends Over Time' if listing_type == 'selling' else 'Rent Trends Over Time'
    y1_label = 'Average Price (€)' if listing_type == 'selling' else 'Average Monthly Rent (€)'
    y2_label = 'Price/m² (€)' if listing_type == 'selling' else 'Rent/m² (€)'
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Add price line on primary y-axis
    fig.add_trace(go.Scatter(
        x=daily_avg['scrape_date'],
        y=daily_avg['price'],
        name=y1_label,
        line=dict(color='blue')
    ))
    
    # Add price per m² line on secondary y-axis
    fig.add_trace(go.Scatter(
        x=daily_avg['scrape_date'],
        y=daily_avg['price_per_m2'],
        name=y2_label,
        line=dict(color='red'),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title=y1_label,
        yaxis2=dict(
            title=y2_label,
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

def create_price_vs_size_plot(df, listing_type):
    """Create scatter plot of price vs size with trendline."""
    title = 'Monthly Rent vs Size' if listing_type == 'rental' else 'Price vs Size'
    y_label = 'Monthly Rent (€)' if listing_type == 'rental' else 'Price (€)'
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=df['plotas'],
        y=df['price'],
        mode='markers',
        name='Properties',
        marker=dict(
            size=8,
            opacity=0.6
        )
    ))
    
    # Add trendline
    X = sm.add_constant(df['plotas'])
    model = sm.OLS(df['price'], X).fit()
    x_range = np.linspace(df['plotas'].min(), df['plotas'].max(), 100)
    y_pred = model.params[0] + model.params[1] * x_range
    
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_pred,
        mode='lines',
        name=f'Trendline (R² = {model.rsquared:.3f})',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Area (m²)',
        yaxis_title=y_label,
        showlegend=True
    )
    return fig

def create_price_vs_year_plot(df, listing_type):
    """Create scatter plot of price vs construction year grouped by decades."""
    # Return empty figure if no data
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No data available",
            xaxis_title="Construction Year",
            yaxis_title="Price (€)"
        )
        return fig
    
    # Filter out rows with missing or invalid years
    df = df[df['metai'].notna() & (df['metai'] >= 1900) & (df['metai'] <= datetime.now().year)]
    
    # Return empty figure if no valid data after filtering
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No valid construction year data available",
            xaxis_title="Construction Year",
            yaxis_title="Price (€)"
        )
        return fig
    
    # Create decade groups
    df['decade'] = (df['metai'] // 10) * 10
    df['decade'] = df['decade'].map(lambda x: f"{int(x)}s")
    
    title = 'Monthly Rent vs Construction Year' if listing_type == 'rental' else 'Price vs Construction Year'
    y_label = 'Monthly Rent (€)' if listing_type == 'rental' else 'Price (€)'
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=df['metai'],
        y=df['price'],
        mode='markers',
        marker=dict(
            color=df['decade'].astype('category').cat.codes,
            colorscale='Viridis',
            opacity=0.6,
            size=8
        ),
        text=df['decade'],
        name='Properties'
    ))
    
    # Add trend lines per decade
    decades = sorted(df['decade'].unique())
    for decade in decades:
        decade_data = df[df['decade'] == decade]
        # Skip decades with insufficient data points
        if len(decade_data) <= 2:
            continue
            
        try:
            X = sm.add_constant(decade_data['metai'])
            model = sm.OLS(decade_data['price'], X).fit()
            
            # Only add trendline if the model fit is successful
            if hasattr(model, 'params') and len(model.params) >= 2:
                x_range = np.linspace(decade_data['metai'].min(), decade_data['metai'].max(), 10)
                y_pred = model.params[0] + model.params[1] * x_range
                
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=y_pred,
                    mode='lines',
                    name=f'{decade} trend',
                    line=dict(dash='dash'),
                    showlegend=False
                ))
        except Exception as e:
            print(f"Error creating trendline for {decade}: {str(e)}")
            continue
    
    fig.update_layout(
        title=title,
        xaxis_title='Construction Year',
        yaxis_title=y_label,
        showlegend=True,
        xaxis=dict(
            showgrid=True,
            dtick=10  # Show gridlines every 10 years
        )
    )
    return fig

def create_property_type_prices_plot(df, listing_type):
    """Create bar chart for average prices by property type."""
    avg_prices = df.groupby('pastato_tipas')['price'].mean().round(2)
    title = 'Average Monthly Rent by Property Type' if listing_type == 'rental' else 'Average Price by Property Type'
    y_label = 'Average Monthly Rent (€)' if listing_type == 'rental' else 'Average Price (€)'
    
    fig = px.bar(
        x=avg_prices.index,
        y=avg_prices.values,
        title=title,
        labels={'x': 'Property Type', 'y': y_label}
    )
    return fig

def display_market_analysis():
    """Display market analysis section with Streamlit connection management."""
    st.header("📊 Market Analysis")
    
    # Add type selector, date range, and refresh button
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        listing_type = st.radio(
            "Select Type",
            ["selling", "rental"],
            format_func=lambda x: "Rental Properties" if x == "rental" else "Properties for Sale",
            horizontal=True
        )
    
    with col2:
        default_end_date = datetime.now()
        default_start_date = datetime(default_end_date.year, 1, 1)
        date_range = st.date_input(
            "Date Range",
            value=(default_start_date, default_end_date),
            help="Filter data by date range"
        )
    
    with col3:
        if st.button("🔄 Refresh Data"):
            # Clear all caches
            load_market_data_streamlit.clear()
            get_cities_streamlit.clear()
            st.cache_data.clear()
            st.success("Data refreshed!")

    # Test connection first
    try:
        conn = get_streamlit_connection()
        if conn is None:
            st.error("❌ Unable to connect to database. Market analysis is temporarily unavailable.")
            st.info("💡 This feature requires database access. Please try again later or contact support.")
            return
        
        # Test with a simple query
        test_df = conn.query("SELECT 1 as test", ttl=0)
        if test_df.empty:
            st.error("❌ Database connection test failed.")
            return
            
    except Exception as e:
        st.error("❌ Database connection failed. Market analysis is temporarily unavailable.")
        st.info("💡 This feature requires database access. Please try again later.")
        st.exception(e)  # For debugging
        return

    # Location selector
    selected_location = st.selectbox(
        "Select Location",
        get_cities_streamlit(listing_type),
        index=0
    )

    # Load and filter data
    with st.spinner("Loading market data..."):
        start_date, end_date = date_range if len(date_range) == 2 else (None, None)
        df = load_market_data_streamlit(listing_type, start_date, end_date)
        filtered_df = filter_by_location(df, selected_location)

    if len(filtered_df) == 0:
        st.warning("No data available for the selected location and date range.")
        return

    # Create tabs for different analysis views
    tab1, tab2 = st.tabs(["📈 Market Overview", "🔍 Property Analysis"])
    
    with tab1:
        # Display basic statistics
        stats = calculate_basic_stats(filtered_df)
        trends = calculate_time_trends(filtered_df, listing_type)
        
        # Show metrics in two rows
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Listings", f"{stats['total_listings']:,}")
        with col2:
            price_label = "Average Monthly Rent" if listing_type == "rental" else "Average Price"
            st.metric(price_label, f"€{stats['avg_price']:,.2f}")
        with col3:
            price_m2_label = "Average Rent/m²" if listing_type == "rental" else "Average Price/m²"
            st.metric(price_m2_label, f"€{stats['avg_price_per_m2']:,.2f}")

        # Show trend information
        st.subheader("Price Trends")
        st.markdown(f"""
        Period: {trends['period']} ({trends['total_days']} days)
        - {trends['label']} Change: €{trends['price_change']:,.2f}
        - Percentage Change: {trends['price_change_pct']:.1f}%
        """)

        # Display time series plot
        st.plotly_chart(create_time_series_plot(filtered_df, listing_type), use_container_width=True)
    
    with tab2:
        # Add price range filter
        price_range = st.slider(
            "Filter by Price Range (€)",
            min_value=float(filtered_df['price'].min()),
            max_value=float(filtered_df['price'].max()),
            value=(float(filtered_df['price'].min()), float(filtered_df['price'].max())),
            format="%.2f"
        )
        
        # Add area range filter
        area_range = st.slider(
            "Filter by Area (m²)",
            min_value=float(filtered_df['plotas'].min()),
            max_value=float(filtered_df['plotas'].max()),
            value=(float(filtered_df['plotas'].min()), float(filtered_df['plotas'].max())),
            format="%.1f"
        )
        
        # Filter data based on selections
        mask = (
            (filtered_df['price'].between(price_range[0], price_range[1])) &
            (filtered_df['plotas'].between(area_range[0], area_range[1]))
        )
        analysis_df = filtered_df[mask]
        
        if len(analysis_df) == 0:
            st.warning("No properties match the selected criteria.")
            return
        
        # Show summary statistics first
        st.subheader("Summary Statistics")
        stats_df = pd.DataFrame({
            'Metric': ['Count', 'Average Price', 'Median Price', 'Average Price/m²', 'Average Area'],
            'Value': [
                f"{len(analysis_df):,}",
                f"€{analysis_df['price'].mean():,.2f}",
                f"€{analysis_df['price'].median():,.2f}",
                f"€{(analysis_df['price'] / analysis_df['plotas']).mean():,.2f}",
                f"{analysis_df['plotas'].mean():,.1f} m²"
            ]
        })
        st.table(stats_df)

        # Display visualizations
        st.subheader("Property Distribution")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_price_distribution_plot(analysis_df, listing_type), use_container_width=True)
            st.plotly_chart(create_price_vs_size_plot(analysis_df, listing_type), use_container_width=True)
        with col2:
            st.plotly_chart(create_property_type_prices_plot(analysis_df, listing_type), use_container_width=True)
            st.plotly_chart(create_price_vs_year_plot(analysis_df, listing_type), use_container_width=True)
