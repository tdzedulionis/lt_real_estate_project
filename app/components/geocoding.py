import streamlit as st
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
from tenacity import retry, stop_after_attempt, wait_exponential

# Define Lithuania's geographical boundaries
# These coordinates represent a bounding box around Lithuania
LITHUANIA_BOUNDS = {
    "min_lat": 53.8,   # Southern boundary
    "max_lat": 56.5,   # Northern boundary
    "min_lon": 20.9,   # Western boundary
    "max_lon": 26.8    # Eastern boundary
}

def is_in_lithuania(lat, lon):
    """Check if the coordinates are within Lithuania's boundaries."""
    return (LITHUANIA_BOUNDS["min_lat"] <= lat <= LITHUANIA_BOUNDS["max_lat"] and 
            LITHUANIA_BOUNDS["min_lon"] <= lon <= LITHUANIA_BOUNDS["max_lon"])

def get_valid_coordinates(lat, lon, city="Vilnius"):
    """
    Validate coordinates are within Lithuania.
    Return valid coordinates or city center coordinates if invalid.
    """
    if is_in_lithuania(lat, lon):
        return lat, lon
    
    # If invalid, get city center coordinates
    city_coords = get_city_coordinates(city)
    if city_coords:
        return city_coords
    
    # Default to Vilnius if city coordinates not found
    return 54.687157, 25.279652

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_city_coordinates(city_name):
    """Get latitude and longitude for a Lithuanian city using geocoding API."""
    try:
        geolocator = Nominatim(user_agent="aruodas_price_predictor")
        location = geolocator.geocode(f"{city_name}, Lithuania", timeout=10)
        if location:
            return location.latitude, location.longitude
        return None
    except Exception as e:
        st.warning(f"Error getting coordinates for {city_name}: {str(e)}")
        return None
    
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_address_coordinates(address):
    """Get latitude, longitude and formatted address using geocoding API."""
    try:
        geolocator = Nominatim(user_agent="aruodas_price_predictor")
        if not address.lower().endswith("lithuania"):
            address = f"{address}, Lithuania"
            
        location = geolocator.geocode(address, timeout=10)
        if location:
            # Verify the coordinates are in Lithuania
            if is_in_lithuania(location.latitude, location.longitude):
                return location.latitude, location.longitude, location.address
            else:
                st.warning("The specified address is outside Lithuania. Please enter a location within Lithuania.")
                return None
        return None
    except Exception as e:
        st.warning(f"Error geocoding address: {str(e)}")
        return None

def display_map(latitude, longitude):
    """Create and display a folium map centered at the given coordinates."""
    # Create the map with Lithuania bounds as max bounds
    m = folium.Map(
        location=[latitude, longitude], 
        zoom_start=14,
        min_zoom=7,
        max_bounds=True,
        tiles='OpenStreetMap'
    )
    
    # Add bounds to restrict panning (Lithuania with some padding)
    sw = [LITHUANIA_BOUNDS["min_lat"] - 0.5, LITHUANIA_BOUNDS["min_lon"] - 0.5]
    ne = [LITHUANIA_BOUNDS["max_lat"] + 0.5, LITHUANIA_BOUNDS["max_lon"] + 0.5]
    m.fit_bounds([sw, ne])
    
    # Add a marker at the property location
    folium.Marker([latitude, longitude], popup="Property location", draggable=True).add_to(m)
    folium_static(m, width=800, height=500)

def location_input_section(city):
    """Create a location input section with address search and map display."""
    # Store location coordinates in session state if not already present
    if "latitude" not in st.session_state:
        st.session_state.latitude = None
    if "longitude" not in st.session_state:
        st.session_state.longitude = None
    if "formatted_address" not in st.session_state:
        st.session_state.formatted_address = None
    if "show_map" not in st.session_state:
        st.session_state.show_map = True  # Always show map by default
    
    # Always show address search field
    address_col1, address_col2 = st.columns([3, 1])
    
    with address_col1:
        address_input = st.text_input("Search address", 
                                      placeholder="Enter address (e.g., Gedimino pr. 1, Vilnius)",
                                      help="Enter a street address to locate on map")
    
    with address_col2:
        search_button = st.button("🔍 Search", help="Search for the entered address")
    
    # Get coordinates based on address when search is pressed
    if search_button and address_input:
        try:
            with st.spinner("Searching for address..."):
                # Use retry-decorated function for geocoding
                coords = get_address_coordinates(address_input)
                if coords:
                    latitude, longitude, formatted_address = coords
                    st.session_state.latitude = latitude
                    st.session_state.longitude = longitude
                    st.session_state.formatted_address = formatted_address
                    st.success(f"Found: {formatted_address}")
                else:
                    st.info("Address not found. Using city center coordinates.")
                    default_coords = get_city_coordinates(city)
                    if (default_coords and is_in_lithuania(*default_coords)):
                        st.session_state.latitude, st.session_state.longitude = default_coords
                        st.session_state.formatted_address = f"{city}, Lithuania"
                    else:
                        st.info("Using default Vilnius city center coordinates.")
                        st.session_state.latitude, st.session_state.longitude = 54.687157, 25.279652
                        st.session_state.formatted_address = "Vilnius, Lithuania"
        except Exception as e:
            st.error(f"Error processing address: {str(e)}")
            # Fall back to city center coordinates
            default_coords = get_city_coordinates(city)
            if (default_coords and is_in_lithuania(*default_coords)):
                st.session_state.latitude, st.session_state.longitude = default_coords
                st.session_state.formatted_address = f"{city}, Lithuania"
            else:
                st.session_state.latitude, st.session_state.longitude = 54.687157, 25.279652
                st.session_state.formatted_address = "Vilnius, Lithuania"
    
    # Use coordinates from session state if available, otherwise use city coordinates
    if st.session_state.latitude is not None and st.session_state.longitude is not None:
        lat = st.session_state.latitude
        lon = st.session_state.longitude
    else:
        # Get default coordinates for the selected city
        default_coords = get_city_coordinates(city)
        if (default_coords and is_in_lithuania(*default_coords)):
            lat, lon = default_coords
        else:
            # Default to Vilnius if city coordinates not found
            lat, lon = 54.687157, 25.279652
        # Store in session state
        st.session_state.latitude = lat
        st.session_state.longitude = lon
        st.session_state.formatted_address = f"{city}, Lithuania"
    
    # Display map
    map_container = st.container()
    with map_container:
        display_map(lat, lon)
    
    # Get coordinates from map
    col1, col2 = st.columns(2)
    with col1:
        latitude = st.number_input("Latitude", value=lat, format="%.6f", key="lat_input")
    with col2:
        longitude = st.number_input("Longitude", value=lon, format="%.6f", key="lon_input")
    
    # Validate coordinates are within Lithuania
    if not is_in_lithuania(latitude, longitude):
        st.warning("The specified coordinates are outside Lithuania. Coordinates must be within Lithuania's boundaries.")
        # Reset to valid coordinates
        valid_lat, valid_lon = get_valid_coordinates(lat, lon, city)
        latitude = valid_lat
        longitude = valid_lon
        # Update UI values (doesn't change the widget but ensures returned values are valid)
        
    # Update session state if coordinates changed
    if latitude != st.session_state.latitude:
        st.session_state.latitude = latitude
    if longitude != st.session_state.longitude:
        st.session_state.longitude = longitude
            
    # Reverse geocode when coordinates change manually
    if (latitude != lat or longitude != lon):
        with st.spinner("Updating address..."):
            try:
                geolocator = Nominatim(user_agent="aruodas_price_predictor")
                location = geolocator.reverse(f"{latitude}, {longitude}", timeout=10)
                if location:
                    st.session_state.formatted_address = location.address
                    st.info(f"Address: {location.address}")
                else:
                    st.info(f"No address found for coordinates ({latitude}, {longitude})")
            except Exception as e:
                st.warning(f"Could not retrieve address for coordinates: {str(e)}")
    
    return latitude, longitude

def distance_inputs():
    """Create input fields for distances to amenities."""
    st.subheader("Distances to Amenities")
    st.markdown("Enter distances to nearby amenities in kilometers")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        distance_to_darzeliai = st.number_input(
            "🎨 To kindergartens (km)",
            min_value=0.0,
            max_value=10.0,
            value=0.5,
            step=0.1,
            help="Distance to nearest kindergarten in kilometers"
        )
    
    with col2:
        distance_to_mokyklos = st.number_input(
            "🏫 To schools (km)",
            min_value=0.0,
            max_value=10.0,
            value=0.7,
            step=0.1,
            help="Distance to nearest school in kilometers"
        )
    
    with col3:
        distance_to_stoteles = st.number_input(
            "🚌 To public transport (km)",
            min_value=0.0,
            max_value=10.0,
            value=0.3,
            step=0.1,
            help="Distance to nearest public transport stop in kilometers"
        )
    
    with col4:
        distance_to_parduotuves = st.number_input(
            "🏪 To stores (km)",
            min_value=0.0,
            max_value=10.0,
            value=0.4,
            step=0.1,
            help="Distance to nearest store in kilometers"
        )
    
    return {
        "distance_to_darzeliai": distance_to_darzeliai,
        "distance_to_mokyklos": distance_to_mokyklos,
        "distance_to_stoteles": distance_to_stoteles,
        "distance_to_parduotuves": distance_to_parduotuves
    }