# app.py

import os
import pandas as pd
import streamlit as st
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from streamlit_folium import st_folium
import folium
from folium.plugins import MarkerCluster

# ---------------------------
# Streamlit Page Configuration
# ---------------------------
st.set_page_config(
    page_title="🏠 Moscow Nearby Items Finder",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# Title and Description
# ---------------------------
st.title("🏠 Moscow Nearby Items Finder")
st.markdown("""
    Enter your location and specify a radius to find nearby houses, places, or monuments within the specified distance.
""")

# ---------------------------
# Sidebar for User Inputs
# ---------------------------
st.sidebar.header("🔍 User Inputs")

# ---------------------------
# Function to Load Data
# ---------------------------
@st.cache_data
def load_data(filepath):
    if not os.path.exists(filepath):
        st.error(f"Dataset not found at {filepath}. Please ensure the file exists.")
        return pd.DataFrame()
    df = pd.read_csv(filepath, encoding='utf-8')
    # Assign an ID based on the DataFrame index
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'ID'}, inplace=True)
    return df

# Load the dataset
data_path = os.path.join("datasets", "details_filtered.csv")
df = load_data(data_path)

if df.empty:
    st.stop()

# ---------------------------
# Initialize Session State
# ---------------------------
if 'user_location' not in st.session_state:
    st.session_state.user_location = None  # Tuple of (lat, lon)

if 'search_results' not in st.session_state:
    st.session_state.search_results = None  # DataFrame of search results

# ---------------------------
# User Input for Location
# ---------------------------
location_input = st.sidebar.text_input(
    "📍 Enter your address in Moscow:",
    placeholder="e.g., Красная площадь, Москва"
)

# Optionally, allow users to select location on a map
use_map = st.sidebar.checkbox("🗺️ Select location on map", value=False)

if use_map:
    # Initialize map centered at Moscow
    m = folium.Map(location=[55.7558, 37.6173], zoom_start=10)
    
    # Add LatLngPopup to capture clicks
    folium.LatLngPopup().add_to(m)
    
    st.sidebar.markdown("### 📍 Click on the map to select your location:")
    with st.sidebar:
        selected_location = st_folium(m, width=280, height=400)
    
    if selected_location and 'last_clicked' in selected_location and selected_location['last_clicked']:
        user_lat = selected_location['last_clicked']['lat']
        user_lon = selected_location['last_clicked']['lng']
        st.session_state.user_location = (user_lat, user_lon)
    elif use_map and st.session_state.user_location is None:
        st.session_state.user_location = None  # Reset if no selection

# ---------------------------
# User Input for Distance
# ---------------------------
distance_input = st.sidebar.number_input(
    "📏 Search radius (meters):",
    min_value=100,
    max_value=5000,
    value=100,
    step=100
)

# ---------------------------
# Button to Trigger Search
# ---------------------------
search_button = st.sidebar.button("🔎 Find Nearby Items")

# ---------------------------
# Initialize Geolocator
# ---------------------------
geolocator = Nominatim(user_agent="streamlit_app")

# ---------------------------
# Function to Geocode Address
# ---------------------------
@st.cache_data
def geocode_address(address):
    try:
        location = geolocator.geocode(address)
        if location:
            return (location.latitude, location.longitude)
    except Exception as e:
        st.error(f"Error geocoding address: {e}")
    return (None, None)

# ---------------------------
# Handle Search Button Click
# ---------------------------
if search_button:
    with st.spinner("🔄 Processing..."):
        # Determine user location
        if use_map:
            if st.session_state.user_location is None:
                st.error("❗ Please click on the map to select your location.")
                st.session_state.search_results = None  # Clear previous results
            else:
                user_lat, user_lon = st.session_state.user_location
        else:
            if not location_input.strip():
                st.error("❗ Please enter a valid address or select a location on the map.")
                st.session_state.search_results = None  # Clear previous results
            else:
                user_lat, user_lon = geocode_address(location_input)
                if user_lat is None or user_lon is None:
                    st.error("❗ Unable to geocode the provided address. Please try a different one.")
                    st.session_state.search_results = None  # Clear previous results
                else:
                    st.session_state.user_location = (user_lat, user_lon)  # Update session state
        
        # Proceed only if user location is valid
        if st.session_state.user_location:
            user_location = st.session_state.user_location
            
            # Calculate distances
            # Vectorized distance calculation using apply (can be optimized further)
            df['Distance_m'] = df.apply(
                lambda row: geodesic(user_location, (row['Latitude'], row['Longitude'])).meters,
                axis=1
            )
            
            # Filter based on distance
            filtered_df = df[df['Distance_m'] <= distance_input].copy()
            
            if filtered_df.empty:
                st.warning("🚫 No items found within the specified radius.")
                st.session_state.search_results = None  # Clear previous results
            else:
                # Store the filtered results in session state
                st.session_state.search_results = filtered_df

# ---------------------------
# Display Search Results
# ---------------------------
if st.session_state.search_results is not None:
    filtered_df = st.session_state.search_results
    
    st.subheader(f"📌 Items within {distance_input} meters of your location:")
    st.write(f"**Total items found:** {filtered_df.shape[0]}")
    
    # Display as a table with clickable links
    def make_clickable(link):
        return f'<a href="{link}" target="_blank">🔗 Link</a>'
    
    if 'SourceLink' in filtered_df.columns:
        filtered_df['SourceLink'] = filtered_df['SourceLink'].apply(make_clickable)
        st.markdown(filtered_df[['ID', 'Category', 'Name', 'Geo', 'Distance_m', 'SourceLink']].to_html(escape=False, index=False), unsafe_allow_html=True)
    else:
        st.table(filtered_df[['ID', 'Category', 'Name', 'Geo', 'Distance_m']])
    
    # Display on a map
    st.subheader("🗺️ Map of Nearby Items:")
    folium_map = folium.Map(location=st.session_state.user_location, zoom_start=15)
    
    # Add a marker for user location
    folium.Marker(
        location=st.session_state.user_location,
        popup="📍 Your Location",
        icon=folium.Icon(color='red', icon='user')
    ).add_to(folium_map)
    
    # Initialize marker cluster
    marker_cluster = MarkerCluster().add_to(folium_map)
    
    # Add markers for nearby items
    for _, row in filtered_df.iterrows():
        folium.Marker(
            location=(row['Latitude'], row['Longitude']),
            popup=f"{row['Name']} (ID: {row['ID']})<br>Distance: {row['Distance_m']:.2f} meters",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(marker_cluster)
    
    # Render the map
    st_folium(folium_map, width=700, height=500)
