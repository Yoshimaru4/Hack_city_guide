# app.py

import os
import pandas as pd
import streamlit as st
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from streamlit_folium import st_folium
import folium
from folium.plugins import MarkerCluster


st.set_page_config(
    page_title="🏠 Moscow Nearby Items Finder",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.title("🏠 Moscow Nearby Items Finder")
st.markdown("""
    Enter your location and specify a radius to find nearby houses, places, or monuments within the specified distance.
""")


st.sidebar.header("🔍 User Inputs")


@st.cache_data
def load_data(filepath):
    if not os.path.exists(filepath):
        st.error(f"Dataset not found at {filepath}. Please ensure the file exists.")
        return pd.DataFrame()
    df = pd.read_csv(filepath, encoding='utf-8')

    df.reset_index(inplace=True)
    df.rename(columns={'index': 'ID'}, inplace=True)
    return df


data_path = os.path.join("datasets", "details_filtered.csv")
df = load_data(data_path)

if df.empty:
    st.stop()


if 'user_location' not in st.session_state:
    st.session_state.user_location = None  

if 'search_results' not in st.session_state:
    st.session_state.search_results = None  


location_input = st.sidebar.text_input(
    "📍 Enter your address in Moscow:",
    placeholder="e.g., Красная площадь, Москва"
)


use_map = st.sidebar.checkbox("🗺️ Select location on map", value=False)

if use_map:

    m = folium.Map(location=[55.7558, 37.6173], zoom_start=10)
    

    folium.LatLngPopup().add_to(m)
    
    st.sidebar.markdown("### 📍 Click on the map to select your location:")
    with st.sidebar:
        selected_location = st_folium(m, width=280, height=400)
    
    if selected_location and 'last_clicked' in selected_location and selected_location['last_clicked']:
        user_lat = selected_location['last_clicked']['lat']
        user_lon = selected_location['last_clicked']['lng']
        st.session_state.user_location = (user_lat, user_lon)
    elif use_map and st.session_state.user_location is None:
        st.session_state.user_location = None  


distance_input = st.sidebar.number_input(
    "📏 Search radius (meters):",
    min_value=100,
    max_value=5000,
    value=100,
    step=100
)


search_button = st.sidebar.button("🔎 Find Nearby Items")


geolocator = Nominatim(user_agent="streamlit_app")


@st.cache_data
def geocode_address(address):
    try:
        location = geolocator.geocode(address)
        if location:
            return (location.latitude, location.longitude)
    except Exception as e:
        st.error(f"Error geocoding address: {e}")
    return (None, None)


if search_button:
    with st.spinner("🔄 Processing..."):

        if use_map:
            if st.session_state.user_location is None:
                st.error("❗ Please click on the map to select your location.")
                st.session_state.search_results = None  
            else:
                user_lat, user_lon = st.session_state.user_location
        else:
            if not location_input.strip():
                st.error("❗ Please enter a valid address or select a location on the map.")
                st.session_state.search_results = None  
            else:
                user_lat, user_lon = geocode_address(location_input)
                if user_lat is None or user_lon is None:
                    st.error("❗ Unable to geocode the provided address. Please try a different one.")
                    st.session_state.search_results = None  
                else:
                    st.session_state.user_location = (user_lat, user_lon)  
        

        if st.session_state.user_location:
            user_location = st.session_state.user_location
            

            df['Distance_m'] = df.apply(
                lambda row: geodesic(user_location, (row['Latitude'], row['Longitude'])).meters,
                axis=1
            )
            
            filtered_df = df[df['Distance_m'] <= distance_input].copy()
            
            if filtered_df.empty:
                st.warning("🚫 No items found within the specified radius.")
                st.session_state.search_results = None  
            else:
                st.session_state.search_results = filtered_df


if st.session_state.search_results is not None:
    filtered_df = st.session_state.search_results
    
    st.subheader(f"📌 Items within {distance_input} meters of your location:")
    st.write(f"**Total items found:** {filtered_df.shape[0]}")
    
    def make_clickable(link):
        return f'<a href="{link}" target="_blank">🔗 Link</a>'
    
    if 'SourceLink' in filtered_df.columns:
        filtered_df['SourceLink'] = filtered_df['SourceLink'].apply(make_clickable)
        st.markdown(filtered_df[['ID', 'Category', 'Name', 'Geo', 'Distance_m', 'SourceLink']].to_html(escape=False, index=False), unsafe_allow_html=True)
    else:
        st.table(filtered_df[['ID', 'Category', 'Name', 'Geo', 'Distance_m']])
    
    st.subheader("🗺️ Map of Nearby Items:")
    folium_map = folium.Map(location=st.session_state.user_location, zoom_start=15)
    
    folium.Marker(
        location=st.session_state.user_location,
        popup="📍 Your Location",
        icon=folium.Icon(color='red', icon='user')
    ).add_to(folium_map)
    
    marker_cluster = MarkerCluster().add_to(folium_map)
    
    for _, row in filtered_df.iterrows():
        folium.Marker(
            location=(row['Latitude'], row['Longitude']),
            popup=f"{row['Name']} (ID: {row['ID']})<br>Distance: {row['Distance_m']:.2f} meters",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(marker_cluster)
    
    st_folium(folium_map, width=700, height=500)
