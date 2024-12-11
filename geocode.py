import os
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from tqdm import tqdm

def geocode_address(address, geolocator, geocode_func):
    """
    Geocode the given address and return (latitude, longitude).
    Returns (None, None) if geocoding fails.
    """
    try:
        location = geocode_func(address)
        if location:
            return (location.latitude, location.longitude)
    except Exception as e:
        print(f"Error geocoding address '{address}': {e}")
    return (None, None)

def main():
    input_file = os.path.join("datasets", "details_processed.csv")
    output_file = os.path.join("datasets", "details_with_coords.csv")
    cache_file = os.path.join("datasets", "geocache.csv")  # Optional: For caching

    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found.")
        return

    # Read the combined CSV
    df = pd.read_csv(input_file, encoding='utf-8')

    # Initialize geolocator with rate limiter to respect usage policies
    geolocator = Nominatim(user_agent="moscow_scraper")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=3, error_wait_seconds=5.0)

    # Check if 'Geo' column exists
    if 'Geo' not in df.columns:
        print("The input CSV does not contain a 'Geo' column.")
        return

    # Initialize latitude and longitude columns
    df['Latitude'] = None
    df['Longitude'] = None

    # Iterate through the DataFrame and geocode each 'Geo' entry
    print("Geocoding 'Geo' addresses...")
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Geocoding Geo entries"):
        if pd.isna(row['Geo']) or row['Geo'].strip() == "":
            continue  # Skip empty Geo entries

        # Geocode the address
        lat, lon = geocode_address(row['Geo'], geolocator, geocode)
        df.at[idx, 'Latitude'] = lat
        df.at[idx, 'Longitude'] = lon

    # Save the enhanced DataFrame
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Enhanced data saved to {output_file}.")

if __name__ == "__main__":
    main()