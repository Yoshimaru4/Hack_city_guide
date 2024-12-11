import os
import pandas as pd
from tqdm import tqdm

def filter_missing_coordinates(input_file, output_file):
    """
    Reads the input CSV, filters out rows without Latitude or Longitude,
    and writes the result to the output CSV.

    Parameters:
    - input_file (str): Path to the input CSV file.
    - output_file (str): Path to save the filtered CSV file.
    """
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Input file '{input_file}' does not exist.")
        return

    # Read the CSV file into a DataFrame
    print(f"Reading data from '{input_file}'...")
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
    except Exception as e:
        print(f"Error reading '{input_file}': {e}")
        return

    # Verify that 'Latitude' and 'Longitude' columns exist
    required_columns = ['Latitude', 'Longitude']
    for col in required_columns:
        if col not in df.columns:
            print(f"Column '{col}' not found in the input CSV.")
            return

    # Display initial data statistics
    total_records = df.shape[0]
    print(f"Total records before filtering: {total_records}")

    # Filter out rows where 'Latitude' or 'Longitude' is missing
    filtered_df = df.dropna(subset=['Latitude', 'Longitude'])

    # Display filtered data statistics
    filtered_records = filtered_df.shape[0]
    removed_records = total_records - filtered_records
    print(f"Total records after filtering: {filtered_records}")
    print(f"Number of records removed: {removed_records}")

    # Optionally, reset the index
    filtered_df.reset_index(drop=True, inplace=True)

    # Save the filtered DataFrame to the output CSV
    try:
        filtered_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Filtered data saved to '{output_file}'.")
    except Exception as e:
        print(f"Error saving '{output_file}': {e}")

def main():
    input_file = os.path.join("datasets", "details_with_coords.csv")
    output_file = os.path.join("datasets", "details_filtered.csv")

    filter_missing_coordinates(input_file, output_file)

if __name__ == "__main__":
    main()
