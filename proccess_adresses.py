import os
import pandas as pd
from tqdm import tqdm

def prepend_moscow_to_geo(input_file, output_file):
    """
    Reads the input CSV, prepends 'Москва ' to the 'Geo' column entries,
    and writes the result to the output CSV.

    Parameters:
    - input_file (str): Path to the input CSV file.
    - output_file (str): Path to save the processed CSV file.
    """
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist.")
        return

    # Read the CSV file into a DataFrame
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file, encoding='utf-8')

    # Verify that 'Geo' column exists
    if 'Geo' not in df.columns:
        print("The input CSV does not contain a 'Geo' column.")
        return

    # Ensure 'Geo' column is of string type
    df['Geo'] = df['Geo'].astype(str)

    # Create a boolean mask for entries that do NOT start with 'Москва '
    mask = ~df['Geo'].str.lower().str.startswith("москва, ")

    # Apply the mask to prepend 'Москва ' where necessary
    df.loc[mask, 'Geo'] = "Москва, " + df.loc[mask, 'Geo'].str.strip()

    # Save the processed DataFrame to the output CSV
    print(f"Saving processed data to {output_file}...")
    df.to_csv(output_file, index=False, encoding='utf-8')

    print("Processing completed successfully.")

def main():
    input_file = os.path.join("datasets", "details.csv")
    output_file = os.path.join("datasets", "details_processed.csv")  # Name as per your preference

    prepend_moscow_to_geo(input_file, output_file)

if __name__ == "__main__":
    main()
