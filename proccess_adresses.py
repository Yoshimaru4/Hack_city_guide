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

    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist.")
        return

    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file, encoding='utf-8')

    if 'Geo' not in df.columns:
        print("The input CSV does not contain a 'Geo' column.")
        return


    df['Geo'] = df['Geo'].astype(str)


    mask = ~df['Geo'].str.lower().str.startswith("москва, ")


    df.loc[mask, 'Geo'] = "Москва, " + df.loc[mask, 'Geo'].str.strip()

    print(f"Saving processed data to {output_file}...")
    df.to_csv(output_file, index=False, encoding='utf-8')

    print("Processing completed successfully.")

def main():
    input_file = os.path.join("datasets", "details.csv")
    output_file = os.path.join("datasets", "details_processed.csv") 

    prepend_moscow_to_geo(input_file, output_file)

if __name__ == "__main__":
    main()
