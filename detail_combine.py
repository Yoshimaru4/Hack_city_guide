import os
import pandas as pd

def combine_csv_files(input_folder, output_file):
    """
    Combines all CSV files in the input_folder into a single CSV file.

    Parameters:
    - input_folder (str): Path to the folder containing CSV files to combine.
    - output_file (str): Path to the output combined CSV file.
    """
    # List to hold DataFrames
    df_list = []

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_folder, filename)
            print(f"Reading {file_path}...")
            # Read CSV file into DataFrame
            df = pd.read_csv(file_path, encoding='utf-8')
            df_list.append(df)

    if not df_list:
        print("No CSV files found in the specified folder.")
        return

    # Concatenate all DataFrames
    combined_df = pd.concat(df_list, ignore_index=True)

    # Optionally, remove duplicates if any
    #combined_df.drop_duplicates(inplace=True)

    # Save the combined DataFrame to a new CSV file
    combined_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"All files combined into {output_file} successfully.")

def main():
    input_folder = os.path.join("datasets", "details_data")  # Path to your details_data folder
    output_file = os.path.join("datasets", "combined_details.csv")  # Desired output file path

    combine_csv_files(input_folder, output_file)

if __name__ == "__main__":
    main()
