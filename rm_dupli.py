import os
import pandas as pd
import shutil

# ---------------------------
# Backup Original Data
# ---------------------------
original_file = 'datasets/combined_details.csv'
backup_file = 'datasets/combined_details_backup.csv'
shutil.copyfile(original_file, backup_file)
print(f"Backup created at '{backup_file}'.")

# ---------------------------
# Load Dataset
# ---------------------------
df = pd.read_csv(original_file)
print(f"Original dataset loaded with {df.shape[0]} records.")

# ---------------------------
# Remove Exact Duplicates
# ---------------------------
df_cleaned = df.drop_duplicates(keep='first')
print(f"Dataset after removing exact duplicates: {df_cleaned.shape[0]} records.")

# ---------------------------
# Remove Partial Duplicates (Optional)
# ---------------------------
columns_to_check = ['Name']
df_cleaned_partial = df_cleaned.drop_duplicates(subset=columns_to_check, keep='first')
print(f"Dataset after removing partial duplicates based on {columns_to_check}: {df_cleaned_partial.shape[0]} records.")

# ---------------------------
# Save Cleaned Dataset
# ---------------------------
cleaned_file = 'datasets/combined_details_cleaned.csv'
df_cleaned_partial.to_csv(cleaned_file, index=False)
print(f"Cleaned dataset saved at '{cleaned_file}'.")
