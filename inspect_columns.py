import pandas as pd

# Load the dataset
file_path = 'play_by_play_2023.csv'  # Adjust to the correct path of your dataset
try:
    data = pd.read_csv(file_path, low_memory=False)
    print(f"Dataset successfully loaded from '{file_path}'\n")
except FileNotFoundError:
    print(f"Error: File not found at '{file_path}'. Please verify the file path and try again.")
    exit()

# Display all columns available in the dataset
print("Columns available in the dataset:")
print(data.columns.tolist())

# Define required columns
required_columns = ['time_remaining', 'quarter', 'yardline_100', 'ydstogo', 'score_differential', 'wind']

# Check for missing required columns
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    print("\nWarning: The following required columns are missing in the dataset:")
    print(missing_columns)
else:
    print("\nAll required columns are present.")

# Provide an overview of each required column
print("\nInspecting required columns for data types and sample values:")
for col in required_columns:
    if col in data.columns:
        print(f"\nColumn: {col}")
        print(f"  Data type: {data[col].dtype}")
        print(f"  Sample values: {data[col].dropna().unique()[:5]}")
    else:
        print(f"\nColumn '{col}' is missing from the dataset.")

# Additional checks for derived columns
print("\nChecking if derived columns can be created:")
if 'game_seconds_remaining' in data.columns:
    print("  - 'game_seconds_remaining' is available. 'time_remaining' can be derived.")
else:
    print("  - 'game_seconds_remaining' is missing. 'time_remaining' cannot be derived.")

if 'qtr' in data.columns:
    print("  - 'qtr' is available. 'quarter' can be derived.")
else:
    print("  - 'qtr' is missing. 'quarter' cannot be derived.")

# Check and inspect rows related to punts
if 'play_type' in data.columns:
    punt_data = data[data['play_type'] == 'punt']
    print(f"\nSummary of rows with 'play_type' == 'punt':")
    print(f"  Total rows in dataset: {len(data)}")
    print(f"  Rows with 'play_type' == 'punt': {len(punt_data)}")
else:
    print("\nColumn 'play_type' is missing, unable to filter for punts.")

# Inspect wind data specifically
if 'wind' in data.columns:
    print("\nInspection of 'wind' column:")
    print(f"  Data type: {data['wind'].dtype}")
    print(f"  Unique values (up to 5): {data['wind'].dropna().unique()[:5]}")
    print(f"  Missing values: {data['wind'].isna().sum()}")
else:
    print("\nColumn 'wind' is missing from the dataset.")
