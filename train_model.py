import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load your dataset
file_path = 'play_by_play_2023.csv'  # Update with your actual file path
data = pd.read_csv(file_path, low_memory=False)

# Filter for punt plays and create a true copy to avoid SettingWithCopyWarning
punt_data = data[data['play_type'] == 'punt'].copy()

# Derive 'time_remaining' and 'quarter'
if 'game_seconds_remaining' in punt_data.columns:
    punt_data['time_remaining'] = punt_data['game_seconds_remaining']
else:
    raise KeyError("'game_seconds_remaining' is missing, unable to derive 'time_remaining'.")

if 'qtr' in punt_data.columns:
    punt_data['quarter'] = punt_data['qtr']
else:
    raise KeyError("'qtr' is missing, unable to derive 'quarter'.")

# Select relevant features
features = ['yardline_100', 'ydstogo', 'time_remaining', 'score_differential', 'wind', 'quarter']
X = punt_data[features]
y = punt_data['success']  # Replace 'success' with your chosen target column

# Check for NaN values
print("NaN counts per feature column:\n", X.isna().sum())
print("NaN count in target (y):", y.isna().sum())

# Drop rows with NaN in the target column
X = X[~y.isna()]
y = y.dropna()

# If data is empty after dropping NaNs, output a message and exit
if len(X) == 0:
    print("No data left after filtering. Please check 'success' column for more details.")
else:
    # Handle missing data in features
    X.fillna(0, inplace=True)

    # Train the model
    model = RandomForestRegressor()
    model.fit(X, y)

    # Save the trained model
    with open('trick_play_model.pkl', 'wb') as file:
        pickle.dump(model, file)

    print("Model trained and saved.")
