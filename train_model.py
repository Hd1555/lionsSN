import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load your dataset
data = pd.read_csv('play_by_play_2023.csv', low_memory=False)

# Filter for punt plays only
punt_data = data[data['play_type'] == 'punt']
print("Total rows in dataset:", len(data))
print("Rows after filtering for punts:", len(punt_data))

# Select relevant features, adding EPA to existing features
features = ['down', 'yardline_100', 'ydstogo', 'epa']
X = punt_data[features].copy()
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

    print("Model trained with EPA as an additional feature and saved.")
