# inspect_columns.py
import pandas as pd

data = pd.read_csv('play_by_play_2023.csv', low_memory=False)
print("Columns in dataset:")
print(data.columns.tolist())

