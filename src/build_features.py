from pathlib import Path
import pandas as pd

# Assuming this file lives in src/ and project root is one level up
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "flights_sample_3m.csv"

df = pd.read_csv(DATA_PATH)

print(df.head())

df["dep_hour"] = df["CRS_DEP_TIME"] // 100

# Extract temporal features from FL_DATE
df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
df['month'] = df['FL_DATE'].dt.month
df['day_of_week'] = df['FL_DATE'].dt.dayofweek  # 0=Monday, 6=Sunday

# Frequency Encoding (can be done before train/test split since it doesn't use target)
# Count how often each airport appears
df["origin_freq"] = df["ORIGIN"].map(df["ORIGIN"].value_counts())
df["dest_freq"] = df["DEST"].map(df["DEST"].value_counts())

# Get proportions
df["origin_freq_proportion"] = df["ORIGIN"].map(df["ORIGIN"].value_counts(normalize=True))
df["dest_freq_proportion"] = df["DEST"].map(df["DEST"].value_counts(normalize=True))

# Drop leakage columns (actual times/delays that happen after scheduled departure)
leakage_cols = [
    'DEP_TIME', 'DEP_DELAY', 'TAXI_OUT', 'WHEELS_OFF', 'WHEELS_ON',
    'TAXI_IN', 'ARR_TIME', 'ARR_DELAY', 'ELAPSED_TIME', 'AIR_TIME',
    'DELAY_DUE_CARRIER', 'DELAY_DUE_LATE_AIRCRAFT', 'DELAY_DUE_NAS',
    'DELAY_DUE_SECURITY', 'DELAY_DUE_WEATHER', 'DIVERTED', 'DOT_CODE', 'FL_NUMBER'
]
df = df.drop(columns=[col for col in leakage_cols if col in df.columns])

# Path to processed data folder and file
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_PATH = PROCESSED_DIR / "flights_features.csv"

# Save processed dataset (without index)
df.to_csv(PROCESSED_PATH, index=False)

print(f"Saved processed features to: {PROCESSED_PATH}")