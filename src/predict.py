from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# --------------------
# Paths & loading
# --------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "random_forest.pkl"
PREPROC_PATH = MODELS_DIR / "preprocessing.pkl"

# This CSV should look like the original raw data (same columns as flights_sample_3m.csv)
NEW_DATA_PATH = PROJECT_ROOT / "data" / "new_flights.csv"

rf_model = joblib.load(MODEL_PATH)
preproc = joblib.load(PREPROC_PATH)

feature_names = preproc["feature_names"]
airline_target_means = preproc["airline_target_means"]
global_mean = preproc["global_mean"]

# --------------------
# Preprocess new data
# --------------------
def preprocess_new_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # 1) Feature engineering (same as build_features.py)
    df["dep_hour"] = df["CRS_DEP_TIME"] // 100

    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"])
    df["month"] = df["FL_DATE"].dt.month
    df["day_of_week"] = df["FL_DATE"].dt.dayofweek

    origin_freq_map = preproc["origin_freq_map"]
    dest_freq_map = preproc["dest_freq_map"]
    total_flights = preproc["total_flights"]

    df["origin_freq"] = df["ORIGIN"].map(origin_freq_map).fillna(0)
    df["dest_freq"] = df["DEST"].map(dest_freq_map).fillna(0)
    df["origin_freq_proportion"] = df["origin_freq"] / total_flights
    df["dest_freq_proportion"] = df["dest_freq"] / total_flights

    leakage_cols = [
        "DEP_TIME", "DEP_DELAY", "TAXI_OUT", "WHEELS_OFF", "WHEELS_ON",
        "TAXI_IN", "ARR_TIME", "ARR_DELAY", "ELAPSED_TIME", "AIR_TIME",
        "DELAY_DUE_CARRIER", "DELAY_DUE_LATE_AIRCRAFT", "DELAY_DUE_NAS",
        "DELAY_DUE_SECURITY", "DELAY_DUE_WEATHER", "DIVERTED", "DOT_CODE",
        "FL_NUMBER",
    ]
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns], errors="ignore")

    # 2) Target encoding for AIRLINE using training means
    if "AIRLINE" in df.columns:
        df["airline_target_encoded"] = df["AIRLINE"].map(airline_target_means)
        df["airline_target_encoded"] = df["airline_target_encoded"].fillna(global_mean)
        df = df.drop(columns=["AIRLINE"])

    # 3) Drop remaining unencoded categoricals
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if categorical_cols:
        df = df.drop(columns=categorical_cols)

    # 4) Drop raw freq counts if you only trained on proportions
    for col in ["origin_freq", "dest_freq"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # 5) Keep only numeric, fill NaNs
    df = df.select_dtypes(include=[np.number])
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median())

    # 6) Reorder columns to match training
    # (any missing columns will be added as 0; any extra will be dropped)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0
    df = df[feature_names]

    return df

# --------------------
# Run prediction
# --------------------
if __name__ == "__main__":
    # Load new data (one or many rows)
    df_new_raw = pd.read_csv(NEW_DATA_PATH)

    X_new = preprocess_new_data(df_new_raw)

    # Predictions
    y_proba = rf_model.predict_proba(X_new)[:, 1]
    y_pred = rf_model.predict(X_new)

    # Attach to original rows for inspection
    results = df_new_raw.copy()
    results["pred_cancel_prob"] = y_proba
    results["pred_cancel_label"] = y_pred

    print(results[["pred_cancel_prob", "pred_cancel_label"]].head())