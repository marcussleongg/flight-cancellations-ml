from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "flights_features.csv"

df = pd.read_csv(PROCESSED_PATH)

X = df.drop(columns=["CANCELLED"])  # Features
y = df["CANCELLED"]  # Target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_processed = X_train.copy()
X_test_processed = X_test.copy()

# Target Encoding for AIRLINE (using only training data to avoid leakage)
# Calculate mean cancellation rate per airline from TRAINING data only
if 'AIRLINE' in X_train_processed.columns:
    # Create a temporary dataframe with aligned indices
    temp_df = pd.DataFrame({
        'AIRLINE': X_train_processed['AIRLINE'],
        'CANCELLED': y_train.values
    }, index=X_train_processed.index)
    
    airline_target_means = temp_df.groupby('AIRLINE')['CANCELLED'].mean()
    
    # Apply to both train and test
    X_train_processed['airline_target_encoded'] = X_train_processed['AIRLINE'].map(airline_target_means)
    X_test_processed['airline_target_encoded'] = X_test_processed['AIRLINE'].map(airline_target_means)
    
    # Fill any unseen airlines in test with global training mean
    global_mean = y_train.mean()
    X_test_processed['airline_target_encoded'] = X_test_processed['airline_target_encoded'].fillna(global_mean)
    
    # Also fill any NaN in train (shouldn't happen, but just in case)
    X_train_processed['airline_target_encoded'] = X_train_processed['airline_target_encoded'].fillna(global_mean)
    
    # Drop original AIRLINE column (we have the encoded version)
    X_train_processed = X_train_processed.drop(columns=['AIRLINE'])
    X_test_processed = X_test_processed.drop(columns=['AIRLINE'])

# Select final feature set: use frequency-encoded and target-encoded features
# Drop remaining categorical columns that aren't encoded yet
categorical_cols = X_train_processed.select_dtypes(include=['object']).columns.tolist()
if categorical_cols:
    print(f"Dropping unencoded categorical columns: {categorical_cols}")
    X_train_processed = X_train_processed.drop(columns=categorical_cols)
    X_test_processed = X_test_processed.drop(columns=categorical_cols)

# Drop any columns that shouldn't be used (like CANCELLATION_CODE - only exists for cancelled flights)
columns_to_drop = ['CANCELLATION_CODE'] if 'CANCELLATION_CODE' in X_train_processed.columns else []
if columns_to_drop:
    X_train_processed = X_train_processed.drop(columns=columns_to_drop)
    X_test_processed = X_test_processed.drop(columns=columns_to_drop)

X_train_processed = X_train_processed.drop(columns=['origin_freq', 'dest_freq'])
X_test_processed = X_test_processed.drop(columns=['origin_freq', 'dest_freq'])

# Fill any remaining NaN values with median (for numerical) or 0
# Only fill numeric columns
numeric_cols = X_train_processed.select_dtypes(include=[np.number]).columns
X_train_processed[numeric_cols] = X_train_processed[numeric_cols].fillna(X_train_processed[numeric_cols].median())
X_test_processed[numeric_cols] = X_test_processed[numeric_cols].fillna(X_train_processed[numeric_cols].median())

# Ensure all columns are numeric (convert any remaining non-numeric to numeric)
for col in X_train_processed.columns:
    if X_train_processed[col].dtype == 'object':
        # Try to convert to numeric
        X_train_processed[col] = pd.to_numeric(X_train_processed[col], errors='coerce')
        X_test_processed[col] = pd.to_numeric(X_test_processed[col], errors='coerce')

# Final check: ensure no NaN or infinite values
X_train_processed = X_train_processed.replace([np.inf, -np.inf], np.nan)
X_test_processed = X_test_processed.replace([np.inf, -np.inf], np.nan)
X_train_processed = X_train_processed.fillna(X_train_processed.median())
X_test_processed = X_test_processed.fillna(X_train_processed.median())

# Convert to numpy arrays for sklearn (or keep as DataFrames - sklearn accepts both)
# But ensure all are numeric
X_train_processed = X_train_processed.select_dtypes(include=[np.number])
X_test_processed = X_test_processed.select_dtypes(include=[np.number])

# Using a subset for faster training - can adjust n_estimators and remove sampling for full training
rf_model = RandomForestClassifier(
    n_estimators=100,  # Reduce for faster training; increase for better performance
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    max_depth=20,  # Limit depth for faster training
    min_samples_split=100,
    min_samples_leaf=50
)

rf_model.fit(X_train_processed, y_train)

# Predictions
rf_pred = rf_model.predict(X_test_processed)
rf_proba = rf_model.predict_proba(X_test_processed)[:, 1]

# Helper function to evaluate models
def evaluate_model(y_true, y_pred, y_proba, model_name):
    """Evaluate model performance and return metrics dictionary"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    
    results = {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'ROC-AUC': roc_auc,
        'PR-AUC': pr_auc
    }
    
    return results

# Evaluate
rf_results = evaluate_model(y_test, rf_pred, rf_proba, 'Random Forest')
print("\nRandom Forest Performance:")
for metric, value in rf_results.items():
    if metric != 'Model':
        print(f"  {metric}: {value:.4f}")

from pathlib import Path
import joblib

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

model_path = MODELS_DIR / "random_forest.pkl"
joblib.dump(rf_model, model_path)

# Save metadata needed for prediction
preproc_path = MODELS_DIR / "preprocessing.pkl"
preproc_artifacts = {
    "feature_names": X_train_processed.columns.tolist(),
    "airline_target_means": airline_target_means.to_dict(),
    "global_mean": float(global_mean),
    "origin_freq_map": df["ORIGIN"].value_counts().to_dict(),  # from full training data
    "dest_freq_map": df["DEST"].value_counts().to_dict(),
    "total_flights": len(df),  # for computing proportions
    # For inference when the user only provides ORIGIN/DEST:
    # Use typical route distance from historical data
    "route_distance_map": (
        df.groupby(["ORIGIN", "DEST"])["DISTANCE"].median().to_dict()
        if all(c in df.columns for c in ["ORIGIN", "DEST", "DISTANCE"])
        else {}
    ),
    "global_median_distance": float(df["DISTANCE"].median()) if "DISTANCE" in df.columns else None,
    "global_median_crs_elapsed": float(df["CRS_ELAPSED_TIME"].median()) if "CRS_ELAPSED_TIME" in df.columns else None,
}
joblib.dump(preproc_artifacts, preproc_path)

print(f"Saved model to: {model_path}")
print(f"Saved preprocessing artifacts to: {preproc_path}")