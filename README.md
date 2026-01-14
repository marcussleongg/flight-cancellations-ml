# Flight Cancellation Prediction ML Project

A machine learning project to predict flight cancellations using Random Forest classification.

## Project Structure

```
flight-cancellations-ml/
├── data/
│   ├── flights_sample_3m.csv          # Raw data
│   └── processed/
│       └── flights_features.csv       # Processed features
├── models/
│   ├── random_forest.pkl              # Trained model
│   └── preprocessing.pkl              # Preprocessing artifacts
├── notebooks/
│   ├── eda.ipynb                      # Exploratory data analysis
│   ├── feature_engineering.ipynb      # Feature engineering experiments
│   ├── modeling.ipynb                 # Model training & comparison
│   └── evaluation.ipynb                # Model evaluation & threshold selection
└── src/
    ├── build_features.py               # Feature engineering pipeline
    ├── train_model.py                  # Model training script
    ├── predict.py                      # Prediction logic
    ├── app.py                          # Flask web API
    └── templates/
        └── index.html                  # Frontend web interface
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run feature engineering:**
   ```bash
   python src/build_features.py
   ```

3. **Train the model:**
   ```bash
   python src/train_model.py
   ```

## Running the Web Application

1. **Start the Flask server:**
   ```bash
   python src/app.py
   ```

2. **Open your browser:**
   Navigate to `http://localhost:5000`

3. **Fill out the form:**
   - Select airline
   - Enter flight date
   - Enter origin and destination airport codes (3-letter codes like JFK, LAX)
   - Enter scheduled departure/arrival times (HHMM format, e.g., 800 for 8:00 AM)
   - Enter distance in miles and scheduled elapsed time in minutes

4. **Click "Predict Cancellation"** to get the prediction

## API Usage

You can also use the API directly:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "AIRLINE": "United Air Lines Inc.",
    "FL_DATE": "2024-01-15",
    "ORIGIN": "JFK",
    "DEST": "LAX",
    "CRS_DEP_TIME": 800,
    "CRS_ARR_TIME": 1200,
    "DISTANCE": 2475,
    "CRS_ELAPSED_TIME": 360
  }'
```

Response:
```json
{
  "success": true,
  "prediction": 0,
  "probability": 0.0234,
  "message": "Flight will not be cancelled"
}
```

## Model Details

- **Algorithm:** Random Forest Classifier
- **Key Features:**
  - Temporal features (hour, month, day of week)
  - Frequency encoding for airports
  - Target encoding for airlines
  - Distance and scheduled times

## Notes

- Make sure `models/random_forest.pkl` and `models/preprocessing.pkl` exist before running the web app
- The model expects specific airline names (see dropdown in frontend)
- Airport codes should be 3-letter IATA codes (e.g., JFK, LAX, ORD)
