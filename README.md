# Flight Cancellation Prediction ML Project

A machine learning project to predict flight cancellations using Random Forest classification.

On 3rd January 2026, my flight was cancelled on the day of departure, with no reason for cancellation provided. The several hours after being informed of the cancellation was spent frantically figuring out my options for rebooking a new flight to ensure my following plans would be affected as little as possible. That was the inspiration for this project.

Over the summer, I had self-learned the fundamental math behind machine learning, but never found the time to work on a hands-on project. This being my first machine learning project meant that it was important for me to keep it to a manageable scale. Although I did not have much interaction with the fundamental math that I learned in the code for this project, it was nice to be able to appreciate/guess what was going on under the hood while operating many layers of abstractions up using these libraries.

## Key learnings

- The **structure** of ML projects: using Python **Notebooks** for exploratory data analysis, figuring out what features to use, testing different models, and deciding on thresholds. Building the production pipeline for deployment only after.
- **Decision-making** in the features to be used, taking note of leakage features, checking if grouping or log normalization is necessary, different types of encodings of features.
- Important **metrics** in studying the efficacy of different models: confusion matrix, recall, precision, ROC-AUC, PR-AUC. The **math** behind them.
- How **real-life/business considerations** play into metrics with more importance placed on in the choice of **thresholds**, which out of TP, FP, TN, FN are to be minimized/maximized.
- Different models to be considered (in this binary classification problem): logisitc regression, random forest, XGBoost, LightGBM. The intuition and math behind **decision trees** and **random forests**, and the basics of gradient boosting.
- The structure of using _.pkl_ files in the backend, and the pipeline of how the frontend would interact with the predictions.

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
   - Enter scheduled departure/arrival times (HHMM format, e.g., 0800 for 8:00 AM)
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
- Front end was completely vibe-coded with GPT-5.2 on Cursor, given that it is not a priority of this project.
