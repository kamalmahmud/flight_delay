# Flight Delay Prediction

A machine learning project that predicts whether a scheduled flight will arrive **30 minutes or more late** by combining U.S. flight schedule data with ASOS weather observations at the origin and destination airports.

The project is implemented as a Jupyter Notebook: `Flight_Delay_Prediction.ipynb`.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Main Goal](#main-goal)
- [Dataset Inputs](#dataset-inputs)
- [How the Pipeline Works](#how-the-pipeline-works)
- [Features Used](#features-used)
- [Models Trained](#models-trained)
- [Evaluation Results](#evaluation-results)
- [Final Saved Files](#final-saved-files)
- [How to Run the Project](#how-to-run-the-project)
- [How to Use the Saved Model](#how-to-use-the-saved-model)
- [Project Structure](#project-structure)
- [Important Settings](#important-settings)
- [Data Leakage Prevention](#data-leakage-prevention)
- [Limitations](#limitations)
- [Possible Improvements](#possible-improvements)
- [Conclusion](#conclusion)

---

## Project Overview

This project builds a binary classification model for flight delay prediction.

The notebook:

- Loads U.S. Bureau of Transportation Statistics on-time flight performance data from 2019 through 2026.
- Loads ASOS historical weather observations from 2019 through 2026.
- Converts local scheduled departure and arrival times to UTC.
- Matches each flight with the most recent weather observation before:
  - scheduled departure at the origin airport;
  - scheduled arrival at the destination airport.
- Creates time, route, airport, airline, and weather-based features.
- Trains and compares:
  - Random Forest Classifier;
  - Neural Network using `MLPClassifier`.
- Selects the best model using average precision first, then ROC AUC.
- Retrains the selected model on all prepared modeling data.
- Saves the final model, threshold, and prepared dataset.

---

## Main Goal

The goal is to answer this question:

> Given a flight schedule, route information, airline, and recent weather conditions, will the flight arrive at least 30 minutes late?

The target variable is:

```text
Delayed = 1 if ArrDelayMinutes >= 30
Delayed = 0 otherwise
```

---

## Dataset Inputs

The notebook expects the following Kaggle datasets.

### Flight Data

Flight files are loaded from:

```text
/kaggle/input/datasets/kamalalqedra/bts-ontime-performance-2019-present-csv/
```

Expected files:

```text
bts_ontime_2019.csv
bts_ontime_2020.csv
bts_ontime_2021.csv
bts_ontime_2022.csv
bts_ontime_2023.csv
bts_ontime_2024.csv
bts_ontime_2025.csv
bts_ontime_2026.csv
```

Main flight columns used:

```text
FlightDate
Reporting_Airline
Origin
OriginState
Dest
DestState
CRSDepTime
CRSArrTime
CRSElapsedTime
Distance
ArrDelayMinutes
```

### Weather Data

Weather data is loaded from:

```text
/kaggle/input/datasets/sehamhakimothman/asos-weather-data-2019-2026/historical_weather_data_2019_2026.csv
```

Main weather columns used:

```text
station
valid
tmpf
dwpf
relh
drct
sknt
p01i
alti
mslp
vsby
gust
skyc1
skyc2
wxcodes
feel
```

---

## How the Pipeline Works

### 1. Settings and Imports

The notebook imports the required Python libraries, defines model settings, file paths, and constants.

Key settings include:

```python
DELAY_THRESHOLD_MINUTES = 30
WEATHER_TOLERANCE = "3h"
PREDICTION_THRESHOLD = 0.50
TEST_SPLIT_DATE = "2025-01-01"
MIN_ROUTE_COUNT = 100
RANDOM_STATE = 42
MLP_MAX_ITER = 60
```

### 2. Input File Validation

Before processing, the notebook checks that all flight and weather files exist. If any file is missing, it raises a `FileNotFoundError`.

### 3. Flight Data Loading

The notebook loads all selected flight CSV files, keeps only the needed columns, standardizes column names, converts date and numeric fields, and combines all files into one dataframe.

In the executed notebook, the raw loaded flight data contained:

| Item | Value |
|---|---:|
| Flight files loaded | 8 |
| Flight rows | 46,307,495 |
| First flight date | 2019-01-01 |
| Last flight date | 2026-01-31 |
| Memory usage | 1,589.91 MB |

### 4. Target Creation

The target column is created from `ArrDelayMinutes`.

Executed target distribution:

| Class | Count | Percentage |
|---|---:|---:|
| Not delayed | 39,602,878 | 87.65% |
| Delayed | 5,580,516 | 12.35% |

This shows that delayed flights are the minority class.

### 5. Scheduled Time Processing

The notebook converts scheduled departure and arrival times from `CRSDepTime` and `CRSArrTime` into local datetime columns:

```text
sched_dep_local
sched_arr_local
```

If the scheduled arrival time is earlier than the scheduled departure time, the notebook treats the arrival as an overnight flight and adds one day.

### 6. UTC Conversion

Because the weather data uses UTC timestamps, scheduled flight times are converted to UTC.

The notebook maps U.S. states and territories to time zones and includes airport-specific overrides for airports where state-level time zones are not precise enough.

Created columns:

```text
OriginTimezone
DestTimezone
sched_dep_utc
sched_arr_utc
```

### 7. Weather Loading and Station Mapping

Weather data is loaded in chunks to reduce memory usage.

The notebook maps ASOS station IDs to airport IATA codes. For many U.S. mainland airports, it removes the leading `K` from four-letter station codes. For special cases such as Alaska, Hawaii, Puerto Rico, Guam, and U.S. territories, it uses a manual station mapping.

Example mappings:

```text
PANC -> ANC
PHNL -> HNL
TJSJ -> SJU
PGUM -> GUM
```

Executed weather summary after filtering:

| Item | Value |
|---|---:|
| Weather rows after filtering | 850,094 |
| Weather columns | 18 |
| Matched weather airports | 11 |
| First weather time | 2019-01-01 |
| Last weather time | 2026-02-03 |
| Memory usage | 307.01 MB |

### 8. Weather Join

The notebook joins weather data to flights twice:

1. Origin weather before scheduled departure.
2. Destination weather before scheduled arrival.

It uses `pd.merge_asof()` with:

```text
direction = "backward"
tolerance = 3 hours
```

This means the selected weather observation must occur before the scheduled flight time and must be within three hours.

Executed join result:

| Item | Value |
|---|---:|
| Rows after origin + destination weather join | 63,327 |
| Origin airports matched | 8 |
| Destination airports matched | 8 |
| Delayed rate | 13.55% |
| Memory usage | 58.66 MB |

---

## Features Used

The final model uses **46 features**:

- 39 numeric features
- 7 categorical features

### Numeric Features

Time and schedule features:

```text
dep_hour
arr_hour
dep_month
dep_dayofweek
is_weekend
CRSElapsedTime
Distance
```

Origin weather features:

```text
origin_wx_tmpf
origin_wx_dwpf
origin_wx_relh
origin_wx_drct
origin_wx_sknt
origin_wx_p01i
origin_wx_alti
origin_wx_mslp
origin_wx_vsby
origin_wx_gust
origin_wx_feel
origin_wx_age_min
origin_wx_rain
origin_wx_snow
origin_wx_fog
origin_wx_thunder
```

Destination weather features:

```text
dest_wx_tmpf
dest_wx_dwpf
dest_wx_relh
dest_wx_drct
dest_wx_sknt
dest_wx_p01i
dest_wx_alti
dest_wx_mslp
dest_wx_vsby
dest_wx_gust
dest_wx_feel
dest_wx_age_min
dest_wx_rain
dest_wx_snow
dest_wx_fog
dest_wx_thunder
```

### Categorical Features

```text
Reporting_Airline
Origin
Dest
origin_wx_skyc1
origin_wx_skyc2
dest_wx_skyc1
dest_wx_skyc2
```

### Route Feature

A route column is created as:

```text
Origin_Dest
```

Routes with fewer than `MIN_ROUTE_COUNT` observations are grouped as `OTHER`.

The route feature is mainly used for route delay-rate analysis. It is not included in the final categorical feature list used by the model.

---

## Models Trained

The notebook trains two machine learning models.

### Random Forest

The Random Forest pipeline uses:

- median imputation for numeric columns;
- most-frequent imputation for categorical columns;
- one-hot encoding for categorical columns;
- class balancing to help with the delayed-flight minority class.

Model configuration:

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
```

### Neural Network

The Neural Network pipeline uses:

- median imputation for numeric columns;
- standard scaling for numeric columns;
- most-frequent imputation for categorical columns;
- one-hot encoding for categorical columns;
- early stopping.

Model configuration:

```python
MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    alpha=0.0005,
    batch_size=512,
    learning_rate_init=0.001,
    max_iter=60,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=8,
    random_state=42
)
```

---

## Evaluation Results

The notebook uses a date-based split:

```text
Training data: flights before 2025-01-01
Test data: flights on or after 2025-01-01
```

Executed split:

| Dataset | Rows | Delayed Rate |
|---|---:|---:|
| Train | 49,209 | 12.09% |
| Test | 14,118 | 18.63% |

### Model Comparison

| Model | Accuracy | Precision Delayed | Recall Delayed | F1 Delayed | ROC AUC | Average Precision |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.8165 | 0.5131 | 0.2897 | 0.3704 | 0.7283 | 0.4342 |
| Neural Network | 0.8207 | 0.6992 | 0.0654 | 0.1196 | 0.6719 | 0.3736 |

The Random Forest was selected because it had the highest average precision and the highest ROC AUC.

### Final Model

The selected final model is:

```text
Random Forest
```

The final model was retrained on all prepared modeling rows:

```text
Rows used for final training: 63,327
Features used: 46
Prediction threshold: 0.50
```

The notebook also performs a sanity check on the final trained model. This sanity check is run on training/prepared data and should not be interpreted as a true holdout-test score.

Sanity-check result:

| Accuracy | Precision Delayed | Recall Delayed | F1 Delayed |
|---:|---:|---:|---:|
| 0.9235 | 0.6851 | 0.8052 | 0.7403 |

---

## Final Saved Files

The notebook saves three output files:

| File | Description |
|---|---|
| `final_selected_delay30_origin_dest_weather.joblib` | Final trained scikit-learn pipeline |
| `final_prediction_threshold.joblib` | Probability threshold used to classify delayed flights |
| `final_modeling_dataset_delay30.parquet` | Prepared modeling dataset with selected features and labels |

Executed file sizes:

| File | Size |
|---|---:|
| `final_selected_delay30_origin_dest_weather.joblib` | 61.44 MB |
| `final_prediction_threshold.joblib` | ~0 MB |
| `final_modeling_dataset_delay30.parquet` | 3.90 MB |

---

## How to Run the Project

### Option 1: Run on Kaggle

This notebook is designed for Kaggle paths.

1. Upload or open `Flight_Delay_Prediction.ipynb` in Kaggle.
2. Attach the required flight dataset.
3. Attach the required ASOS weather dataset.
4. Confirm that the paths in Section 1 match the dataset paths.
5. Run all cells from top to bottom.

### Option 2: Run Locally

To run locally, install the required Python packages:

```bash
pip install pandas numpy matplotlib scikit-learn joblib pyarrow jupyter
```

Then update these variables in the notebook:

```python
FLIGHT_FILES = [
    "path/to/bts_ontime_2019.csv",
    "path/to/bts_ontime_2020.csv",
    ...
]

WEATHER_PATH = "path/to/historical_weather_data_2019_2026.csv"
```

Start Jupyter:

```bash
jupyter notebook
```

Open the notebook and run all cells.

---

## How to Use the Saved Model

The saved model is a scikit-learn pipeline. It includes preprocessing and the selected classifier, but it expects the input data to already contain the same engineered feature columns used during training.

Example:

```python
import joblib
import pandas as pd

model = joblib.load("final_selected_delay30_origin_dest_weather.joblib")
threshold = joblib.load("final_prediction_threshold.joblib")

# Example: load prepared rows with the same feature columns
data = pd.read_parquet("final_modeling_dataset_delay30.parquet")

feature_cols = [
    col for col in data.columns
    if col not in [
        "Delayed",
        "FlightDate",
        "sched_dep_local",
        "sched_arr_local",
        "sched_dep_utc",
        "sched_arr_utc",
        "route"
    ]
]

X = data[feature_cols]

delay_probability = model.predict_proba(X)[:, 1]
predicted_delayed = (delay_probability >= threshold).astype(int)

output = X.copy()
output["delay_probability"] = delay_probability
output["predicted_delayed"] = predicted_delayed

print(output.head())
```

Prediction output meaning:

```text
delay_probability = model-estimated probability that the flight arrives at least 30 minutes late
predicted_delayed = 1 if delay_probability >= threshold, otherwise 0
```

---

## Project Structure

Suggested repository structure:

```text
flight-delay-prediction/
│
├── Flight_Delay_Prediction.ipynb
├── README.md
│
├── outputs/
│   ├── final_selected_delay30_origin_dest_weather.joblib
│   ├── final_prediction_threshold.joblib
│   └── final_modeling_dataset_delay30.parquet
│
└── data/
    ├── bts_ontime_2019.csv
    ├── bts_ontime_2020.csv
    ├── ...
    └── historical_weather_data_2019_2026.csv
```

The notebook currently saves output files in the working directory. If you want to save them into an `outputs/` folder, update:

```python
FINAL_MODEL_FILE = "outputs/final_selected_delay30_origin_dest_weather.joblib"
FINAL_THRESHOLD_FILE = "outputs/final_prediction_threshold.joblib"
FINAL_DATASET_FILE = "outputs/final_modeling_dataset_delay30.parquet"
```

---

## Important Settings

| Setting | Meaning |
|---|---|
| `DELAY_THRESHOLD_MINUTES = 30` | A flight is delayed if arrival delay is at least 30 minutes |
| `WEATHER_TOLERANCE = "3h"` | Weather must be within 3 hours before the scheduled event |
| `PREDICTION_THRESHOLD = 0.50` | Probability cutoff for classifying a flight as delayed |
| `TEST_SPLIT_DATE = "2025-01-01"` | Date used for train/test split |
| `MIN_ROUTE_COUNT = 100` | Minimum route count before keeping route as a common route |
| `RANDOM_STATE = 42` | Reproducibility seed |
| `MLP_MAX_ITER = 60` | Maximum neural network training iterations |

---

## Data Leakage Prevention

The notebook avoids using information that would not be known before the flight.

It does not use actual flight outcome columns as model inputs, such as:

```text
Actual departure time
Actual arrival time
Actual delay cause fields
Actual elapsed time
Arrival delay minutes as a feature
```

`ArrDelayMinutes` is used only to create the target variable.

Weather is matched only from observations before scheduled departure or arrival, not after.

---

## Limitations

This project has several important limitations:

1. **Weather coverage reduces the usable dataset.**  
   The original flight dataset contains more than 46 million rows, but after matching both origin and destination weather, the final modeling dataset contains 63,327 rows.

2. **Only matched airports are included.**  
   In the executed notebook, the final joined dataset matched 8 origin airports and 8 destination airports.

3. **The saved model is not a raw-data prediction system.**  
   It does not automatically convert raw flight schedules and raw weather observations into features. The input must already contain the engineered feature columns.

4. **The target is based only on arrival delay.**  
   The model predicts arrival delay of at least 30 minutes, not departure delay, cancellation, diversion, or delay cause.

5. **The final sanity-check score is not a true generalization score.**  
   The most reliable generalization comparison is the test-set evaluation before the final retraining step.

6. **Weather matching uses nearest previous observation only.**  
   It does not use weather forecasts, future observations, radar, airport traffic, aircraft rotation, crew availability, or air traffic control constraints.

---

## Possible Improvements

Future versions could improve the project by:

- Expanding and validating weather station mapping to cover more airports.
- Adding route as a direct model feature if useful.
- Adding carrier-level historical delay rates.
- Adding airport congestion features.
- Adding rolling delay features by airport, airline, and route.
- Including cancellation and diversion prediction.
- Using calibrated probabilities with `CalibratedClassifierCV`.
- Testing gradient boosting models such as XGBoost, LightGBM, or HistGradientBoosting.
- Saving a full preprocessing function that can transform raw flight and weather data into model-ready rows.
- Building an API or web interface for real-time prediction.

---

## Conclusion

This project predicts whether a flight will arrive at least 30 minutes late using schedule, route, airline, and weather information. It carefully joins origin and destination weather using only observations available before the scheduled flight times, reducing data leakage risk.

The Random Forest model performed better than the Neural Network on average precision and ROC AUC, so it was selected as the final model and saved for later prediction.
