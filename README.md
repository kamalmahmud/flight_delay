# Flight Delay Prediction Using Flight Schedule and Weather Data

This project predicts whether a flight will arrive **30 minutes or more late** using historical flight schedule data and
weather data. The notebook compares two machine-learning models: a **Random Forest Classifier** and a **Neural Network /
MLPClassifier**.

## Project Overview

Flight delays are affected by many factors, including airline, route, scheduled time, distance, and weather conditions.
This notebook builds a supervised machine-learning pipeline that learns from historical flight records and weather
reports to predict whether a flight will be delayed.

The prediction target is:

```python
Delayed = 1 if ArrDelayMinutes >= 30
Delayed = 0
otherwise
```

So, in this project, a delayed flight means the flight arrived **at least 30 minutes late**.

## Main Goal

The goal of this project is to answer the question:

> Can we predict whether a flight will be delayed by 30 minutes or more using only scheduled flight information and
> weather available before the flight?

## Datasets Used

The notebook uses two main datasets:

### 1. BTS Flight On-Time Performance Data

This dataset contains historical U.S. flight records, including:

* Flight date
* Airline
* Origin airport
* Destination airport
* Scheduled departure time
* Scheduled arrival time
* Scheduled elapsed time
* Distance
* Arrival delay minutes

The notebook uses flight data from **2019 onward** by default.

### 2. ASOS Historical Weather Data

This dataset contains historical weather observations from airport weather stations, including:

* Temperature
* Dew point
* Humidity
* Wind speed
* Wind direction
* Precipitation
* Visibility
* Pressure
* Cloud cover
* Weather codes such as rain, snow, fog, and thunderstorm

## How Weather Data Is Used

Weather is joined to each flight in two ways:

### Origin Weather

Weather at the departure airport is matched before the scheduled departure time.

Example:

```text
Flight leaves JFK at 10:00
Use the latest JFK weather report before 10:00
```

### Destination Weather

Weather at the arrival airport is matched before the scheduled arrival time.

Example:

```text
Flight arrives at LAX at 13:30
Use the latest LAX weather report before 13:30
```

The notebook uses a backward time merge with a tolerance of 3 hours:

```python
direction = "backward"
tolerance = "3h"
```

This prevents data leakage because the model only uses weather information that would have been available before the
scheduled flight time.

## Features Used by the Model

The model uses three main groups of features.

### 1. Flight and Time Features

* Scheduled departure hour
* Scheduled arrival hour
* Month
* Day of week
* Weekend flag
* Scheduled elapsed time
* Flight distance
* Airline
* Origin airport
* Destination airport

### 2. Origin Weather Features

Weather at the departure airport before departure:

* Temperature
* Dew point
* Relative humidity
* Wind direction
* Wind speed
* Precipitation
* Altimeter pressure
* Sea-level pressure
* Visibility
* Wind gust
* Feels-like temperature
* Weather report age in minutes
* Rain flag
* Snow flag
* Fog flag
* Thunderstorm flag
* Cloud cover

### 3. Destination Weather Features

Weather at the arrival airport before scheduled arrival:

* Temperature
* Dew point
* Relative humidity
* Wind direction
* Wind speed
* Precipitation
* Altimeter pressure
* Sea-level pressure
* Visibility
* Wind gust
* Feels-like temperature
* Weather report age in minutes
* Rain flag
* Snow flag
* Fog flag
* Thunderstorm flag
* Cloud cover

## Models Used

The notebook trains and compares two models:

### Random Forest Classifier

The Random Forest model is trained using:

* 200 trees
* Maximum depth of 20
* Balanced class weights
* Median imputation for numeric features
* One-hot encoding for categorical features

### Neural Network / MLPClassifier

The neural network uses:

* Two hidden layers: 64 and 32 neurons
* ReLU activation
* Adam optimizer
* Early stopping
* Standard scaling for numeric features
* One-hot encoding for categorical features

## Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC AUC
* Average precision
* Confusion matrix
* ROC curve
* Precision-recall curve

The final model is selected using:

```text
Highest average precision, then ROC AUC as a tie-breaker
```

## Train/Test Split

The notebook uses a time-based split:

```text
Training data: before 2025
Testing data: 2025 and later
```

If the time-based split is not possible, the notebook falls back to a stratified random train/test split.

## Output Files

After training, the notebook saves three files:

```text
final_selected_delay30_origin_dest_weather.joblib
final_prediction_threshold.joblib
final_modeling_dataset_delay30.parquet
```

### File Descriptions

| File                                                | Description                              |
|-----------------------------------------------------|------------------------------------------|
| `final_selected_delay30_origin_dest_weather.joblib` | Saved final trained model                |
| `final_prediction_threshold.joblib`                 | Saved probability threshold              |
| `final_modeling_dataset_delay30.parquet`            | Final prepared dataset used for modeling |

## How to Run the Notebook

### 1. Open the Notebook

Open the notebook in Kaggle, Jupyter Notebook, or JupyterLab.

### 2. Make Sure the Data Paths Are Correct

The notebook currently uses Kaggle dataset paths such as:

```python
FLIGHT_FILES = [
     "/kaggle/input/datasets/kamalalqedra/bts-ontime-performance-2019-present-csv/bts_ontime_2019.csv",
    "/kaggle/input/datasets/kamalalqedra/bts-ontime-performance-2019-present-csv/bts_ontime_2020.csv",
    "/kaggle/input/datasets/kamalalqedra/bts-ontime-performance-2019-present-csv/bts_ontime_2021.csv",
    "/kaggle/input/datasets/kamalalqedra/bts-ontime-performance-2019-present-csv/bts_ontime_2022.csv",
    "/kaggle/input/datasets/kamalalqedra/bts-ontime-performance-2019-present-csv/bts_ontime_2023.csv",
    "/kaggle/input/datasets/kamalalqedra/bts-ontime-performance-2019-present-csv/bts_ontime_2024.csv",
    "/kaggle/input/datasets/kamalalqedra/bts-ontime-performance-2019-present-csv/bts_ontime_2025.csv",
    "/kaggle/input/datasets/kamalalqedra/bts-ontime-performance-2019-present-csv/bts_ontime_2026.csv",
]

WEATHER_PATH = "/kaggle/input/datasets/sehamhakimothman/asos-weather-data-2019-2026/historical_weather_data_2019_2026.csv"
```

If running locally, update these paths to match the location of your downloaded datasets.

### 3. Run All Cells

Run the notebook from top to bottom. The notebook will:

1. Load the flight data
2. Load and filter the weather data
3. Create the delay target
4. Join weather to flights
5. Engineer features
6. Train the models
7. Compare model performance
8. Select the best model
9. Save the final model and dataset

## Requirements

The notebook uses the following Python libraries:

```text
numpy
pandas
matplotlib
scikit-learn
joblib
pyarrow
```

Install them with:

```bash
pip install numpy pandas matplotlib scikit-learn joblib pyarrow
```

## Data Leakage Prevention

The notebook avoids using information that would not be known before the flight. It does not use actual departure time,
actual arrival time, or delay cause columns as input features.

Weather is only taken from reports before the scheduled departure or scheduled arrival time.

This makes the prediction more realistic because the model only uses information available before or during scheduling.

## Example Prediction Output

After training, the saved model can produce:

```text
delay_probability
predicted_delayed
```

Example meaning:

| Output                     | Meaning                                   |
|----------------------------|-------------------------------------------|
| `delay_probability = 0.72` | The model estimates a 72% chance of delay |
| `predicted_delayed = 1`    | The flight is predicted to be delayed     |
| `predicted_delayed = 0`    | The flight is predicted not to be delayed |

The default prediction threshold is:

```python
0.50
```

So if the model probability is greater than or equal to 0.50, the flight is predicted as delayed.

## Project Conclusion

This notebook builds a full machine-learning workflow for flight delay prediction. It combines scheduled flight
information with origin and destination weather data, trains two models, compares their performance, and saves the best
final model.

The project shows that weather, route, airline, time, and distance can be used together to estimate the probability of a
flight arriving 30 minutes or more late.

## Limitations

This model has some limitations:

* It depends on historical data quality.
* It only predicts delays of 30 minutes or more.
* It uses scheduled information and weather, but not real-time airport congestion.
* It may not generalize perfectly to airports, airlines, or weather patterns not well represented in the training data.
* The notebook paths are designed for Kaggle and may need changes for local use.

## Future Improvements

Possible improvements include:

* Adding airport congestion features
* Adding previous-flight delay information
* Using more advanced models such as XGBoost or LightGBM
* Tuning the prediction threshold
* Adding live weather API support
* Building a web app for real-time predictions
* Adding feature importance analysis
