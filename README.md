# Pakistan Rainfall Prediction System

A machine-learning-based rainfall prediction system for selected locations across Pakistan. The project uses historical rainfall and weather observations to predict next-day rainfall occurrence and estimate rainfall amount.

The system includes two trained XGBoost models, an interactive Streamlit dashboard, a Pakistan rainfall probability map, historical prediction simulations, and model evaluation reports.

## 1. Project Objectives

The main objectives of this project are:

- Predict whether rainfall will occur at selected locations across Pakistan.
- Estimate the expected rainfall amount in millimetres.
- Display rainfall predictions through an interactive dashboard.
- Visualize rainfall probabilities on a map of Pakistan.
- Evaluate model performance using appropriate machine-learning metrics.
- Test predictions against historical observations using leakage-aware simulations and backtesting.

## 2. Machine Learning Approach

The system uses a two-stage machine-learning approach consisting of two separately trained XGBoost models.

### 2.1 Rainfall Occurrence Prediction

**Algorithm:** XGBoost Classifier

The occurrence model predicts whether rainfall will occur on the next day.

- Rainfall of at least 1 mm is classified as a rainy day.
- The model produces a probability of rainfall.
- A probability threshold of 30% is used by the dashboard to determine rain alerts.
- The model uses historical weather and rainfall-related features.

The trained model is saved as:

`models/rain_occurrence_xgb.pkl`

The training script also generates the next-day prediction file:

`reports/rain_occurrence_forecast.csv`

### 2.2 Rainfall Amount Estimation

**Algorithm:** XGBoost Regressor

The amount model estimates the expected rainfall amount in millimetres.

- It is trained using rainy-day observations.
- A logarithmic transformation is applied to rainfall amounts during training.
- Predictions are converted back into millimetres.
- The occurrence prediction is used to gate the amount estimate.

If the occurrence model predicts no rain, the estimated rainfall amount is set to zero.

The trained model is saved as:

`models/rain_amount_xgb.pkl`

### 2.3 Combined Prediction

The final prediction combines both models:

1. The occurrence model predicts the probability of rainfall.
2. The occurrence prediction determines whether rain is expected.
3. The amount model estimates rainfall in millimetres.
4. If no rainfall is predicted, the final estimated amount is set to zero.

This approach provides two outputs: rainfall occurrence and estimated rainfall amount.

## 3. Dataset

The project uses a historical rainfall dataset stored locally.

**Dataset file:** `data/rainfall.csv`

Dataset details:

| Property | Description |
|---|---|
| Total records | 365,574 |
| Locations | 22 selected locations |
| Columns | 14 |
| Historical start date | 1 January 1981 |
| Historical end date | 30 June 2026 |
| Data format | CSV |

The dataset contains historical rainfall and weather observations used for feature generation, model training, evaluation, and prediction.

The project currently uses an offline dataset. It does not automatically fetch live weather observations or real-time forecasts from an external API.

## 4. Dashboard Features

The project includes an interactive dashboard developed using Streamlit.

Its main features include:

- **Province Selection:** Filter available predictions by province.
- **Location Selection:** View predictions for an individual location.
- **Rainfall Probability:** Display the predicted probability of rainfall.
- **Rain Alerts:** Identify locations where rainfall probability reaches the configured alert threshold.
- **Rainfall Amount:** Display estimated rainfall in millimetres.
- **Historical Rainfall Chart:** Explore historical rainfall observations.
- **Pakistan Map:** Visualize rainfall probabilities using colour-coded markers.
- **All Locations Table:** View next-day predictions for all available locations.
- **Historical Prediction Simulation:** Select a historical date and compare model predictions with actual observations.
- **Model Performance:** View rainfall occurrence evaluation metrics.

## 5. Project Structure

```text
Pakistan-Rainfall-Prediction/
│
├── app.py
├── historical_simulation.py
├── train_occurrence.py
├── train_amount.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── rainfall.csv
│
├── models/
│   ├── rain_occurrence_xgb.pkl
│   └── rain_amount_xgb.pkl
│
├── reports/
│   ├── rain_occurrence_forecast.csv
│   ├── rain_occurrence_model_evaluation.csv
│   ├── rain_amount_model_evaluation.csv
│   ├── rain_amount_test_predictions.csv
│   ├── historical_backtest_results.csv
│   ├── historical_backtest_summary.csv
│   ├── historical_12_month_backtest_results.csv
│   └── historical_12_month_backtest_summary.csv
│
├── validation/
│   ├── historical_backtest.py
│   └── historical_monthly_backtest.py
│
└── archive/
```

Note: The reports directory contains generated evaluation and validation files. Some files are created only after running their corresponding scripts. The archive directory contains retained experiment files and previous project work.

## 6. Technologies and Dependencies

The project is developed using Python and the following libraries:

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Dataset handling and preprocessing |
| NumPy | Numerical operations |
| XGBoost | Machine-learning models |
| Scikit-learn | Preprocessing and evaluation |
| Joblib | Saving and loading trained models |
| Streamlit | Interactive web dashboard |
| Plotly | Data visualization |
| Folium | Interactive geographical map |
| Streamlit-Folium | Displaying Folium maps in Streamlit |

## 7. Installation and Setup

### 7.1 Prerequisites

- Python 3.12
- pip
- The project dataset
- The trained model and report files required by the dashboard

### 7.2 Clone the Repository

If the project is hosted on GitHub:

```bash
git clone YOUR_REPOSITORY_URL
```

Navigate into the project directory:

```bash
cd Pakistan-Rainfall-Prediction
```

Alternatively, open the downloaded project folder in Visual Studio Code.

### 7.3 Create a Virtual Environment

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
```

### 7.4 Activate the Virtual Environment

```powershell
.venv\Scripts\Activate.ps1
```

### 7.5 Install Dependencies

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install all project dependencies:

```powershell
pip install -r requirements.txt
```

## 8. Running the Dashboard

Before starting the application, make sure the required dataset, model, and report files are available.

Run the following command from the project root directory:

```powershell
streamlit run app.py
```

The dashboard will open in your default browser.

Default local address:

`http://localhost:8501`

To stop the application, press `Ctrl + C` in the terminal.

## 9. Training the Machine-Learning Models

Both models can be trained using their respective Python scripts.

Run all commands from the project root directory.

### 9.1 Train the Rainfall Occurrence Model

```powershell
python train_occurrence.py
```

This script performs the following operations:

1. Loads the historical rainfall dataset.
2. Prepares the model features and prediction targets.
3. Applies a chronological training and testing split.
4. Trains an XGBoost classification model.
5. Evaluates the model using held-out observations.
6. Trains the final production model.
7. Generates next-day rainfall predictions for available locations.
8. Saves the trained model and evaluation reports.

Generated files:

- `models/rain_occurrence_xgb.pkl`
- `reports/rain_occurrence_model_evaluation.csv`
- `reports/rain_occurrence_forecast.csv`

### 9.2 Train the Rainfall Amount Model

```powershell
python train_amount.py
```

This script:

1. Loads the historical rainfall dataset.
2. Prepares rainfall amount prediction features.
3. Applies a chronological training and testing split.
4. Trains an XGBoost rainfall amount regressor.
5. Evaluates rainfall amount predictions.
6. Trains a temporary occurrence classifier for combined evaluation.
7. Evaluates the combined occurrence-and-amount prediction approach.
8. Trains the final production amount model.
9. Saves the trained model and evaluation reports.

Generated files:

- `models/rain_amount_xgb.pkl`
- `reports/rain_amount_model_evaluation.csv`
- `reports/rain_amount_test_predictions.csv`

**Important:** The amount training script protects an existing model. If `models/rain_amount_xgb.pkl` already exists, the script stops instead of overwriting it. Back up or deliberately move the existing model before retraining.

## 10. Historical Prediction Simulation

The project includes a separate historical simulation module:

`historical_simulation.py`

It evaluates how the prediction system would have performed on a selected historical date.

The simulation:

- Uses observations available before the selected prediction date for training.
- Trains temporary occurrence and amount models.
- Generates predictions for the selected historical date.
- Compares predictions with actual rainfall observations.
- Calculates occurrence accuracy and rainfall amount MAE.

Run a simulation using:

```powershell
python historical_simulation.py --date 2025-07-01
```

Replace the date with the historical date you want to evaluate, using the `YYYY-MM-DD` format.

The historical simulation is also integrated into the Streamlit dashboard.

## 11. Historical Backtesting

Two additional validation scripts are included in the `validation/` directory.

### 11.1 Multi-Date Historical Backtest

Run:

```powershell
python validation\historical_backtest.py
```

This script evaluates selected historical dates across multiple years.

It calculates:

- Accuracy
- Precision
- Recall
- F1-score
- Rainfall amount MAE
- Confusion matrix counts

Generated reports:

- `reports/historical_backtest_results.csv`
- `reports/historical_backtest_summary.csv`

### 11.2 Twelve-Month Historical Backtest

Run:

```powershell
python validation\historical_monthly_backtest.py
```

This script evaluates selected dates across the twelve months of 2025.

It provides:

- Overall model performance.
- Monthly performance summaries.
- Accuracy and balanced accuracy.
- Precision, recall, and F1-score.
- Rainfall amount MAE.
- Comparison against an always-dry baseline.

Generated reports:

- `reports/historical_12_month_backtest_results.csv`
- `reports/historical_12_month_backtest_summary.csv`

These scripts evaluate selected historical dates, not every day in the dataset. Their results should be interpreted accordingly.

## 12. Model Evaluation Metrics

Different evaluation metrics are used for the two prediction tasks.

### 12.1 Rainfall Occurrence Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Measures the proportion of correct predictions |
| Precision | Measures how many predicted rainy cases were actually rainy |
| Recall | Measures how many actual rainy cases were identified |
| F1-score | Combines precision and recall |
| Balanced Accuracy | Accounts for both rainy and dry classes |
| ROC-AUC | Measures the classifier's ability to distinguish between classes |
| PR-AUC | Evaluates precision-recall performance |
| Brier Score | Measures the accuracy of predicted probabilities |
| Log Loss | Evaluates the quality of predicted probabilities |

### 12.2 Rainfall Amount Metrics

| Metric | Purpose |
|---|---|
| MAE | Measures average absolute prediction error |
| RMSE | Penalizes larger prediction errors more heavily |
| R² | Measures the proportion of target variance explained by the model |

The evaluation reports are saved in the `reports/` directory.

## 13. Important Project Limitations

The following limitations should be considered when interpreting the predictions:

1. **Offline Dataset:** The system currently uses locally stored historical observations and does not fetch live weather data automatically.
2. **Next-Day Prediction:** The current prediction horizon is one day, not a three-day or seven-day forecast.
3. **Selected Locations:** Predictions cover the 22 locations represented in the dataset, not every geographical point in Pakistan.
4. **Historical Forecast Date:** The generated forecast is based on the latest observations available in the dataset. Since the dataset ends on 30 June 2026, the generated prediction is for 1 July 2026. It is not a current weather forecast.
5. **Rainfall Prediction Only:** The system predicts rainfall occurrence and estimates rainfall amount. It does not currently predict floods, flood severity, or flood extent.
6. **Historical Evaluation:** Historical simulation and backtesting results do not guarantee future prediction accuracy.
7. **Model Limitations:** Prediction performance depends on the quality, coverage, and characteristics of the available historical data.

## 14. Required Files for Project Execution

The following files are important for running the dashboard:

| File | Purpose |
|---|---|
| `data/rainfall.csv` | Historical rainfall dataset |
| `reports/rain_occurrence_forecast.csv` | Generated next-day occurrence predictions |
| `reports/rain_occurrence_model_evaluation.csv` | Occurrence model evaluation results |
| `models/rain_amount_xgb.pkl` | Trained rainfall amount model |
| `historical_simulation.py` | Historical prediction simulation module |

The occurrence model file `models/rain_occurrence_xgb.pkl` is required when retraining or regenerating occurrence predictions. The dashboard reads the generated occurrence forecast CSV rather than loading that classifier directly.

## 15. Dataset and Model File Management

The `.gitignore` file excludes large dataset and model files from Git:

- `data/*.csv`
- `models/*.pkl`

These files are still necessary for executing the complete project.

When sharing or submitting the project, include the required dataset, trained models, and generated reports in the submission package separately.

## 16. Academic Use

This project was developed as an academic machine-learning semester project.

It demonstrates the application of supervised machine learning, historical weather data processing, model evaluation, geographical visualization, and interactive dashboard development.

The system is intended for educational and experimental purposes. Its predictions are not official meteorological warnings and should not replace information from government meteorological or disaster-management authorities.