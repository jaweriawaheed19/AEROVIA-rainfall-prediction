<div align="center">

# AEROVIA

### Pakistan Rainfall Prediction System

### Machine Learning–Based Next-Day Rainfall Prediction Across Pakistan

<p>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/XGBoost-Machine%20Learning-EC6B23?style=for-the-badge" alt="XGBoost">
  <img src="https://img.shields.io/badge/Scikit--learn-Model%20Evaluation-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Status-Academic%20Project-2E8B57?style=for-the-badge" alt="Academic Project">
</p>

<p>
  <strong>Explore historical rainfall · Predict next-day rain · Estimate rainfall amount · Visualize location-wise results</strong>
</p>

</div>

---

## Project Overview

**AEROVIA**, the Pakistan Rainfall Prediction System, is a machine learning academic project that predicts next-day rainfall for **22 selected locations across Pakistan**.

The system uses historical weather observations and a two-stage machine learning approach:

1. An **XGBoost classifier** estimates the probability that rainfall will reach at least **1 mm** on the next day.
2. An **XGBoost regressor** estimates the rainfall amount in millimetres. The dashboard displays an amount of **0 mm** when the occurrence classifier predicts no rain.

A Streamlit dashboard presents the predictions, probability-based alerts, historical rainfall charts, a map of the selected locations, and model evaluation information.

> **Important:** This is an offline, historical-data-based next-day prediction system. It is not a live weather forecasting service and does not predict floods.

---

## Features

| Feature | Description |
|---|---|
| Rainfall occurrence | Predicts whether rainfall will be at least 1 mm on the next day |
| Rainfall amount | Uses a separately trained XGBoost regression model to estimate rainfall in millimetres |
| Location selection | Filter predictions by province and selected location |
| Pakistan map | Displays location markers coloured by predicted rainfall probability |
| Rain alerts | Highlights locations where predicted rain probability reaches the configured threshold |
| Historical trends | Explore historical rainfall over 30 days, 90 days, 365 days, or the complete dataset |
| Historical simulation | Train temporary models using only data before a selected historical date and compare predictions with actual observations |
| Model evaluation | View classification metrics from the saved model's chronological test evaluation |

---

## Machine Learning Approach

The project uses a **two-stage prediction pipeline**. The two outputs have different purposes and are produced by separately trained models.

### Stage 1 — Rainfall occurrence classification

**Algorithm:** XGBoost Classifier

The classifier estimates the probability of rainfall occurrence for the next day.

- Positive class: rainfall of at least **1 mm**
- Negative class: rainfall below **1 mm**
- The dashboard displays the predicted probability and classification.
- The configured rain-alert threshold is **30% probability**.

### Stage 2 — Rainfall amount regression

**Algorithm:** XGBoost Regressor

A separate regression model estimates the next-day rainfall amount in millimetres. Its prediction is transformed back from the model's log-target scale.

The classifier controls how the amount is presented:

- If rain is predicted, the estimated amount is displayed.
- If rain is not predicted, the displayed amount is set to **0 mm**.

This is a two-stage system: occurrence classification followed by amount estimation.

### Prediction workflow

```text
Historical rainfall dataset
          │
          ▼
Prepare latest available location-level observations
          │
          ├─────────────────────────────┐
          ▼                             ▼
 XGBoost Classifier              XGBoost Regressor
 Rain occurrence                 Rainfall amount
          │                             │
          ▼                             │
 Probability of rain                    │
          │                             │
          └──────────────┬──────────────┘
                         ▼
             Apply occurrence decision
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
         Rain predicted          No rain predicted
             │                       │
       Show estimated             Show 0 mm
       rainfall amount
             │                       │
             └───────────┬───────────┘
                         ▼
                Streamlit dashboard
```

---

## Model Performance

The dashboard reads the saved rainfall-occurrence evaluation report and displays the classification metrics.

### Rainfall occurrence classifier

| Metric | Recorded value |
|---|---:|
| Accuracy | 0.8413 |
| Balanced Accuracy | 0.7333 |
| Precision | 0.5157 |
| Recall | 0.5724 |
| F1 Score | 0.5426 |
| ROC-AUC | 0.8443 |
| PR-AUC | 0.5665 |
| Brier Score | 0.1000 |
| Log Loss | 0.3269 |

The occurrence model was evaluated using a chronological split, with training target dates through 2022 and testing observations from 2023 onward.

### Rainfall amount regressor

The amount model was evaluated separately. Its reported rainy-day-only test metrics were:

| Metric | XGBoost regressor |
|---|---:|
| MAE | 5.7303 mm |
| RMSE | 12.4136 mm |
| R² | 0.0017 |

For context, the historical baseline recorded MAE 6.3428 mm, RMSE 12.3812 mm, and R² 0.0069 on the same rainy-day-only evaluation.

The amount model's results are mixed: it improves MAE over that baseline, but does not improve RMSE or R². These limitations should be considered when interpreting rainfall amount estimates.

> **Evaluation note:** Classification metrics and rainfall-amount metrics describe different tasks and should not be compared directly. Historical simulation results for a single selected date are illustrative and are not a substitute for a broad independent test.

---

## Dataset

The project uses a local historical rainfall dataset:

| Dataset detail | Information |
|---|---|
| File | `data/rainfall.csv` |
| Source | NASA POWER historical observations |
| Date coverage | 1 January 1981 – 30 June 2026 |
| Selected locations | 22 |
| Records | 365,574 |
| Format | CSV |

The dashboard reads this local dataset. It does not automatically fetch new observations when opened.

**Data freshness limitation:** The dataset ends on **30 June 2026**. The generated prediction date is therefore based on the final available historical observation, not the current date. The dashboard explicitly identifies historical predictions and warns when the forecast date is in the past.

---

## Dashboard

The Streamlit application includes:

- A nationwide summary of selected locations and rain alerts.
- Province and location filters.
- A selected-location next-day prediction panel.
- A rainfall probability gauge.
- Historical rainfall charts.
- An interactive Pakistan map with probability-coloured markers.
- A table of next-day predictions for the selected province or all locations.
- A historical prediction simulation.
- A model performance section.

---

## Historical Prediction Simulation

The dashboard includes a historical simulation feature to test the prediction pipeline on a selected date.

For the selected date, the simulation:

1. Uses historical records available before that date.
2. Trains temporary rainfall-occurrence and rainfall-amount models.
3. Generates predictions for the selected locations.
4. Compares predicted rainfall occurrence and amount with the actual recorded values.
5. Displays accuracy, precision, recall, F1 score, amount MAE, and location-wise comparisons.

The simulation is separate from the saved production models used for the dashboard's normal next-day prediction.

Two additional validation scripts are included:

- `validation/historical_backtest.py`
- `validation/historical_monthly_backtest.py`

These scripts run selected historical backtests and save their results in the `reports/` directory. Their selected evaluation dates should be considered when interpreting the results; they are not an exhaustive daily evaluation of every possible forecast date.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.12 | Main programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| XGBoost | Rainfall occurrence and amount models |
| Scikit-learn | Machine learning utilities and evaluation |
| Joblib | Saving and loading trained models |
| Streamlit | Interactive dashboard |
| Plotly | Charts and visualizations |
| Folium | Interactive map |
| Streamlit-Folium | Displaying Folium maps in Streamlit |

---

## Project Structure

```text
AEROVIA-rainfall-prediction/
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
└── validation/
    ├── historical_backtest.py
    └── historical_monthly_backtest.py
```

---

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/jaweriawaheed19/AEROVIA-rainfall-prediction.git
cd AEROVIA-rainfall-prediction
```

### 2. Create a virtual environment

Python **3.12** is the recommended version for this project.

**Windows PowerShell:**

```powershell
py -3.12 -m venv .venv
```

### 3. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script activation, use Command Prompt or adjust the execution policy for your user according to your system's security requirements.

### 4. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run the dashboard

```powershell
streamlit run app.py
```

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open that address in your browser.

---

## Training the Models

The repository includes separate training scripts for the two prediction tasks.

### Train the rainfall occurrence classifier

```powershell
python train_occurrence.py
```

### Train the rainfall amount regressor

```powershell
python train_amount.py
```

Run these scripts from the project root with the virtual environment activated. They use the local dataset and generate the corresponding trained model and evaluation outputs.

---

## Limitations and Future Improvements

The current implementation has several important limitations:

- **Historical data only:** The dashboard does not fetch live weather observations or current forecasts.
- **Next-day horizon:** The system predicts the next day only; it does not provide a multi-day forecast.
- **Selected locations:** Coverage is limited to 22 selected locations, not every district or grid cell in Pakistan.
- **Amount-model limitations:** Rainfall amount regression has limited predictive performance and should be improved with further feature engineering, tuning, and evaluation.
- **Rainfall is not flooding:** Rainfall predictions do not directly indicate river flooding, flash floods, flood severity, or affected population.
- **No guarantee of accuracy:** Model performance on historical test data does not guarantee performance on future weather conditions.

Potential future improvements include updating the dataset, expanding geographical coverage, adding meteorological forecast inputs, testing additional features, improving regression performance, and evaluating the system across more historical dates.

---

## Academic Project Note

This project demonstrates the application of supervised machine learning to rainfall occurrence classification and rainfall amount regression, together with historical validation and an interactive visualization dashboard.

It is intended for academic demonstration and experimentation. It should not be used as an official meteorological warning system or for safety-critical decisions.

---

<div align="center">

### AEROVIA

**Pakistan Rainfall Prediction System**

**Built with Python · XGBoost · Streamlit**

<sub>Academic Machine Learning Project</sub>

</div>
