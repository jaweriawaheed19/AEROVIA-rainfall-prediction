import os
import numpy as np
import pandas as pd
import joblib

from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_FILE = "data/rainfall.csv"
MODEL_DIR = "models"
REPORT_DIR = "reports"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

print("=" * 60)
print("PAKISTAN RAINFALL PREDICTION - MODEL TRAINING")
print("=" * 60)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(["location", "date"]).reset_index(drop=True)

print(f"\nDataset loaded: {len(df):,} records")
print(f"Locations: {df['location'].nunique()}")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

# ---------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------
print("\nCreating features...")

group = df.groupby("location", sort=False)

# Historical rainfall features
for lag in [1, 2, 3, 7, 14, 30]:
    df[f"rain_lag_{lag}"] = group["rainfall_mm"].shift(lag)

# Historical weather features
weather_columns = [
    "temperature_c",
    "relative_humidity_percent",
    "wind_speed_mps"
]

for col in weather_columns:
    for lag in [1, 3, 7]:
        df[f"{col}_lag_{lag}"] = group[col].shift(lag)

# Calendar features
df["day_of_year"] = df["date"].dt.dayofyear
df["month"] = df["date"].dt.month

df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

# ---------------------------------------------------------
# 3. SELECT FEATURES
# ---------------------------------------------------------
numeric_features = (
    [f"rain_lag_{lag}" for lag in [1, 2, 3, 7, 14, 30]]
    +
    [
        f"{col}_lag_{lag}"
        for col in weather_columns
        for lag in [1, 3, 7]
    ]
    +
    ["month", "day_sin", "day_cos"]
)

categorical_features = ["location", "province"]

all_features = numeric_features + categorical_features

# ---------------------------------------------------------
# 4. CHRONOLOGICAL TRAIN/TEST SPLIT
# ---------------------------------------------------------
unique_dates = np.sort(df["date"].unique())

split_index = int(len(unique_dates) * 0.80)
split_date = pd.Timestamp(unique_dates[split_index])

print(f"\nTraining cutoff date: {split_date.date()}")
print("Earlier dates = training | Later dates = testing")

# ---------------------------------------------------------
# 5. TRAIN 3 MODELS
# ---------------------------------------------------------
results = []
trained_models = {}

for horizon in [1, 2, 3]:

    print("\n" + "=" * 60)
    print(f"TRAINING MODEL: DAY {horizon}")
    print("=" * 60)

    # Target rainfall for future day
    df[f"target_day_{horizon}"] = (
        df.groupby("location")["rainfall_mm"].shift(-horizon)
    )

    # Remove rows without required features/targets
    model_data = df.dropna(
        subset=all_features + [f"target_day_{horizon}"]
    ).copy()

    # Chronological split
    train_data = model_data[
        model_data["date"] < split_date
    ]

    test_data = model_data[
        model_data["date"] >= split_date
    ]

    X_train = train_data[all_features]
    y_train = train_data[f"target_day_{horizon}"]

    X_test = test_data[all_features]
    y_test = test_data[f"target_day_{horizon}"]

    print(f"Training records: {len(train_data):,}")
    print(f"Testing records:  {len(test_data):,}")

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            ),
            (
                "numeric",
                "passthrough",
                numeric_features
            )
        ]
    )

    # XGBoost regression
    xgb_model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", xgb_model)
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Test
    predictions = pipeline.predict(X_test)

    # Rainfall cannot be negative
    predictions = np.maximum(predictions, 0)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\n========== MODEL EVALUATION ==========")
    print(f"Forecast day: {horizon}")
    print(f"MAE:  {mae:.4f} mm")
    print(f"RMSE: {rmse:.4f} mm")
    print(f"R2:   {r2:.4f}")

    results.append({
        "forecast_day": horizon,
        "MAE_mm": mae,
        "RMSE_mm": rmse,
        "R2": r2,
        "training_records": len(train_data),
        "testing_records": len(test_data)
    })

    # Save model
    model_path = os.path.join(
        MODEL_DIR,
        f"rainfall_day_{horizon}.pkl"
    )

    joblib.dump({
        "pipeline": pipeline,
        "features": all_features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "horizon": horizon
    }, model_path)

    trained_models[horizon] = pipeline

    print(f"\nModel saved: {model_path}")

# ---------------------------------------------------------
# 6. SAVE EVALUATION REPORT
# ---------------------------------------------------------
results_df = pd.DataFrame(results)

report_path = os.path.join(
    REPORT_DIR,
    "model_evaluation.csv"
)

results_df.to_csv(report_path, index=False)

print("\n" + "=" * 60)
print("FINAL MODEL EVALUATION")
print("=" * 60)

print(results_df.to_string(index=False))

print(f"\nEvaluation report saved: {report_path}")

# ---------------------------------------------------------
# 7. GENERATE 3-DAY FORECAST
# ---------------------------------------------------------
print("\nGenerating forecasts for all locations...")

forecast_rows = []

for location in df["location"].unique():

    location_data = df[df["location"] == location].copy()
    location_data = location_data.sort_values("date")

    latest = location_data.iloc[-1]
    latest_date = latest["date"]

    # Create a feature row for each future day
    for horizon in [1, 2, 3]:

        future_date = latest_date + pd.Timedelta(days=horizon)

        row = {
            "location": location,
            "province": latest["province"],
            "date": future_date,
            "month": future_date.month,
            "day_sin": np.sin(
                2 * np.pi * future_date.dayofyear / 365.25
            ),
            "day_cos": np.cos(
                2 * np.pi * future_date.dayofyear / 365.25
            )
        }

        # Rainfall history
        for lag in [1, 2, 3, 7, 14, 30]:
            row[f"rain_lag_{lag}"] = (
                location_data["rainfall_mm"].iloc[-lag]
                if len(location_data) >= lag
                else 0
            )

        # Weather history
        for col in weather_columns:
            for lag in [1, 3, 7]:
                row[f"{col}_lag_{lag}"] = (
                    location_data[col].iloc[-lag]
                    if len(location_data) >= lag
                    else 0
                )

        X_future = pd.DataFrame([row])[all_features]

        prediction = trained_models[horizon].predict(X_future)[0]

        prediction = max(0, float(prediction))

        forecast_rows.append({
            "location": location,
            "province": latest["province"],
            "latitude": latest["latitude"],
            "longitude": latest["longitude"],
            "forecast_date": future_date.date(),
            "forecast_day": horizon,
            "predicted_rainfall_mm": round(prediction, 2)
        })

forecast_df = pd.DataFrame(forecast_rows)

forecast_path = os.path.join(
    REPORT_DIR,
    "latest_forecast.csv"
)

forecast_df.to_csv(forecast_path, index=False)

print("\n========== FORECAST PREVIEW ==========")
print(forecast_df.head(15).to_string(index=False))

print(f"\nForecast saved: {forecast_path}")

print("\n" + "=" * 60)
print("TRAINING AND FORECASTING COMPLETED SUCCESSFULLY")
print("=" * 60)
