import os
import numpy as np
import pandas as pd

from xgboost import XGBRegressor, XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score
)

DATA_FILE = "data/rainfall.csv"
REPORT_DIR = "reports"

os.makedirs(REPORT_DIR, exist_ok=True)

print("=" * 75)
print("TWO-STAGE RAINFALL MODEL EXPERIMENT")
print("Original models and reports will NOT be overwritten")
print("=" * 75)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["location", "date"]
).reset_index(drop=True)

print(f"\nDataset records: {len(df):,}")
print(f"Locations: {df['location'].nunique()}")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

# ---------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------

group = df.groupby("location", sort=False)

for lag in [1, 2, 3, 7, 14, 30]:
    df[f"rain_lag_{lag}"] = group["rainfall_mm"].shift(lag)

weather_columns = [
    "temperature_c",
    "relative_humidity_percent",
    "wind_speed_mps"
]

for col in weather_columns:
    for lag in [1, 3, 7]:
        df[f"{col}_lag_{lag}"] = group[col].shift(lag)

df["day_of_year"] = df["date"].dt.day_of_year
df["month"] = df["date"].dt.month

df["day_sin"] = np.sin(
    2 * np.pi * df["day_of_year"] / 365.25
)

df["day_cos"] = np.cos(
    2 * np.pi * df["day_of_year"] / 365.25
)

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
# 3. CHRONOLOGICAL SPLIT
# ---------------------------------------------------------

unique_dates = np.sort(df["date"].unique())

split_index = int(len(unique_dates) * 0.80)
split_date = pd.Timestamp(unique_dates[split_index])

print(f"\nTest period begins: {split_date.date()}")
print("Training targets must occur BEFORE this date.")

# ---------------------------------------------------------
# 4. PREPROCESSOR
# ---------------------------------------------------------

def create_preprocessor():

    return ColumnTransformer(
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


# ---------------------------------------------------------
# 5. MODEL SETTINGS
# ---------------------------------------------------------

def create_regressor():

    return XGBRegressor(
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


def create_classifier():

    return XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )


# ---------------------------------------------------------
# 6. TRAIN AND COMPARE THREE HORIZONS
# ---------------------------------------------------------

results = []

for horizon in [1, 2, 3]:

    print("\n" + "=" * 75)
    print(f"FORECAST DAY {horizon}")
    print("=" * 75)

    target_col = f"target_day_{horizon}"
    target_date_col = f"target_date_{horizon}"

    df[target_col] = (
        df.groupby("location")["rainfall_mm"].shift(-horizon)
    )

    df[target_date_col] = (
        df.groupby("location")["date"].shift(-horizon)
    )

    model_data = df.dropna(
        subset=all_features + [target_col, target_date_col]
    ).copy()

    # Corrected split:
    # Training target date must be earlier than test cutoff.
    train_data = model_data[
        model_data[target_date_col] < split_date
    ]

    # Test inputs begin at the cutoff date.
    test_data = model_data[
        model_data["date"] >= split_date
    ]

    X_train = train_data[all_features]
    y_train = train_data[target_col]

    X_test = test_data[all_features]
    y_test = test_data[target_col]

    print(f"Training records: {len(train_data):,}")
    print(f"Testing records:  {len(test_data):,}")

    # -----------------------------------------------------
    # A. CORRECTED SINGLE-STAGE BASELINE
    # -----------------------------------------------------

    print("\nTraining corrected single-stage XGBoost...")

    baseline = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", create_regressor())
    ])

    baseline.fit(X_train, y_train)

    baseline_predictions = np.maximum(
        baseline.predict(X_test), 0
    )

    # -----------------------------------------------------
    # B. STAGE 1: RAIN OCCURRENCE CLASSIFIER
    # -----------------------------------------------------

    print("Training rainfall occurrence classifier...")

    rain_labels = (y_train >= 1.0).astype(int)

    classifier = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", create_classifier())
    ])

    classifier.fit(X_train, rain_labels)

    rain_probabilities = classifier.predict_proba(X_test)[:, 1]

    # -----------------------------------------------------
    # C. STAGE 2: RAINFALL AMOUNT REGRESSOR
    # -----------------------------------------------------

    print("Training rainfall amount regressor...")

    wet_training_rows = y_train >= 1.0

    X_wet_train = X_train.loc[wet_training_rows]
    y_wet_train = y_train.loc[wet_training_rows]

    amount_regressor = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", create_regressor())
    ])

    amount_regressor.fit(X_wet_train, y_wet_train)

    amount_predictions = np.maximum(
        amount_regressor.predict(X_test), 0
    )

    # -----------------------------------------------------
    # D. COMBINE BOTH STAGES
    # -----------------------------------------------------

    # Rainfall is predicted only when classifier probability
    # reaches 0.50. This threshold is fixed before test evaluation.

    two_stage_predictions = np.where(
        rain_probabilities >= 0.50,
        np.maximum(amount_predictions, 1.0),
        0.0
    )

    # -----------------------------------------------------
    # E. EVALUATION
    # -----------------------------------------------------

    actual = y_test.to_numpy()

    dry = actual < 1.0
    rainy = actual >= 1.0
    heavy = actual >= 10.0

    print("\n---------- OVERALL COMPARISON ----------")

    for name, predictions in [
        ("Corrected XGBoost", baseline_predictions),
        ("Two-stage model", two_stage_predictions)
    ]:

        mae = mean_absolute_error(actual, predictions)
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        r2 = r2_score(actual, predictions)

        print(f"\n{name}")
        print(f"MAE:  {mae:.4f} mm")
        print(f"RMSE: {rmse:.4f} mm")
        print(f"R2:   {r2:.4f}")

        results.append({
            "forecast_day": horizon,
            "model": name,
            "MAE_mm": mae,
            "RMSE_mm": rmse,
            "R2": r2
        })

    print("\n---------- TWO-STAGE RAINFALL ANALYSIS ----------")

    if rainy.any():
        print(
            "Actual rainy-day average:",
            f"{actual[rainy].mean():.3f} mm"
        )

        print(
            "Two-stage predicted average on actual rainy days:",
            f"{two_stage_predictions[rainy].mean():.3f} mm"
        )

        print(
            "Two-stage rainy-day MAE:",
            f"{mean_absolute_error(actual[rainy], two_stage_predictions[rainy]):.3f} mm"
        )

    if heavy.any():
        print(
            "Two-stage heavy-rain MAE (10+ mm):",
            f"{mean_absolute_error(actual[heavy], two_stage_predictions[heavy]):.3f} mm"
        )

    print("\n---------- RAINFALL EVENT DETECTION ----------")

    for threshold in [1, 10]:

        actual_event = actual >= threshold
        predicted_event = two_stage_predictions >= threshold

        print(f"\nThreshold: {threshold} mm")

        print(
            "Precision:",
            f"{precision_score(actual_event, predicted_event, zero_division=0):.4f}"
        )

        print(
            "Recall:   ",
            f"{recall_score(actual_event, predicted_event, zero_division=0):.4f}"
        )

        print(
            "F1 score: ",
            f"{f1_score(actual_event, predicted_event, zero_division=0):.4f}"
        )

# ---------------------------------------------------------
# 7. SAVE EXPERIMENT RESULTS
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

output_file = os.path.join(
    REPORT_DIR,
    "two_stage_experiment_results.csv"
)

results_df.to_csv(output_file, index=False)

print("\n" + "=" * 75)
print("FINAL EXPERIMENT COMPARISON")
print("=" * 75)

print(results_df.to_string(index=False))

print(f"\nResults saved to: {output_file}")

print("\nOriginal models and forecast reports were not modified.")
print("EXPERIMENT COMPLETED")
