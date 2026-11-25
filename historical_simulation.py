from pathlib import Path

import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


DATA_FILE = Path("data/rainfall.csv")

RAIN_THRESHOLD_MM = 1.0
PROBABILITY_THRESHOLD = 0.30


def build_model(numeric_features, categorical_features):

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median"))
                ]),
                numeric_features,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    classifier = XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def run_historical_simulation(selected_date):

    selected_date = pd.Timestamp(selected_date).normalize()

    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    df = df.sort_values(
        ["location", "date"]
    ).reset_index(drop=True)

    group = df.groupby("location", sort=False)

    # Use the exact rainfall lag features from the final classifier.
    for lag in [1, 2, 3, 4, 5, 6, 7]:
        df[f"rain_lag_{lag}"] = group["rainfall_mm"].shift(lag)

    # The target is rainfall on the following day.
    df["target_rainfall"] = group["rainfall_mm"].shift(-1)
    df["target_date"] = group["date"].shift(-1)

    df["target"] = (
        df["target_rainfall"] >= RAIN_THRESHOLD_MM
    ).astype(int)

    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    feature_columns = [
        "rain_lag_1",
        "rain_lag_2",
        "rain_lag_3",
        "rain_lag_7",
        "temperature_c",
        "relative_humidity_percent",
        "wind_speed_mps",
        "month",
        "day",
        "location",
        "province",
    ]

    numeric_features = [
        "rain_lag_1",
        "rain_lag_2",
        "rain_lag_3",
        "rain_lag_7",
        "temperature_c",
        "relative_humidity_percent",
        "wind_speed_mps",
        "month",
        "day",
    ]

    categorical_features = ["location", "province"]

    required_columns = [
        "target_date",
        "rain_lag_1",
        "rain_lag_2",
        "rain_lag_3",
        "rain_lag_7",
    ]

    model_data = df.dropna(
        subset=required_columns
    ).copy()

    # Train only on labels whose target dates are before the
    # selected simulation date.
    training_data = model_data[
        model_data["target_date"] < selected_date
    ].copy()

    # The prediction features must come from the day before
    # the selected target date.
    prediction_rows = model_data[
        model_data["target_date"] == selected_date
    ].copy()

    if training_data.empty:
        raise ValueError(
            "There is not enough historical training data "
            "before the selected date."
        )

    if prediction_rows.empty:
        raise ValueError(
            "The selected date does not have sufficient previous-day "
            "observations and rainfall history for prediction."
        )

    if training_data["target"].nunique() < 2:
        raise ValueError(
            "The training period contains only one rainfall class. "
            "Choose a later historical date."
        )

    # Train the temporary model.
    model = build_model(
        numeric_features,
        categorical_features
    )

    model.fit(
        training_data[feature_columns],
        training_data["target"].astype(int),
    )

    # Make predictions before accessing actual target rainfall.
    probabilities = model.predict_proba(
        prediction_rows[feature_columns]
    )[:, 1]

    predicted_rain = (
        probabilities >= PROBABILITY_THRESHOLD
    ).astype(int)

    # Calculate the historical amount estimate using only
    # observations strictly before the selected date.
    available_history = df[
        df["date"] < selected_date
    ].copy()

    available_history = available_history[
        available_history["date"].dt.month == selected_date.month
    ]

    monthly_averages = available_history.groupby(
        "location"
    )["rainfall_mm"].mean()

    location_averages = df[
        df["date"] < selected_date
    ].groupby("location")["rainfall_mm"].mean()

    overall_average = (
        df.loc[
            df["date"] < selected_date,
            "rainfall_mm"
        ].mean()
    )

    # Actual rainfall is accessed only after predictions are made.
    results = prediction_rows[
        [
            "location",
            "province",
            "latitude",
            "longitude",
            "target_date",
            "target_rainfall",
        ]
    ].copy()

    results["rain_probability"] = probabilities
    results["predicted_rain"] = predicted_rain
    results["prediction"] = np.where(
        predicted_rain == 1,
        "Rain Expected",
        "No Rain Expected",
    )

    results["estimated_rainfall_mm"] = results[
        "location"
    ].map(monthly_averages)

    results["estimated_rainfall_mm"] = results[
        "estimated_rainfall_mm"
    ].fillna(
        results["location"].map(location_averages)
    ).fillna(overall_average)

    results["actual_rainfall_mm"] = results[
        "target_rainfall"
    ]

    results["actual_rain"] = (
        results["actual_rainfall_mm"] >= RAIN_THRESHOLD_MM
    ).astype(int)

    results["amount_absolute_error_mm"] = (
        results["estimated_rainfall_mm"]
        - results["actual_rainfall_mm"]
    ).abs()

    results["occurrence_correct"] = (
        results["predicted_rain"] == results["actual_rain"]
    )

    results = results.rename(
        columns={"target_date": "simulation_date"}
    )

    results["rain_probability_percent"] = (
        results["rain_probability"] * 100
    )

    results["amount_absolute_error_mm"] = (
        results["amount_absolute_error_mm"].round(3)
    )

    results["estimated_rainfall_mm"] = (
        results["estimated_rainfall_mm"].round(3)
    )

    return results.sort_values("location").reset_index(drop=True)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Run a leakage-safe historical rainfall simulation."
    )

    parser.add_argument(
        "--date",
        required=True,
        help="Historical prediction date in YYYY-MM-DD format.",
    )

    args = parser.parse_args()

    results = run_historical_simulation(args.date)

    print("=" * 70)
    print("HISTORICAL RAINFALL SIMULATION")
    print("=" * 70)

    print("Simulation date:", args.date)
    print("Locations:", len(results))
    print("Rain occurrence accuracy:",
          f"{results['occurrence_correct'].mean():.4f}")
    print("Amount MAE:",
          f"{results['amount_absolute_error_mm'].mean():.4f} mm")

    print("\nRESULTS")
    print(
        results[
            [
                "location",
                "rain_probability_percent",
                "prediction",
                "estimated_rainfall_mm",
                "actual_rainfall_mm",
                "occurrence_correct",
            ]
        ].to_string(index=False)
    )
