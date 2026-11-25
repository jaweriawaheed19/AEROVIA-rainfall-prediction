from pathlib import Path

import numpy as np
import pandas as pd

from xgboost import XGBClassifier, XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


DATA_FILE = Path("data/rainfall.csv")

RAIN_THRESHOLD_MM = 1.0
PROBABILITY_THRESHOLD = 0.30


def build_preprocessor(numeric_features, categorical_features):

    return ColumnTransformer(
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


def build_classifier(numeric_features, categorical_features):

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features
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


def build_regressor(numeric_features, categorical_features):

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features
    )

    regressor = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor),
    ])


def run_historical_simulation(selected_date):

    selected_date = pd.Timestamp(selected_date).normalize()

    df = pd.read_csv(DATA_FILE, parse_dates=["date"])

    df = df.sort_values(
        ["location", "date"]
    ).reset_index(drop=True)

    group = df.groupby("location", sort=False)

    # Create rainfall lag features.
    for lag in [1, 2, 3, 4, 5, 6, 7]:
        df[f"rain_lag_{lag}"] = group["rainfall_mm"].shift(lag)

    # Next-day rainfall target.
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

    # Training data must have target dates before
    # the selected simulation date.
    training_data = model_data[
        model_data["target_date"] < selected_date
    ].copy()

    # Prediction features correspond to the day
    # immediately before the selected target date.
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

    # =====================================================
    # TRAIN OCCURRENCE CLASSIFIER
    # =====================================================

    classifier = build_classifier(
        numeric_features,
        categorical_features
    )

    classifier.fit(
        training_data[feature_columns],
        training_data["target"].astype(int),
    )

    # =====================================================
    # TRAIN AMOUNT REGRESSOR
    # Rainy observations only
    # =====================================================

    rainy_training_data = training_data[
        training_data["target_rainfall"] >= RAIN_THRESHOLD_MM
    ].copy()

    if rainy_training_data.empty:
        raise ValueError(
            "No rainy training observations are available "
            "before the selected simulation date."
        )

    amount_regressor = build_regressor(
        numeric_features,
        categorical_features
    )

    # Log transformation reduces the effect of extreme rainfall.
    amount_regressor.fit(
        rainy_training_data[feature_columns],
        np.log1p(rainy_training_data["target_rainfall"]),
    )

    # =====================================================
    # MAKE PREDICTIONS BEFORE ACCESSING ACTUAL TARGETS
    # =====================================================

    probabilities = classifier.predict_proba(
        prediction_rows[feature_columns]
    )[:, 1]

    predicted_rain = (
        probabilities >= PROBABILITY_THRESHOLD
    ).astype(int)

    predicted_log_amount = amount_regressor.predict(
        prediction_rows[feature_columns]
    )

    predicted_amount = np.expm1(
        predicted_log_amount
    )

    predicted_amount = np.maximum(
        predicted_amount,
        0
    )

    # If classifier predicts no rain, final rainfall is zero.
    final_amount = np.where(
        predicted_rain == 1,
        predicted_amount,
        0.0
    )

    # =====================================================
    # RESULTS
    # Actual rainfall is accessed only after predictions.
    # =====================================================

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

    # Keep the existing column name so the dashboard
    # remains compatible during this integration.
    results["estimated_rainfall_mm"] = final_amount

    results["actual_rainfall_mm"] = (
        results["target_rainfall"]
    )

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

    return results.sort_values(
        "location"
    ).reset_index(drop=True)


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

    print(
        "Rain occurrence accuracy:",
        f"{results['occurrence_correct'].mean():.4f}"
    )

    print(
        "Amount MAE:",
        f"{results['amount_absolute_error_mm'].mean():.4f} mm"
    )

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
