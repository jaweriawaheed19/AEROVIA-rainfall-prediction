from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier, XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================================================
# CONFIGURATION
# =========================================================

DATA_FILE = Path("data/rainfall.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_FILE = MODEL_DIR / "rain_amount_xgb.pkl"
EVALUATION_FILE = REPORT_DIR / "rain_amount_model_evaluation.csv"
PREDICTIONS_FILE = REPORT_DIR / "rain_amount_test_predictions.csv"

RAIN_THRESHOLD_MM = 1.0
PROBABILITY_THRESHOLD = 0.30
TEST_START = pd.Timestamp("2023-01-01")

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

if MODEL_FILE.exists():
    raise FileExistsError(
        f"{MODEL_FILE} already exists. "
        "Stopping to protect the existing model."
    )

print("=" * 75)
print("PAKISTAN RAINFALL AMOUNT REGRESSION MODEL")
print("=" * 75)

# =========================================================
# 1. LOAD DATA
# =========================================================

df = pd.read_csv(DATA_FILE, parse_dates=["date"])
df = df.sort_values(["location", "date"]).reset_index(drop=True)

print("\nDataset records:", f"{len(df):,}")
print("Locations:", df["location"].nunique())
print("Date range:", df["date"].min().date(), "to", df["date"].max().date())

# =========================================================
# 2. FEATURE ENGINEERING
# =========================================================

group = df.groupby("location", sort=False)

for lag in [1, 2, 3, 4, 5, 6, 7]:
    df[f"rain_lag_{lag}"] = group["rainfall_mm"].shift(lag)

df["target_rainfall"] = group["rainfall_mm"].shift(-1)
df["target_date"] = group["date"].shift(-1)

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

model_data = df.dropna(
    subset=[
        "target_date",
        "target_rainfall",
        "rain_lag_1",
        "rain_lag_2",
        "rain_lag_3",
        "rain_lag_7",
    ]
).copy()

model_data["target_rainfall"] = model_data["target_rainfall"].clip(lower=0)

# =========================================================
# 3. CHRONOLOGICAL SPLIT
# =========================================================

final_train = model_data[
    model_data["target_date"] < TEST_START
].copy()

test = model_data[
    model_data["date"] >= TEST_START
].copy()

# The amount model learns only from rainy training observations.
rainy_train = final_train[
    final_train["target_rainfall"] >= RAIN_THRESHOLD_MM
].copy()

rainy_test = test[
    test["target_rainfall"] >= RAIN_THRESHOLD_MM
].copy()

if rainy_train.empty or rainy_test.empty:
    raise ValueError("Insufficient rainy observations for model training or testing.")

print("\n========== CHRONOLOGICAL SPLIT ==========")
print("Training records:", f"{len(final_train):,}")
print("Rainy training records:", f"{len(rainy_train):,}")
print("Testing records:", f"{len(test):,}")
print("Rainy testing records:", f"{len(rainy_test):,}")
print("Training target dates end:", final_train["target_date"].max().date())
print("Testing feature dates start:", test["date"].min().date())

# =========================================================
# 4. MODEL BUILDERS
# =========================================================

def build_regressor():
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


def build_occurrence_classifier():
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


# =========================================================
# 5. TRAIN AND EVALUATE AMOUNT MODEL
# =========================================================

print("\nTraining XGBoost rainfall amount regressor...")
print("Applying log1p transformation to rainfall amounts...")

amount_model = build_regressor()

amount_model.fit(
    rainy_train[feature_columns],
    np.log1p(rainy_train["target_rainfall"]),
)

# Predict all test rows. The occurrence classifier will decide
# whether the amount prediction should be used in the final system.
log_amount_predictions = amount_model.predict(test[feature_columns])

amount_predictions = np.maximum(
    0,
    np.expm1(log_amount_predictions)
)

# =========================================================
# 6. TRAIN TEMPORARY OCCURRENCE MODEL FOR FAIR TESTING
# =========================================================

print("\nTraining temporary occurrence classifier for combined evaluation...")

occurrence_train = final_train.copy()
occurrence_train["target"] = (
    occurrence_train["target_rainfall"] >= RAIN_THRESHOLD_MM
).astype(int)

occurrence_test = test.copy()
occurrence_test["target"] = (
    occurrence_test["target_rainfall"] >= RAIN_THRESHOLD_MM
).astype(int)

classifier = build_occurrence_classifier()

classifier.fit(
    occurrence_train[feature_columns],
    occurrence_train["target"],
)

rain_probabilities = classifier.predict_proba(
    occurrence_test[feature_columns]
)[:, 1]

predicted_rain = (
    rain_probabilities >= PROBABILITY_THRESHOLD
).astype(int)

actual_amounts = test["target_rainfall"].to_numpy()
actual_occurrence = occurrence_test["target"].to_numpy()

# Final two-stage output: zero if the classifier predicts dry.
combined_predictions = amount_predictions * predicted_rain

# =========================================================
# 7. HISTORICAL BASELINE
# =========================================================

# Calculate average rainfall on rainy days using training data only.
# Use location + month, with location and global fallbacks.

rainy_monthly_means = (
    rainy_train.groupby(
        ["location", "month"]
    )["target_rainfall"].mean()
)

rainy_location_means = (
    rainy_train.groupby("location")["target_rainfall"].mean()
)

global_rainy_mean = float(
    rainy_train["target_rainfall"].mean()
)

baseline_amounts = []

for row in test.itertuples():
    value = rainy_monthly_means.get(
        (row.location, row.month),
        np.nan
    )

    if pd.isna(value):
        value = rainy_location_means.get(
            row.location,
            global_rainy_mean
        )

    baseline_amounts.append(float(value))

baseline_amounts = np.array(baseline_amounts)

baseline_combined = baseline_amounts * predicted_rain

# =========================================================
# 8. EVALUATION
# =========================================================

def regression_metrics(actual, predicted):
    return {
        "MAE_mm": mean_absolute_error(actual, predicted),
        "RMSE_mm": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
    }


evaluation_rows = []

# Conditional amount prediction: only actual rainy test days.
actual_rainy_amounts = actual_amounts >= RAIN_THRESHOLD_MM

for name, predictions in [
    ("XGBoost Regressor - Rainy Days Only", amount_predictions),
    ("Historical Rainy-Day Average - Rainy Days Only", baseline_amounts),
]:
    metrics = regression_metrics(
        actual_amounts[actual_rainy_amounts],
        predictions[actual_rainy_amounts],
    )

    evaluation_rows.append({
        "Evaluation": name,
        "Records": int(actual_rainy_amounts.sum()),
        **metrics,
    })

# Full two-stage system: all test days.
for name, predictions in [
    ("Two-Stage XGBoost System", combined_predictions),
    ("Two-Stage Historical Average Baseline", baseline_combined),
]:
    metrics = regression_metrics(
        actual_amounts,
        predictions,
    )

    evaluation_rows.append({
        "Evaluation": name,
        "Records": len(actual_amounts),
        **metrics,
    })

evaluation_df = pd.DataFrame(evaluation_rows)
evaluation_df.to_csv(EVALUATION_FILE, index=False)

print("\n========== AMOUNT MODEL EVALUATION ==========")
print(evaluation_df.to_string(index=False))

print("\n========== OCCURRENCE GATING ==========")
print("Actual rainy test cases:", int(actual_occurrence.sum()))
print("Predicted rainy test cases:", int(predicted_rain.sum()))
print(
    "Rain occurrence accuracy:",
    f"{(predicted_rain == actual_occurrence).mean():.4f}"
)

# =========================================================
# 9. SAVE TEST PREDICTIONS
# =========================================================

prediction_df = test[
    ["date", "target_date", "location", "province", "target_rainfall"]
].copy()

prediction_df = prediction_df.rename(
    columns={
        "date": "feature_date",
        "target_date": "forecast_date",
        "target_rainfall": "actual_rainfall_mm",
    }
)

prediction_df["rain_probability"] = rain_probabilities
prediction_df["predicted_rain"] = predicted_rain
prediction_df["regression_amount_mm"] = amount_predictions
prediction_df["final_predicted_amount_mm"] = combined_predictions
prediction_df["historical_baseline_amount_mm"] = baseline_combined

prediction_df.to_csv(PREDICTIONS_FILE, index=False)

# =========================================================
# 10. TRAIN FINAL PRODUCTION AMOUNT MODEL
# =========================================================

print("\nTraining final production amount model on all available rainy records...")

all_rainy_data = model_data[
    model_data["target_rainfall"] >= RAIN_THRESHOLD_MM
].copy()

production_amount_model = build_regressor()

production_amount_model.fit(
    all_rainy_data[feature_columns],
    np.log1p(all_rainy_data["target_rainfall"]),
)

model_package = {
    "model": production_amount_model,
    "feature_columns": feature_columns,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "rainfall_target_mm": RAIN_THRESHOLD_MM,
    "target_transformation": "log1p",
    "prediction_inverse_transformation": "expm1",
    "model_name": "XGBoost Rainfall Amount Regressor",
    "training_records": len(all_rainy_data),
    "training_target_end": str(model_data["target_date"].max().date()),
    "dataset_last_date": str(df["date"].max().date()),
    "forecast_horizon_days": 1,
}

joblib.dump(model_package, MODEL_FILE)

print("\n========== SAVED FILES ==========")
print("Production model:", MODEL_FILE)
print("Evaluation report:", EVALUATION_FILE)
print("Test predictions:", PREDICTIONS_FILE)

print("\n" + "=" * 75)
print("RAINFALL AMOUNT MODEL TRAINING COMPLETED")
print("=" * 75)
