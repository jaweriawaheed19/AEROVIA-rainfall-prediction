from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
)

# =========================================================
# CONFIGURATION
# =========================================================

DATA_FILE = Path("data/rainfall.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_FILE = MODEL_DIR / "rain_occurrence_xgb.pkl"
FORECAST_FILE = REPORT_DIR / "rain_occurrence_forecast.csv"
EVALUATION_FILE = REPORT_DIR / "rain_occurrence_model_evaluation.csv"

RAIN_THRESHOLD_MM = 1.0
PROBABILITY_THRESHOLD = 0.30
TEST_START = pd.Timestamp("2023-01-01")

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

print("=" * 75)
print("FINAL PAKISTAN RAINFALL OCCURRENCE MODEL")
print("=" * 75)

# =========================================================
# 1. LOAD DATA
# =========================================================

df = pd.read_csv(DATA_FILE, parse_dates=["date"])
df = df.sort_values(["location", "date"]).reset_index(drop=True)

print("\nDATASET")
print("Records:", f"{len(df):,}")
print("Locations:", df["location"].nunique())
print("Date range:", df["date"].min().date(), "to", df["date"].max().date())

# =========================================================
# 2. FEATURE ENGINEERING
# =========================================================

print("\nCreating features...")

group = df.groupby("location", sort=False)

for lag in [1, 2, 3, 4, 5, 6, 7]:
    df[f"rain_lag_{lag}"] = group["rainfall_mm"].shift(lag)

# The target is rainfall occurrence on the following day.
df["target_rainfall"] = group["rainfall_mm"].shift(-1)
df["target_date"] = group["date"].shift(-1)

df["target"] = (
    df["target_rainfall"] >= RAIN_THRESHOLD_MM
).astype(int)

df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day

# Exact classifier feature set from the selected experiment.
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

# Keep only rows with a known next-day target and required rainfall history.
model_data = df.dropna(
    subset=[
        "target_date",
        "rain_lag_1",
        "rain_lag_2",
        "rain_lag_3",
        "rain_lag_7",
    ]
).copy()

model_data["target"] = model_data["target"].astype(int)

print("Usable labelled records:", f"{len(model_data):,}")
print("Rain occurrence rate:", f"{model_data['target'].mean():.4f}")

# =========================================================
# 3. CHRONOLOGICAL EVALUATION SPLIT
# =========================================================

final_train = model_data[
    model_data["target_date"] < TEST_START
].copy()

test = model_data[
    model_data["date"] >= TEST_START
].copy()

print("\nCHRONOLOGICAL EVALUATION")
print("Training records:", f"{len(final_train):,}")
print("Testing records:", f"{len(test):,}")
print("Training target dates end:", final_train["target_date"].max().date())
print("Testing feature dates start:", test["date"].min().date())

# =========================================================
# 4. MODEL BUILDER
# =========================================================

def build_model():
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
# 5. TRAIN EVALUATION MODEL
# =========================================================

print("\nTraining evaluation model...")

evaluation_model = build_model()

evaluation_model.fit(
    final_train[feature_columns],
    final_train["target"],
)

test_probabilities = evaluation_model.predict_proba(
    test[feature_columns]
)[:, 1]

test_predictions = (
    test_probabilities >= PROBABILITY_THRESHOLD
).astype(int)

y_test = test["target"].to_numpy()

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1],
).ravel()

evaluation = {
    "Model": "Unweighted XGBoost Rainfall Occurrence Classifier",
    "Target": "Next-day rainfall >= 1 mm",
    "Probability_Threshold": PROBABILITY_THRESHOLD,
    "Precision": precision_score(y_test, test_predictions, zero_division=0),
    "Recall": recall_score(y_test, test_predictions, zero_division=0),
    "F1": f1_score(y_test, test_predictions, zero_division=0),
    "Accuracy": accuracy_score(y_test, test_predictions),
    "Balanced_Accuracy": balanced_accuracy_score(y_test, test_predictions),
    "ROC_AUC": roc_auc_score(y_test, test_probabilities),
    "PR_AUC": average_precision_score(y_test, test_probabilities),
    "Brier_Score": brier_score_loss(y_test, test_probabilities),
    "Log_Loss": log_loss(y_test, test_probabilities),
    "Actual_Rain_Rate": y_test.mean(),
    "Predicted_Rain_Rate": test_predictions.mean(),
    "Average_Probability": test_probabilities.mean(),
    "True_Positives": int(tp),
    "False_Positives": int(fp),
    "True_Negatives": int(tn),
    "False_Negatives": int(fn),
    "Training_Records": len(final_train),
    "Testing_Records": len(test),
    "Training_Target_End": str(final_train["target_date"].max().date()),
    "Test_Feature_Start": str(test["date"].min().date()),
    "Test_Feature_End": str(test["date"].max().date()),
}

evaluation_df = pd.DataFrame([evaluation])
evaluation_df.to_csv(EVALUATION_FILE, index=False)

print("\n========== HELD-OUT TEST RESULTS ==========")

for key, value in evaluation.items():
    if isinstance(value, (float, np.floating)):
        print(f"{key:25}: {value:.4f}")
    else:
        print(f"{key:25}: {value}")

# =========================================================
# 6. TRAIN FINAL PRODUCTION MODEL
# =========================================================

print("\n" + "=" * 75)
print("TRAINING FINAL PRODUCTION MODEL")
print("=" * 75)

production_model = build_model()

production_model.fit(
    model_data[feature_columns],
    model_data["target"],
)

# Save model and all inference settings together.
model_package = {
    "model": production_model,
    "feature_columns": feature_columns,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "rainfall_target_mm": RAIN_THRESHOLD_MM,
    "probability_threshold": PROBABILITY_THRESHOLD,
    "model_name": "Unweighted XGBoost Rainfall Occurrence Classifier",
    "training_records": len(model_data),
    "training_target_end": str(model_data["target_date"].max().date()),
    "dataset_last_date": str(df["date"].max().date()),
    "forecast_horizon_days": 1,
}

joblib.dump(model_package, MODEL_FILE)

print("Final model saved:", MODEL_FILE)

# =========================================================
# 7. GENERATE LATEST AVAILABLE-DATA PREDICTIONS
# =========================================================

print("\nGenerating next-day predictions for all locations...")

forecast_rows = []

for location in sorted(df["location"].unique()):

    location_data = (
        df[df["location"] == location]
        .sort_values("date")
        .copy()
    )

    latest = location_data.iloc[-1]

    if len(location_data) < 7:
        print("Skipping location with insufficient history:", location)
        continue

    row = {
        "rain_lag_1": location_data["rainfall_mm"].iloc[-1],
        "rain_lag_2": location_data["rainfall_mm"].iloc[-2],
        "rain_lag_3": location_data["rainfall_mm"].iloc[-3],
        "rain_lag_7": location_data["rainfall_mm"].iloc[-7],
        "temperature_c": latest["temperature_c"],
        "relative_humidity_percent": latest["relative_humidity_percent"],
        "wind_speed_mps": latest["wind_speed_mps"],
        "month": latest["date"].month,
        "day": latest["date"].day,
        "location": location,
        "province": latest["province"],
    }

    X_future = pd.DataFrame([row])[feature_columns]

    probability = float(
        production_model.predict_proba(X_future)[0, 1]
    )

    predicted_rain = int(probability >= PROBABILITY_THRESHOLD)

    forecast_date = latest["date"] + pd.Timedelta(days=1)

    forecast_rows.append({
        "location": location,
        "province": latest["province"],
        "latitude": latest["latitude"],
        "longitude": latest["longitude"],
        "last_observed_date": latest["date"].date(),
        "forecast_date": forecast_date.date(),
        "rain_probability": round(probability, 4),
        "probability_percent": round(probability * 100, 2),
        "predicted_rain": predicted_rain,
        "prediction": "Rain Expected" if predicted_rain else "No Rain Expected",
        "alert": "Rain Alert" if predicted_rain else "No Alert",
    })

forecast_df = pd.DataFrame(forecast_rows)

forecast_df.to_csv(FORECAST_FILE, index=False)

print("\n========== FORECAST PREVIEW ==========")
print(forecast_df.to_string(index=False))

print("\nLocations forecast:", len(forecast_df))
print("Rain alerts:", int(forecast_df["predicted_rain"].sum()))
print("No-rain predictions:", int((forecast_df["predicted_rain"] == 0).sum()))
print("Last observed date:", df["date"].max().date())
print("Forecast date:", forecast_df["forecast_date"].min())

print("\nSaved forecast:", FORECAST_FILE)
print("Saved evaluation:", EVALUATION_FILE)
print("Saved production model:", MODEL_FILE)

print("\n" + "=" * 75)
print("FINAL RAINFALL OCCURRENCE MODEL TRAINING COMPLETED")
print("=" * 75)
