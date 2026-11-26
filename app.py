from pathlib import Path
from datetime import date

import numpy as np
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium

from streamlit_folium import st_folium
from folium.plugins import MarkerCluster

from historical_simulation import run_historical_simulation


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Pakistan Rainfall Prediction",
    page_icon="🌧️",
    layout="wide"
)

DATA_FILE = Path("data/rainfall.csv")
FORECAST_FILE = Path("reports/rain_occurrence_forecast.csv")
EVALUATION_FILE = Path("reports/rain_occurrence_model_evaluation.csv")
AMOUNT_MODEL_FILE = Path("models/rain_amount_xgb.pkl")

PROBABILITY_THRESHOLD = 0.30

@st.cache_data(show_spinner=False)
def cached_historical_simulation(selected_date):
    return run_historical_simulation(selected_date)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    historical = pd.read_csv(DATA_FILE)
    forecast = pd.read_csv(FORECAST_FILE)
    evaluation = pd.read_csv(EVALUATION_FILE)

    historical["date"] = pd.to_datetime(historical["date"])
    forecast["forecast_date"] = pd.to_datetime(forecast["forecast_date"])
    forecast["last_observed_date"] = pd.to_datetime(
        forecast["last_observed_date"]
    )

    # =====================================================
    # TRAINED RAINFALL AMOUNT MODEL
    # =====================================================

    historical["rainfall_mm"] = pd.to_numeric(
        historical["rainfall_mm"], errors="coerce"
    )

    amount_package = joblib.load(AMOUNT_MODEL_FILE)
    amount_model = amount_package["model"]
    feature_columns = amount_package["feature_columns"]

    amount_input_rows = []

    for _, forecast_row in forecast.iterrows():

        location_history = historical[
            (historical["location"] == forecast_row["location"])
            & (
                historical["date"]
                <= forecast_row["last_observed_date"]
            )
        ].sort_values("date")

        if location_history.empty:
            raise ValueError(
                f"No historical data found for {forecast_row['location']}."
            )

        latest = location_history.iloc[-1]

        # The model uses rainfall lags from the latest
        # observation available before the forecast date.
        rainfall_values = location_history[
            "rainfall_mm"
        ].tolist()

        feature_row = {
            "temperature_c": latest["temperature_c"],
            "relative_humidity_percent": latest[
                "relative_humidity_percent"
            ],
            "wind_speed_mps": latest["wind_speed_mps"],
            "month": latest["date"].month,
            "day": latest["date"].day,
            "location": forecast_row["location"],
            "province": forecast_row["province"],
        }

        for lag in [1, 2, 3, 7]:
            feature_row[f"rain_lag_{lag}"] = (
                rainfall_values[-(lag + 1)]
                if len(rainfall_values) >= lag + 1
                else np.nan
            )

        amount_input_rows.append(feature_row)

    amount_features = pd.DataFrame(
        amount_input_rows,
        columns=feature_columns
    )

    predicted_log_amounts = amount_model.predict(
        amount_features
    )

    predicted_amounts = np.expm1(
        predicted_log_amounts
    )

    predicted_amounts = np.maximum(
        predicted_amounts,
        0
    )

    # Classifier controls whether the predicted amount
    # is displayed as rainfall or zero.
    forecast["predicted_rainfall_mm"] = np.where(
        forecast["predicted_rain"] == 1,
        predicted_amounts,
        0.0
    )

    # Keep the previous column name temporarily so that
    # existing dashboard sections remain compatible.
    forecast["estimated_rainfall_mm"] = (
        forecast["predicted_rainfall_mm"]
    )

    return historical, forecast, evaluation


try:
    historical, forecast, evaluation = load_data()

except Exception as error:
    st.error(f"Unable to load project data: {error}")
    st.stop()


# =========================================================
# HEADER
# =========================================================

st.title("Pakistan Rainfall Prediction System")

st.markdown(
    "### Machine Learning Based Next-Day Rainfall Prediction"
)

st.markdown(
    """
    This system uses a **two-stage XGBoost machine learning system** to
    predict next-day rainfall at 22 selected locations across Pakistan.

    - **Classifier:** predicts rainfall occurrence and probability.
    - **Regressor:** estimates rainfall amount in millimetres when rain is predicted.
    """
)

latest_observation = historical["date"].max()
forecast_date = forecast["forecast_date"].max()

st.info(
    f"Historical dataset available through **{latest_observation:%d %B %Y}**. "
    f"The generated next-day prediction is for **{forecast_date:%d %B %Y}**. "
    "This is an offline historical-data prediction, not a live weather forecast."
)

if forecast_date.date() < date.today():
    st.warning(
        "These predictions are historical and are not valid as current "
        "weather forecasts. Updated weather observations are required "
        "to generate predictions for today's date."
    )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Dashboard Controls")

province_options = (
    ["All Provinces"] +
    sorted(forecast["province"].dropna().unique().tolist())
)

selected_province = st.sidebar.selectbox(
    "Select Province",
    province_options
)

if selected_province == "All Provinces":
    available_locations = sorted(forecast["location"].unique())
else:
    available_locations = sorted(
        forecast.loc[
            forecast["province"] == selected_province,
            "location"
        ].unique()
    )

selected_location = st.sidebar.selectbox(
    "Select Location",
    available_locations
)

st.sidebar.divider()

st.sidebar.caption(
    "Prediction target: Next-day rainfall of at least 1 mm"
)

st.sidebar.caption(
    f"Rain alert threshold: {PROBABILITY_THRESHOLD:.0%} probability"
)


# =========================================================
# FILTER DATA
# =========================================================

location_forecast = forecast[
    forecast["location"] == selected_location
].copy()

location_history = historical[
    historical["location"] == selected_location
].sort_values("date").copy()

if selected_province == "All Provinces":
    filtered_forecast = forecast.copy()
else:
    filtered_forecast = forecast[
        forecast["province"] == selected_province
    ].copy()


# =========================================================
# DASHBOARD METRICS
# =========================================================

total_locations = filtered_forecast["location"].nunique()

total_alerts = int(
    filtered_forecast["predicted_rain"].sum()
)

highest_probability = filtered_forecast["rain_probability"].max()

average_probability = filtered_forecast["rain_probability"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "📍 Locations Covered",
    total_locations
)

col2.metric(
    "🚨 Rain Alerts",
    total_alerts
)

col3.metric(
    "⬆️ Highest Rain Probability",
    f"{highest_probability * 100:.2f}%"
)

col4.metric(
    "📊 Average Rain Probability",
    f"{average_probability * 100:.2f}%"
)

st.divider()


# =========================================================
# SELECTED LOCATION PREDICTION
# =========================================================

st.subheader(f"Next-Day Prediction: {selected_location}")

selected_row = location_forecast.iloc[0]

probability = float(selected_row["rain_probability"])

province = selected_row["province"]

prediction_date = selected_row["forecast_date"]

last_observed = selected_row["last_observed_date"]

st.caption(
    f"Province: {province} | "
    f"Last observation: {last_observed:%d %B %Y} | "
    f"Prediction date: {prediction_date:%d %B %Y}"
)

prediction_col1, prediction_col2, prediction_col3, prediction_col4 = st.columns(4)

prediction_col1.metric(
    "Rain Probability",
    f"{probability * 100:.2f}%"
)

prediction_col2.metric(
    "Predicted Status",
    selected_row["prediction"]
)

prediction_col3.metric(
    "Alert Status",
    selected_row["alert"]
)

prediction_col4.metric(
    "Predicted Rainfall Amount",
    f"{selected_row['estimated_rainfall_mm']:.2f} mm"
)

st.caption(
    "Rainfall amount is predicted by a trained XGBoost regression model. "
    "The amount prediction is set to zero when the occurrence classifier "
    "predicts no rain."
)

if probability >= PROBABILITY_THRESHOLD:
    st.warning(
        f"Rain alert: {selected_location} has a predicted rainfall "
        f"probability of {probability * 100:.2f}%."
    )
else:
    st.success(
        f"No rain alert: the predicted rainfall probability is "
        f"{probability * 100:.2f}%, below the 30% threshold."
    )


# =========================================================
# PROBABILITY GAUGE
# =========================================================

st.subheader("Rainfall Probability")

gauge = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%"},
        title={"text": f"Rain Probability — {selected_location}"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#2878B5"},
            "steps": [
                {"range": [0, 10], "color": "#D9EAD3"},
                {"range": [10, 20], "color": "#FFF2CC"},
                {"range": [20, 30], "color": "#FCE5CD"},
                {"range": [30, 100], "color": "#F4CCCC"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 30,
            },
        },
    )
)

gauge.update_layout(
    height=300,
    margin=dict(t=60, b=20, l=30, r=30)
)

st.plotly_chart(gauge, width="stretch")


# =========================================================
# HISTORICAL RAINFALL CHART
# =========================================================

st.divider()
st.subheader("Historical Rainfall")

history_period = st.selectbox(
    "Historical chart period",
    [
        "Last 30 days",
        "Last 90 days",
        "Last 365 days",
        "All data"
    ]
)

if history_period == "Last 30 days":
    history_chart = location_history.tail(30)

elif history_period == "Last 90 days":
    history_chart = location_history.tail(90)

elif history_period == "Last 365 days":
    history_chart = location_history.tail(365)

else:
    history_chart = location_history

fig_history = px.line(
    history_chart,
    x="date",
    y="rainfall_mm",
    labels={
        "date": "Date",
        "rainfall_mm": "Observed Rainfall (mm)"
    },
    title=f"Historical Rainfall — {selected_location}"
)

fig_history.update_layout(
    xaxis_title="Date",
    yaxis_title="Rainfall (mm)"
)

st.plotly_chart(fig_history, width="stretch")


# =========================================================
# PAKISTAN RAINFALL PROBABILITY MAP
# =========================================================

st.divider()
st.subheader("🇵🇰 Pakistan Rainfall Probability Map")

map_data = filtered_forecast.copy()

map_data["latitude"] = pd.to_numeric(
    map_data["latitude"],
    errors="coerce"
)

map_data["longitude"] = pd.to_numeric(
    map_data["longitude"],
    errors="coerce"
)

map_data = map_data.dropna(
    subset=[
        "latitude",
        "longitude",
        "rain_probability"
    ]
)

pakistan_map = folium.Map(
    location=[30.3753, 69.3451],
    zoom_start=5,
    tiles="OpenStreetMap",
    control_scale=True
)

marker_cluster = MarkerCluster(
    name="Rainfall Probability Locations",
    show=True
).add_to(pakistan_map)


def probability_color(probability):

    if probability < 0.10:
        return "green"

    elif probability < 0.20:
        return "yellow"

    elif probability < PROBABILITY_THRESHOLD:
        return "orange"

    return "red"


for row in map_data.itertuples():

    probability = float(row.rain_probability)

    color = probability_color(probability)

    popup_html = f"""
    <div style="font-family:Arial; width:210px;">
        <h4 style="margin-bottom:8px;">{row.location}</h4>
        <b>Province:</b> {row.province}<br>
        <b>Prediction Date:</b> {row.forecast_date:%d %B %Y}<br>
        <b>Rain Probability:</b> {probability * 100:.2f}%<br>
        <b>Predicted Rainfall Amount:</b> {row.estimated_rainfall_mm:.2f} mm<br>
        <b>Prediction:</b> {row.prediction}<br>
        <b>Alert:</b> {row.alert}<br>
        <b>Last Observation:</b> {row.last_observed_date:%d %B %Y}
    </div>
    """

    folium.CircleMarker(
        location=[row.latitude, row.longitude],
        radius=8 + probability * 12,
        color=color,
        weight=2,
        fill=True,
        fill_color=color,
        fill_opacity=0.75,
        tooltip=(
            f"{row.location}: "
            f"{probability * 100:.2f}% rain probability"
        ),
        popup=folium.Popup(
            popup_html,
            max_width=270
        )
    ).add_to(marker_cluster)


if not map_data.empty:

    bounds = [
        [
            map_data["latitude"].min(),
            map_data["longitude"].min()
        ],
        [
            map_data["latitude"].max(),
            map_data["longitude"].max()
        ]
    ]

    pakistan_map.fit_bounds(
        bounds,
        padding=[35, 35]
    )


# Map legend
legend_html = """
<div style="
    position: fixed;
    bottom: 25px;
    right: 25px;
    z-index: 9999;
    background: white;
    padding: 12px 16px;
    border: 1px solid #999;
    border-radius: 7px;
    font-size: 13px;
    box-shadow: 0 1px 5px rgba(0,0,0,0.3);
    color: #222;
">
    <b>Rain Probability</b><br>
    <span style="display:inline-block;width:10px;height:10px;background:green;border-radius:50%;margin-right:7px;"></span>
    Low: Below 10%<br>
    <span style="display:inline-block;width:10px;height:10px;background:#d6b900;border-radius:50%;margin-right:7px;"></span>
    Moderate: 10–20%<br>
    <span style="display:inline-block;width:10px;height:10px;background:orange;border-radius:50%;margin-right:7px;"></span>
    Elevated: 20–30%<br>
    <span style="display:inline-block;width:10px;height:10px;background:red;border-radius:50%;margin-right:7px;"></span>
    Rain Alert: 30% or more
</div>
"""

pakistan_map.get_root().html.add_child(
    folium.Element(legend_html)
)

st.caption(
    f"Showing {len(map_data)} locations. "
    "Click a marker to view its rainfall probability and prediction status."
)

st_folium(
    pakistan_map,
    width=None,
    height=620,
    use_container_width=True,
    returned_objects=[]
)


# =========================================================
# ALL LOCATIONS PREDICTION TABLE
# =========================================================

st.divider()
st.subheader("All Locations — Next-Day Predictions")

display_forecast = filtered_forecast[
    [
        "location",
        "province",
        "forecast_date",
        "probability_percent",
        "estimated_rainfall_mm",
        "prediction",
        "alert",
    ]
].copy()

display_forecast["forecast_date"] = (
    display_forecast["forecast_date"].dt.strftime("%Y-%m-%d")
)

display_forecast["estimated_rainfall_mm"] = (
    display_forecast["estimated_rainfall_mm"].round(2)
)

display_forecast.columns = [
    "Location",
    "Province",
    "Prediction Date",
    "Rain Probability (%)",
    "Predicted Rainfall Amount (mm)",
    "Prediction",
    "Alert Status",
]

display_forecast = display_forecast.sort_values(
    "Rain Probability (%)",
    ascending=False
)

st.dataframe(
    display_forecast,
    width="stretch",
    hide_index=True
)


# =========================================================
# =========================================================
# HISTORICAL PREDICTION SIMULATION
# =========================================================

st.divider()
st.subheader("Historical Prediction Simulation")

st.markdown(
    """
    Test the two-stage rainfall prediction system on a past date.

    Temporary XGBoost classification and regression models are trained
    using only information available before the selected date.
    Predictions are then compared with the actual rainfall recorded
    on that date.
    """
)

simulation_min_date = date(1982, 1, 1)
simulation_max_date = latest_observation.date()

if simulation_max_date < simulation_min_date:
    st.warning("There is not enough historical data for this simulation.")
else:
    selected_simulation_date = st.date_input(
        "Select Historical Simulation Date",
        value=simulation_max_date,
        min_value=simulation_min_date,
        max_value=simulation_max_date,
        key="historical_simulation_date_picker"
    )

    run_simulation = st.button(
        "▶ Run Historical Simulation",
        type="primary",
        key="run_historical_simulation"
    )

    if run_simulation:
        st.session_state["historical_simulation_result_date"] = (
            selected_simulation_date.isoformat()
        )

    stored_simulation_date = st.session_state.get(
        "historical_simulation_result_date"
    )

    if stored_simulation_date == selected_simulation_date.isoformat():
        try:
            with st.spinner(
                "Training temporary model and evaluating predictions..."
            ):
                simulation_results = cached_historical_simulation(
                    selected_simulation_date.isoformat()
                )

            if simulation_results.empty:
                st.warning("No simulation results were returned.")
            else:
                actual_values = simulation_results["actual_rain"]
                predicted_values = simulation_results["predicted_rain"]

                true_positive = int(
                    ((actual_values == 1) & (predicted_values == 1)).sum()
                )
                false_positive = int(
                    ((actual_values == 0) & (predicted_values == 1)).sum()
                )
                false_negative = int(
                    ((actual_values == 1) & (predicted_values == 0)).sum()
                )

                accuracy = float(
                    simulation_results["occurrence_correct"].mean()
                )

                precision = (
                    true_positive / (true_positive + false_positive)
                    if (true_positive + false_positive) > 0
                    else 0.0
                )

                recall = (
                    true_positive / (true_positive + false_negative)
                    if (true_positive + false_negative) > 0
                    else 0.0
                )

                f1_score = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )

                amount_mae = float(
                    simulation_results["amount_absolute_error_mm"].mean()
                )

                st.success(
                    f"Historical simulation completed for "
                    f"{selected_simulation_date.strftime('%d %B %Y')}."
                )

                st.markdown("#### Occurrence Prediction Performance")

                metric_columns = st.columns(5)

                metric_columns[0].metric(
                    "Accuracy", f"{accuracy * 100:.1f}%"
                )
                metric_columns[1].metric(
                    "Precision", f"{precision * 100:.1f}%"
                )
                metric_columns[2].metric(
                    "Recall", f"{recall * 100:.1f}%"
                )
                metric_columns[3].metric(
                    "F1 Score", f"{f1_score * 100:.1f}%"
                )
                metric_columns[4].metric(
                    "Amount MAE", f"{amount_mae:.2f} mm"
                )

                st.markdown("#### Estimated vs Actual Rainfall")

                comparison_chart = px.bar(
                    simulation_results,
                    x="location",
                    y=[
                        "estimated_rainfall_mm",
                        "actual_rainfall_mm"
                    ],
                    barmode="group",
                    title="Historical Rainfall Comparison by Location",
                    labels={
                        "location": "Location",
                        "value": "Rainfall (mm)",
                        "variable": "Rainfall Type"
                    }
                )

                comparison_chart.update_layout(
                    xaxis_tickangle=-45,
                    legend_title_text="",
                    height=500
                )

                st.plotly_chart(
                    comparison_chart,
                    use_container_width=True
                )

                st.markdown("#### Location-wise Simulation Results")

                display_columns = [
                    "location",
                    "province",
                    "rain_probability_percent",
                    "prediction",
                    "estimated_rainfall_mm",
                    "actual_rainfall_mm",
                    "occurrence_correct",
                    "amount_absolute_error_mm"
                ]

                st.dataframe(
                    simulation_results[display_columns],
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    "These metrics represent one selected historical date "
                    "across the available locations. They should not be "
                    "interpreted as a complete independent evaluation."
                )

                st.info(
                    "The historical simulation uses two temporary trained "
                    "models: a classifier for rain occurrence and a regressor "
                    "for rainfall amount. Both models use only data available "
                    "before the selected simulation date. Predicted amounts "
                    "are set to zero when no rain is predicted."
                )

        except Exception as error:
            st.error(f"Historical simulation could not be completed: {error}")
            st.info(
                "Try another historical date. Your normal next-day "
                "prediction uses a separate saved production model."
            )

# MODEL PERFORMANCE
# =========================================================

st.divider()
st.subheader("Model Performance")

st.markdown(
    """
    Performance is reported using a **chronological holdout test**.
    Models were trained using target dates through 2022 and evaluated
    on later observations from 2023 onward.
    """
)

st.markdown("#### 1. Rain Occurrence — XGBoost Classifier")
st.caption(
    "Predicts whether daily rainfall will be at least 1 mm. "
    "The alert threshold used by the dashboard is 30% probability."
)

metrics = evaluation.iloc[0]
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

metric_col1.metric("Precision", f"{metrics['Precision']:.4f}")
metric_col2.metric("Recall", f"{metrics['Recall']:.4f}")
metric_col3.metric("F1 Score", f"{metrics['F1']:.4f}")
metric_col4.metric("PR-AUC", f"{metrics['PR_AUC']:.4f}")

occurrence_performance_columns = [
    "Accuracy",
    "Balanced_Accuracy",
    "ROC_AUC",
    "PR_AUC",
    "Brier_Score",
    "Log_Loss",
    "Precision",
    "Recall",
    "F1",
]

occurrence_performance_table = evaluation[
    occurrence_performance_columns
].T.reset_index()
occurrence_performance_table.columns = [
    "Evaluation Metric",
    "Value"
]
occurrence_performance_table["Value"] = (
    occurrence_performance_table["Value"].round(4)
)

st.dataframe(
    occurrence_performance_table,
    width="stretch",
    hide_index=True
)

st.divider()
st.markdown("#### 2. Rainfall Amount — XGBoost Regressor")
st.caption(
    "The regressor is trained on rainy-day observations (rainfall ≥ 1 mm). "
    "These metrics evaluate amount estimates only on test days when rain actually occurred."
)

amount_metric_col1, amount_metric_col2, amount_metric_col3 = st.columns(3)
amount_metric_col1.metric("Rainy-Day MAE", "5.73 mm")
amount_metric_col2.metric("Rainy-Day RMSE", "12.41 mm")
amount_metric_col3.metric("Rainy-Day R²", "0.0017")

st.markdown(
    "**Rainy-day historical-average baseline:** MAE 6.34 mm | "
    "RMSE 12.38 mm | R² 0.0069"
)
st.caption(
    "The regressor improves rainy-day MAE over the baseline, but its RMSE "
    "and R² are slightly weaker. Rainfall amount prediction remains challenging."
)

st.divider()
st.markdown("#### 3. Complete Two-Stage System — Overall Rainfall Amount")
st.caption(
    "The classifier first predicts rain/no rain. If rain is predicted, "
    "the regressor estimates the amount; otherwise the final amount is 0 mm. "
    "These metrics include all test days, including dry days."
)

combined_metric_col1, combined_metric_col2, combined_metric_col3 = st.columns(3)
combined_metric_col1.metric("Overall MAE", "1.57 mm")
combined_metric_col2.metric("Overall RMSE", "5.48 mm")
combined_metric_col3.metric("Overall R²", "0.1229")

st.markdown(
    "**Two-stage historical-average baseline:** MAE 1.74 mm | "
    "RMSE 5.60 mm | R² 0.0833"
)
st.success(
    "On the held-out test set, the two-stage ML system has lower overall "
    "MAE and RMSE and higher R² than the historical-average baseline."
)

st.caption(
    "All values above are from the recorded test evaluation, not from the "
    "currently displayed forecast date. Metrics describe historical held-out "
    "performance and do not guarantee future accuracy."
)

st.caption(
    "The classifier predicts rainfall occurrence (≥ 1 mm), not flood occurrence "
    "or flood severity. The amount model estimates rainfall in millimetres."
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Pakistan Rainfall Prediction System | "
    "XGBoost Machine Learning | "
    "Historical Dataset: 1981–2026 | "
    "Prediction Horizon: Next Day | "
    "Offline Dataset-Based System"
)
