from pathlib import Path
from datetime import date

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium

from streamlit_folium import st_folium
from folium.plugins import MarkerCluster


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

PROBABILITY_THRESHOLD = 0.30


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

    # Historical monthly-average rainfall estimate.
    # Only observations available by the last observed date are used.
    historical["rainfall_mm"] = pd.to_numeric(
        historical["rainfall_mm"], errors="coerce"
    )

    cutoff_date = forecast["last_observed_date"].max()

    available_history = historical[
        historical["date"] <= cutoff_date
    ].copy()

    available_history["month"] = available_history["date"].dt.month

    monthly_averages = available_history.groupby(
        ["location", "month"]
    )["rainfall_mm"].mean()

    location_averages = available_history.groupby(
        "location"
    )["rainfall_mm"].mean()

    overall_average = available_history["rainfall_mm"].mean()

    def estimate_monthly_rainfall(row):
        month = row["forecast_date"].month
        key = (row["location"], month)

        return monthly_averages.get(
            key,
            location_averages.get(row["location"], overall_average)
        )

    forecast["estimated_rainfall_mm"] = forecast.apply(
        estimate_monthly_rainfall,
        axis=1
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

st.title("🌧️ Pakistan Rainfall Prediction System")

st.markdown(
    "### Machine Learning Based Next-Day Rainfall Prediction"
)

st.markdown(
    """
    This system uses an **XGBoost classification model** to estimate
    the probability of rainfall at 22 selected locations across Pakistan.
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

st.subheader(f"📍 Next-Day Prediction: {selected_location}")

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
    "Historical Rainfall Estimate",
    f"{selected_row['estimated_rainfall_mm']:.2f} mm"
)

st.caption(
    "The rainfall amount is the historical average daily rainfall "
    "for this location and forecast month. It is not a weather-driven "
    "rainfall amount prediction."
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

st.subheader("📊 Rainfall Probability")

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
st.subheader("📈 Historical Rainfall")

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
        <b>Historical Rainfall Estimate:</b> {row.estimated_rainfall_mm:.2f} mm<br>
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
st.subheader("🌧️ All Locations — Next-Day Predictions")

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
    "Historical Rainfall Estimate (mm)",
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
# MODEL PERFORMANCE
# =========================================================

st.divider()
st.subheader("🤖 Model Performance")

st.markdown(
    """
    The model was evaluated using a chronological test dataset:
    training target dates through 2022 and testing observations from 2023 onward.
    """
)

metrics = evaluation.iloc[0]

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

metric_col1.metric(
    "Precision",
    f"{metrics['Precision']:.4f}"
)

metric_col2.metric(
    "Recall",
    f"{metrics['Recall']:.4f}"
)

metric_col3.metric(
    "F1 Score",
    f"{metrics['F1']:.4f}"
)

metric_col4.metric(
    "PR-AUC",
    f"{metrics['PR_AUC']:.4f}"
)

performance_columns = [
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

performance_table = evaluation[
    performance_columns
].T.reset_index()

performance_table.columns = [
    "Evaluation Metric",
    "Value"
]

performance_table["Value"] = performance_table["Value"].round(4)

st.dataframe(
    performance_table,
    width="stretch",
    hide_index=True
)

st.caption(
    "The evaluation metrics describe historical held-out test performance. "
    "They do not guarantee future prediction accuracy."
)

st.caption(
    "The classifier predicts rainfall occurrence (at least 1 mm), "
    "not rainfall amount, flood occurrence, or flood severity."
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
