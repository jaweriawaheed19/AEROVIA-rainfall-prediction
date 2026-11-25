import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# -------------------- CONFIGURATION --------------------
st.set_page_config(
    page_title="Pakistan Rainfall Prediction",
    page_icon="🌧️",
    layout="wide"
)

DATA_FILE = Path("data/rainfall.csv")
FORECAST_FILE = Path("reports/latest_forecast.csv")
EVALUATION_FILE = Path("reports/model_evaluation.csv")

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    historical = pd.read_csv(DATA_FILE)
    forecast = pd.read_csv(FORECAST_FILE)
    evaluation = pd.read_csv(EVALUATION_FILE)

    historical["date"] = pd.to_datetime(historical["date"])
    forecast["forecast_date"] = pd.to_datetime(forecast["forecast_date"])

    return historical, forecast, evaluation

historical, forecast, evaluation = load_data()

# -------------------- HEADER --------------------
st.title("🌧️ Pakistan Rainfall Prediction System")
st.markdown(
    "### Machine Learning Based 3-Day Rainfall Forecast"
)

st.info(
    "Forecasts are generated using the trained machine learning models "
    "and the latest date available in the historical dataset. "
    "This is an offline prediction system and does not use live weather APIs."
)

# -------------------- SIDEBAR --------------------
st.sidebar.title("Dashboard Controls")

province_options = ["All Provinces"] + sorted(forecast["province"].unique())

selected_province = st.sidebar.selectbox(
    "Select Province",
    province_options,
    key="province_select"
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
    "Select District",
    available_locations,
    key="district_select"
)

# -------------------- FILTER DATA --------------------
location_forecast = forecast[
    forecast["location"] == selected_location
].sort_values("forecast_date")

location_history = historical[
    historical["location"] == selected_location
].sort_values("date")

latest_date = historical["date"].max()

# -------------------- METRICS --------------------
total_locations = forecast["location"].nunique()

total_predicted_rainfall = forecast["predicted_rainfall_mm"].sum()

max_predicted_rainfall = forecast["predicted_rainfall_mm"].max()

avg_mae = evaluation["MAE_mm"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("📍 Locations Covered", total_locations)
col2.metric("🌧️ Total Forecast Rain", f"{total_predicted_rainfall:.2f} mm")
col3.metric("⬆️ Maximum Predicted Rain", f"{max_predicted_rainfall:.2f} mm")
col4.metric("🤖 Average Model MAE", f"{avg_mae:.2f} mm")

st.divider()

# -------------------- LOCATION FORECAST --------------------
st.subheader(f"📍 Rainfall Forecast: {selected_location}")

province = location_forecast["province"].iloc[0]

st.caption(
    f"Province: {province} | Forecast based on historical data through "
    f"{latest_date.strftime('%d %B %Y')}"
)

forecast_columns = st.columns(3)

for i, row in enumerate(location_forecast.itertuples()):
    forecast_columns[i].metric(
        f"Day {row.forecast_day} — {row.forecast_date.strftime('%d %b %Y')}",
        f"{max(0, row.predicted_rainfall_mm):.2f} mm"
    )

# -------------------- FORECAST CHART --------------------
st.subheader("📊 Three-Day Rainfall Forecast")

fig_forecast = px.bar(
    location_forecast,
    x="forecast_date",
    y="predicted_rainfall_mm",
    text="predicted_rainfall_mm",
    labels={
        "forecast_date": "Forecast Date",
        "predicted_rainfall_mm": "Predicted Rainfall (mm)"
    },
    title=f"Predicted Rainfall — {selected_location}"
)

fig_forecast.update_traces(
    texttemplate="%{text:.2f} mm",
    textposition="outside"
)

fig_forecast.update_layout(
    showlegend=False,
    xaxis_title="Date",
    yaxis_title="Rainfall (mm)"
)

st.plotly_chart(fig_forecast, width="stretch")

# -------------------- HISTORICAL CHART --------------------
st.subheader("📈 Historical Rainfall")

history_period = st.selectbox(
    "Historical chart period",
    ["Last 30 days", "Last 90 days", "Last 365 days", "All data"]
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
        "rainfall_mm": "Rainfall (mm)"
    },
    title=f"Historical Rainfall — {selected_location}"
)

fig_history.update_layout(
    xaxis_title="Date",
    yaxis_title="Rainfall (mm)"
)

st.plotly_chart(fig_history, width="stretch")

# -------------------- FORECAST TABLE --------------------
st.subheader("📋 Forecast Data")

display_forecast = location_forecast[
    ["forecast_date", "forecast_day", "predicted_rainfall_mm"]
].copy()

display_forecast["forecast_date"] = (
    display_forecast["forecast_date"].dt.strftime("%Y-%m-%d")
)

display_forecast["predicted_rainfall_mm"] = (
    display_forecast["predicted_rainfall_mm"].clip(lower=0).round(2)
)

display_forecast.columns = [
    "Date",
    "Forecast Day",
    "Predicted Rainfall (mm)"
]

st.dataframe(
    display_forecast,
    width="stretch",
    hide_index=True
)

# -------------------- PAKISTAN MAP --------------------
st.divider()
st.subheader("🗺️ Pakistan Rainfall Forecast Map")

map_data = forecast.copy()

map_data["predicted_rainfall_mm"] = (
    map_data["predicted_rainfall_mm"].clip(lower=0)
)

map_data["forecast_day"] = map_data["forecast_day"].astype(str)

map_day = st.selectbox(
    "Select forecast day for map",
    ["1", "2", "3"],
    format_func=lambda x: f"Day {x}"
)

map_data = map_data[
    map_data["forecast_day"] == map_day
]

if selected_province != "All Provinces":
    map_data = map_data[
        map_data["province"] == selected_province
    ]

fig_map = px.scatter_geo(
    map_data,
    lat="latitude",
    lon="longitude",
    color="predicted_rainfall_mm",
    size="predicted_rainfall_mm",
    hover_name="location",
    hover_data={
        "province": True,
        "predicted_rainfall_mm": ":.2f",
        "latitude": False,
        "longitude": False
    },
    color_continuous_scale="Blues",
    size_max=25,
    scope="asia",
    title=f"Rainfall Forecast — Day {map_day}"
)

fig_map.update_geos(
    center={"lat": 30.5, "lon": 69.3},
    projection_scale=4,
    showcountries=True,
    countrycolor="black",
    showland=True,
    landcolor="lightgray",
    showocean=True,
    oceancolor="lightblue"
)

fig_map.update_layout(
    height=650,
    geo=dict(
        lataxis_range=[23, 37],
        lonaxis_range=[60, 78]
    )
)

st.plotly_chart(fig_map, width="stretch")

# -------------------- ALL LOCATIONS TABLE --------------------
st.subheader("🌧️ All Locations — Three-Day Forecast")

all_forecasts = forecast.copy()

all_forecasts["predicted_rainfall_mm"] = (
    all_forecasts["predicted_rainfall_mm"].clip(lower=0).round(2)
)

all_forecasts["forecast_date"] = (
    all_forecasts["forecast_date"].dt.strftime("%Y-%m-%d")
)

st.dataframe(
    all_forecasts[
        [
            "location",
            "province",
            "forecast_date",
            "forecast_day",
            "predicted_rainfall_mm"
        ]
    ].rename(columns={
        "location": "Location",
        "province": "Province",
        "forecast_date": "Forecast Date",
        "forecast_day": "Forecast Day",
        "predicted_rainfall_mm": "Predicted Rainfall (mm)"
    }),
    width="stretch",
    hide_index=True
)

# -------------------- MODEL PERFORMANCE --------------------
st.divider()
st.subheader("🤖 Model Performance")

st.caption(
    "Evaluation results on the time-based testing dataset. "
    "These metrics represent the current model's performance, "
    "not guaranteed future accuracy."
)

st.dataframe(
    evaluation.round(4),
    width="stretch",
    hide_index=True
)

fig_metrics = px.bar(
    evaluation,
    x="forecast_day",
    y="MAE_mm",
    labels={
        "forecast_day": "Forecast Day",
        "MAE_mm": "Mean Absolute Error (mm)"
    },
    title="Model MAE by Forecast Day"
)

st.plotly_chart(fig_metrics, width="stretch")

# -------------------- FOOTER --------------------
st.divider()

st.caption(
    "Pakistan Rainfall Prediction System | "
    "Machine Learning Semester Project | "
    "Historical Dataset: 1981–2026 | "
    "Forecast Horizon: 3 Days"
)

