import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
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
st.subheader("Pakistan Rainfall Forecast Map")

map_day = st.selectbox(
    "Select forecast day for map",
    ["1", "2", "3"],
    format_func=lambda x: f"Day {x}"
)

map_data = forecast.copy()

map_data["predicted_rainfall_mm"] = (
    map_data["predicted_rainfall_mm"].clip(lower=0)
)

map_data["forecast_day"] = map_data["forecast_day"].astype(str)

map_data = map_data[
    map_data["forecast_day"] == map_day
].copy()

map_data["latitude"] = pd.to_numeric(
    map_data["latitude"], errors="coerce"
)

map_data["longitude"] = pd.to_numeric(
    map_data["longitude"], errors="coerce"
)

map_data = map_data.dropna(
    subset=["latitude", "longitude", "predicted_rainfall_mm"]
)

# Create a real interactive Leaflet map
pakistan_map = folium.Map(
    location=[30.3753, 69.3451],
    zoom_start=5,
    tiles="OpenStreetMap",
    control_scale=True
)

marker_cluster = MarkerCluster(
    name="Rainfall Forecast Locations",
    show=True
).add_to(pakistan_map)

def rainfall_color(rainfall):
    if rainfall < 1:
        return "green"
    elif rainfall < 2:
        return "yellow"
    elif rainfall < 5:
        return "orange"
    else:
        return "red"

# Add rainfall markers for all dataset locations
for row in map_data.itertuples():
    rainfall = max(0, float(row.predicted_rainfall_mm))
    color = rainfall_color(rainfall)

    popup_html = f"""
    <div style="font-family:Arial; width:190px;">
        <h4 style="margin-bottom:8px;">{row.location}</h4>
        <b>Province:</b> {row.province}<br>
        <b>Forecast Day:</b> {row.forecast_day}<br>
        <b>Forecast Date:</b> {row.forecast_date.strftime('%d %B %Y')}<br>
        <b>Predicted Rainfall:</b> {rainfall:.2f} mm
    </div>
    """

    folium.CircleMarker(
        location=[row.latitude, row.longitude],
        radius=7 + min(rainfall, 10) * 1.2,
        color=color,
        weight=2,
        fill=True,
        fill_color=color,
        fill_opacity=0.75,
        tooltip=f"{row.location}: {rainfall:.2f} mm",
        popup=folium.Popup(popup_html, max_width=250)
    ).add_to(marker_cluster)

# Fit the map around all plotted locations
if not map_data.empty:
    bounds = [
        [map_data["latitude"].min(), map_data["longitude"].min()],
        [map_data["latitude"].max(), map_data["longitude"].max()]
    ]
    pakistan_map.fit_bounds(bounds, padding=[35, 35])

# Add rainfall legend
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
    <b>Predicted Rainfall</b><br>
    <span style="display:inline-block;width:10px;height:10px;background:green;border-radius:50%;margin-right:7px;"></span>Low: Below 1 mm<br>
    <span style="display:inline-block;width:10px;height:10px;background:#d6b900;border-radius:50%;margin-right:7px;"></span>Moderate: 1?2 mm<br>
    <span style="display:inline-block;width:10px;height:10px;background:orange;border-radius:50%;margin-right:7px;"></span>High: 2?5 mm<br>
    <span style="display:inline-block;width:10px;height:10px;background:red;border-radius:50%;margin-right:7px;"></span>Very High: 5 mm or more
</div>
"""

pakistan_map.get_root().html.add_child(
    folium.Element(legend_html)
)

st.caption(
    f"Showing {len(map_data)} forecast locations for Day {map_day}. "
    "Click a marker to view its rainfall prediction."
)

st_folium(
    pakistan_map,
    width=None,
    height=620,
    use_container_width=True,
    returned_objects=[]
)

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

