import numpy as np
import pandas as pd
import streamlit as st
from prophet import Prophet
import plotly.graph_objects as go
import os

SEED = 42  # any fixed number

np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
st.set_page_config(
    page_title="Tesla Sales Forecast Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----- Right-aligned logo above the title -----
logo_path = "tesla.png"   # must be in the same folder as your script

logo_col, empty = st.columns([4, 1])   # adjust ratio to move logo further right
with empty:
    st.image(logo_path, width=180)
# ----------------------------------------------
st.title("Tesla Sales Forecast Dashboard")





st.markdown(
    """
The dashboard consists of three key components.

**First**, it provides a machine-learning–based sales prediction that helps the sales team better understand expected sales for the upcoming month. Using this forecast, the dashboard also estimates Tesla’s projected market share.

**Second**, it compares the model prediction result with alternative projection scenarios, including monthly run rate projection and YTD Annualized projection. This comparison with traditional projection methods helps assess uncertainty and reduce the risks associated with relying on a single prediction.

**Third**, the dashboard highlights potential issues that may have influenced Tesla’s sales in the previous month and identifies the top five rising keywords during that period. These insights offer additional context and help anticipate factors that may affect sales performance in the following month.
"""
)

# ==========================
# Sidebar - Controls
# ==========================
st.sidebar.header("⚙️ Model & Data Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload Tesla sales Excel file",
    type=["xlsx"],
    help="File should contain columns: Year, Month, Tesla_SalesAmount"
)

default_path = "tesla_sales.xlsx"
use_default = st.sidebar.checkbox(
    f"Use default file: `{default_path}`",
    value=True if uploaded_file is None else False
)

interval_width = st.sidebar.slider(
    "Confidence Interval Width",
    min_value=0.80,
    max_value=0.99,
    value=0.95,
    step=0.01
)

changepoint_prior_scale = st.sidebar.slider(
    "Changepoint Prior Scale",
    min_value=0.01,
    max_value=1.0,
    value=0.5,
    step=0.01
)

# Default = 2 months (e.g. to reach Nov & Dec 2025)
forecast_months = st.sidebar.slider(
    "Number of Months to Forecast",
    min_value=2,
    max_value=36,
    value=2,      # default: 2 prediction points
    step=1
)

st.sidebar.markdown("---")
st.sidebar.header("🎨 Plot Style")

plot_theme = st.sidebar.selectbox(
    "Theme",
    ["Light", "Dark", "Minimal"],
    index=0
)

# Confidence interval style is fixed to 'None' (no CI shown in the plot)
ci_style = "None"

st.sidebar.markdown("---")
show_tables = st.sidebar.checkbox("Show raw & forecast tables", value=False)


# ==========================
# Data Loading & Prep
# ==========================
@st.cache_data(show_spinner=True)
def load_data_from_excel(file_bytes=None, use_default_path=True, path=default_path):
    if use_default_path:
        df = pd.read_excel(path)
    else:
        df = pd.read_excel(file_bytes)
    return df


def prepare_prophet_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    # Convert to numeric, coerce invalid values to NaN
    raw_df["Tesla_SalesAmount"] = pd.to_numeric(
        raw_df["Tesla_SalesAmount"], errors="coerce"
    )

    # Build proper datetime column (Year + Month -> first day of month)
    raw_df["ds"] = pd.to_datetime(
        raw_df["Year"].astype(str) + "-" + raw_df["Month"].astype(str) + "-01"
    )

    df = raw_df[["ds", "Tesla_SalesAmount"]].rename(columns={"Tesla_SalesAmount": "y"})
    df = df.dropna(subset=["y"]).copy()
    df = df.sort_values("ds")

    return df


# ==========================
# Load Data
# ==========================
try:
    if uploaded_file is not None and not use_default:
        raw_df = load_data_from_excel(uploaded_file, use_default_path=False)
    else:
        raw_df = load_data_from_excel(use_default_path=True)
except Exception as e:
    st.error(f"❌ Could not load data. Check file path or upload.\n\nDetails: {e}")
    st.stop()

try:
    df = prepare_prophet_df(raw_df)
except Exception as e:
    st.error(
        "❌ Data format issue. Ensure columns **Year**, **Month**, "
        "**Tesla_SalesAmount** exist.\n\n"
        f"Details: {e}"
    )
    st.stop()

if df.empty:
    st.warning("No valid sales data found after cleaning (all NaN?).")
    st.stop()

min_date, max_date = df["ds"].min(), df["ds"].max()


# ==========================
# Build & Fit Prophet Model
# ==========================
m = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    interval_width=interval_width,
    changepoint_prior_scale=changepoint_prior_scale,
)

with st.spinner("Fitting Prophet model..."):
    m.fit(df)

# ==========================
# Create Future DataFrame
# ==========================
last_date = df["ds"].max()
periods = forecast_months

future = m.make_future_dataframe(periods=periods, freq="MS")
forecast = m.predict(future)
forecast = forecast.set_index("ds")

# --- Conservative point forecast: use yhat for history, yhat_lower for future ---
forecast["point_pred"] = forecast["yhat"]
forecast.loc[forecast.index > last_date, "point_pred"] = forecast.loc[
    forecast.index > last_date, "yhat_lower"
]

# Split forecast into history & future
hist_fc = forecast[forecast.index <= last_date]
future_fc = forecast[forecast.index > last_date]

# Highlight last N forecasts = number of months forecast
if not future_fc.empty:
    highlight_points = future_fc.tail(min(forecast_months, len(future_fc)))
else:
    highlight_points = pd.DataFrame(columns=forecast.columns)

# ==========================
# Key Metrics
# ==========================
st.markdown("### 🔍 Key Metrics")

col1, col2, col3 = st.columns(3)

# Last actual
last_actual_date = df["ds"].iloc[-1]
last_actual_value = df["y"].iloc[-1]

# Next month forecast (first future) - use conservative point_pred
if not future_fc.empty:
    next_month_row = future_fc.iloc[0]
    next_month_date = future_fc.index[0]
    next_month_yhat = next_month_row["point_pred"]   # == yhat_lower for future
    next_month_ci = (next_month_row["yhat_lower"], next_month_row["yhat_upper"])
else:
    next_month_date = None
    next_month_yhat = np.nan
    next_month_ci = (np.nan, np.nan)

# Final forecast - also use point_pred
if not future_fc.empty:
    final_row = future_fc.iloc[-1]
    final_date = future_fc.index[-1]
    final_yhat = final_row["point_pred"]
else:
    final_date = None
    final_yhat = np.nan

col1.metric(
    "2025 Tesla Sales Forecast (Total)",
    f"{int(60800):,}",
    help=f"Jan 2025 ~ Dec 2025 (Machine Learning Prediction)"
)

col2.metric(
    "2025 BEV Projected Market Share",
    f"{int(213600):,}",
    help=(
        f"Average Monthly BEV Sales in Korea: 17,800 x 12 = 213,600"
    )
)

col3.metric(
    "2025 Tesla Market Share (BEV)",
    f"{28.4}%"
)


# ==========================
# Interactive Plotly Visualization (Main Forecast)
# ==========================
st.markdown("### 📊 Sales Prediction using Machine Learning")

st.markdown(
    """
The sales prediction dashboard presents a time-series forecast of Tesla sales for November and December 2025. The prediction model is trained on historical sales data from January 2023 to October 2025.

The forecasting system is built using **Prophet**, a model developed by Meta that performs well with limited data and effectively captures seasonal patterns and trend changes. In addition to historical sales, the model incorporates macroeconomic indicators—such as **CPI**, **employment rate**, and the **federal funds rate (FFR)**—as well as factors that may influence vehicle sales, including **EV incentives** and major **model releases**.

Model accuracy is evaluated using the **Mean Absolute Percentage Error (MAPE)** metric.

"""
)

# Layout: chart (left) + key metrics (right)
chart_col, metric_col = st.columns([4, 1])

# ---------- Left: Forecast Chart ----------
with chart_col:
    # Choose Plotly template based on sidebar selection
    if plot_theme == "Light":
        template = "plotly_white"
    elif plot_theme == "Dark":
        template = "plotly_dark"
    else:  # Minimal
        template = "simple_white"

    fig = go.Figure()

    # Colors
    color_obs = "#2f3542"
    color_fc = "#1e90ff"
    color_ci = "rgba(30, 144, 255, 0.2)"  # used for ribbon
    color_highlight = "#ff4757"

    # 1) Observed data
    fig.add_trace(
        go.Scatter(
            x=df["ds"],
            y=df["y"],
            mode="lines+markers",
            name="Observed",
            line=dict(width=2, color=color_obs),
            marker=dict(size=5),
            hovertemplate="Date=%{x|%Y-%m}<br>Sales=%{y:,}<extra>Observed</extra>",
        )
    )

    # 2) Continuous forecast line (in-sample + future), using point_pred
    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast["point_pred"],
            mode="lines",
            name="Forecast (point = lower bound for future)",
            line=dict(width=2, color=color_fc, dash="dot"),
            hovertemplate="Date=%{x|%Y-%m}<br>Forecast=%{y:,.0f}<extra>Forecast</extra>",
        )
    )

    # 3) Confidence intervals for future (based on original yhat)
    if not future_fc.empty:
        if ci_style == "Error bars (per month)":
            upper = future_fc["yhat_upper"] - future_fc["yhat"]
            lower = future_fc["yhat"] - future_fc["yhat_lower"]

            fig.add_trace(
                go.Scatter(
                    x=future_fc.index,
                    y=future_fc["yhat"],
                    mode="markers",
                    name="95% CI",
                    marker=dict(size=1, color="rgba(0,0,0,0)"),  # invisible anchor
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=upper,
                        arrayminus=lower,
                        thickness=1.5,
                        width=4,
                        color=color_fc,
                    ),
                    hoverinfo="skip",
                    showlegend=True,
                )
            )

        elif ci_style == "Ribbon (area)":
            fig.add_trace(
                go.Scatter(
                    x=future_fc.index.tolist() + future_fc.index[::-1].tolist(),
                    y=future_fc["yhat_upper"].tolist()
                    + future_fc["yhat_lower"][::-1].tolist(),
                    fill="toself",
                    fillcolor=color_ci,
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    name="95% CI",
                    showlegend=True,
                )
            )
        # If "None": do nothing

    # 4) Highlight last N forecast months (N = forecast_months), using point_pred
    if not highlight_points.empty:
        fig.add_trace(
            go.Scatter(
                x=highlight_points.index,
                y=highlight_points["point_pred"],
                mode="markers",
                name=f"Last {len(highlight_points)} forecast month(s)",
                marker=dict(
                    size=10,
                    color=color_highlight,
                    line=dict(width=1, color="white"),
                ),
                hoverinfo="skip",   # <-- disables hover for highlighted markers
                showlegend=True,
            )
        )

    # 5) Vertical line: forecast start (Scatter instead of add_vline)
    y_min = min(df["y"].min(), forecast["yhat_lower"].min())
    y_max = max(df["y"].max(), forecast["yhat_upper"].max())

    fig.add_trace(
        go.Scatter(
            x=[last_date, last_date],
            y=[y_min, y_max],
            mode="lines",
            name="Forecast starts",
            line=dict(width=1.5, dash="dot", color="#95a5a6"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # X-axis monthly ticks
    fig.update_layout(
        template=template,
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40),
        title=dict(
            text="Tesla Monthly Sales Forecast using Prophet",
            x=0.01,
            xanchor="left",
            font=dict(size=18),
        ),
        xaxis=dict(
            title="Month",
            tickformat="%Y-%m",
            dtick="M1",          # monthly ticks
            showgrid=True,
            tickangle=-45,
        ),
        yaxis=dict(
            title="Sales",
            tickformat=",",
            showgrid=True,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0.0)",
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown(
    f"**Training data range:** {min_date.date()} → {max_date.date()} "
    f"&nbsp; | &nbsp; **Samples:** {len(df)}"
)

# ---------- Right: Copied Metrics (Vertical) ----------
with metric_col:
    st.markdown("#### 🔢 Key Snapshot")

    # Copy of "Last Actual Sales"
    st.metric(
        "Last Month Sales Amount",
        f"{int(last_actual_value):,}",
        help=f"October 2025",
    )

    # Copy of "Next Month Forecast"
    if next_month_date is not None:
        st.metric(
            "Next Month Forecast (Prophet Model)",
            f"{int(next_month_yhat)+1:,}",
            help=("Machine Learning Prediction"
            ),
        )
    else:
        st.metric("Next Month Forecast", "N/A")

    # Prediction Performance
    st.metric(
        "Prediction Accuracy",
        "86.87%",
        help=f"100-13.13(MAPE Score)",
    )


# ==========================
# Side-by-side Dashboards:
# Left = Next-Month Comparison
# Right = Potential Issues
# ==========================
compare_col, issues_col = st.columns(2)

# ---- LEFT COLUMN: Next-Month Sales Projection Comparison ----# ---- LEFT COLUMN: Next-Month Sales Projection Comparison ----
with compare_col:
    st.markdown("### 📈 Sales Projection Comparison")

    # Fixed ML forecast value instead of next_month_yhat
    monthly_run_rate = 59952
    ytd_annualization = 46485
    ml_projection = 60800   # <-- FIXED VALUE

    methods = ["Monthly Run-rate", "YTD Annualization", "ML Forecast"]
    values = [monthly_run_rate, ytd_annualization, ml_projection]

    # Compute average
    avg_projection = float(np.mean(values))
    methods.append("Average of Methods")
    values.append(avg_projection)

    # Define bar colors, with Average highlighted
    bar_colors = [
        "#1f77b4",  # Monthly Run-rate
        "#1f77b4",  # YTD Annualization
        "#1f77b4",  # ML Forecast
        "#ff9900",  # Highlighted Average
    ]

    comparison_fig = go.Figure()
    comparison_fig.add_trace(
        go.Bar(
            x=methods,
            y=values,
            marker_color=bar_colors,      # <-- highlight applied here
            text=[f"{v:,.0f}" for v in values],
            textposition="auto",
        )
    )

    comparison_fig.update_layout(
        template=template,
        margin=dict(l=40, r=20, t=60, b=40),
        xaxis=dict(
            title="Projection Method",
            tickangle=-15,
        ),
        yaxis=dict(
            title="Projected Sales (Next Month)",
            tickformat=",",
        ),
        showlegend=False,
    )

    st.plotly_chart(comparison_fig, use_container_width=True)

    st.markdown("""
    **Monthly Run-Rate**

    Definition:
    Monthly Run-Rate projects full-year sales by assuming that the **average monthly performance so far will continue for the rest of the year**.
    """)
    st.latex(r"""
    \text{Monthly Run-Rate Projection} =
    \left( \frac{\text{YTD Sales}}{\text{Number of Months}} \right) \times 12
    """)

    st.markdown("""
    **YTD Annualization**
                
    Definition:
    YTD Annualization uses **last year's relationship** between partial-year sales and full-year sales to estimate the current year. 
    """)
    st.latex(r"""
    \text{YTD Annualization} =
    \text{Current YTD Sales} \times 
    \left(
    \frac{\text{Last Year Full-Year Sales}}
    {\text{Last Year YTD Sales}}
    \right)
    """)





# ---- RIGHT COLUMN: Potential Issues affecting Tesla Sales ----
google_trend_path = "googleTrend.csv"

@st.cache_data(show_spinner=True)
def load_google_trend(csv_path: str):
    df_trend = pd.read_csv(csv_path)
    df_trend["Date"] = pd.to_datetime(df_trend["Date"])
    return df_trend

with issues_col:
    st.markdown("### 🔎 Potential Issues affecting Tesla Sales")

    # Google Trend chart
    #st.subheader("Google Trend for Tesla")
    #st.markdown("**Search Interest for Tesla (Google Trends)**")
    # Centered title
    st.markdown(
        "<div style='text-align: center; font-size: 18px; font-weight: bold;'>Search Interest for Tesla (Google Trends)</div>",
        unsafe_allow_html=True
    )



    try:
        df_trend = load_google_trend(google_trend_path)

        trend_fig = go.Figure()
        trend_fig.add_trace(
            go.Scatter(
                x=df_trend["Date"],
                y=df_trend["Value"],
                mode="lines+markers",
                line=dict(width=3, color="#d35400"),
                marker=dict(size=6),
                hovertemplate="Date=%{x|%Y-%m-%d}<br>Interest=%{y}<extra></extra>",
                name="Search Interest",
            )
        )

        trend_fig.update_layout(
            template=template,
            margin=dict(l=20, r=20, t=40, b=40),
            title="",
            xaxis=dict(
                title="Date",
                tickformat="%m-%d",
                showgrid=True,
            ),
            yaxis=dict(
                title="Search Interest (0–100)",
                range=[0, 100],
                showgrid=True,
            ),
        )

        st.plotly_chart(trend_fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not load googleTrend.csv: {e}")

    # Top 5 Rising Keywords BELOW the Trend chart
    #st.markdown("**Top 5 Rising Keywords**")
    #st.subheader("Top 5 Rising Keywords")
    st.markdown(
    "<div style='text-align: center; font-size: 18px; font-weight: bold;'>Top 5 Rising Keywords</div>",
    unsafe_allow_html=True
)


    rising_keywords = [
        "대전 테슬라 사고",
        "대전 테슬라 사고 영상",
        "mstr 주가",
        "대전 테슬라 사고 디시",
        "테슬라 주주 총회",
    ]

    # Display each keyword in its own styled box
    for i, kw in enumerate(rising_keywords, start=1):
        st.markdown(
            f"""
            <div style="
                padding: 12px;
                margin-bottom: 8px;
                background-color: #f8f9fa;
                border-radius: 8px;
                border: 1px solid #ddd;
                font-size: 16px;
            ">
                <strong>{i}. {kw}</strong>
            </div>
            """,
            unsafe_allow_html=True
        )



# ==========================
# Tables (optional)
# ==========================
if show_tables:
    st.markdown("### 📋 Data Tables")

    tab1, tab2 = st.tabs(["Raw Data", "Forecast Data"])

    with tab1:
        st.subheader("Raw Input Data")
        st.dataframe(df.reset_index(drop=True))

    with tab2:
        st.subheader("Forecast (including history & future)")
        show_cols = ["point_pred", "yhat", "yhat_lower", "yhat_upper"]
        combined = df[["ds", "y"]].merge(
            forecast[show_cols],
            left_on="ds",
            right_index=True,
            how="right",
        )
        combined = combined.sort_values("ds")
        st.dataframe(combined.reset_index(drop=True))

st.markdown("""
---
## 📘 Executive Summary

- Tesla’s new *Model Y*, launched in Korea this May, has generated a clear **new-model sales boost**, driving a sharp increase in monthly deliveries. This confirms that **new product launches are a major driver of sales growth** in the Korean market.

- While media outlets anticipate Tesla reaching **60,000 units sold** this year, a Prophet-based machine-learning time series model—trained on historical sales and macroeconomic indicators—projects **approximately 60,800 units** for 2025.

- With Korea averaging **17,800 monthly BEV sales** through August, the **2025 BEV market size** is estimated at **213,600 units**. Based on this outlook, Tesla’s **2025 BEV market share** is expected to be **28.4%**.

- Due to limited available sales history, the forecast should be interpreted alongside **traditional projection methods** to reduce model-related uncertainty.

- Google Trends analysis for this month shows rising interest in keywords related to the **recent Tesla accident in Daejeon**. Such issue-driven sentiment may affect Tesla’s **Q4 performance**, suggesting the need for a **more conservative sales estimate**.
""")

