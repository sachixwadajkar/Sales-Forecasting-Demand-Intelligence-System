import os
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# =========================================================
# APP CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="SupplyPulse | Sales Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CUSTOM STYLING
# =========================================================
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1450px;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        }

        [data-testid="stSidebar"] * {
            color: #f8fafc;
        }

        .hero {
            padding: 1.4rem 1.6rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 55%, #0369a1 100%);
            color: white;
            margin-bottom: 1.2rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.15);
        }

        .hero h1 {
            margin: 0;
            font-size: 2.15rem;
        }

        .hero p {
            margin: 0.45rem 0 0 0;
            opacity: 0.88;
            font-size: 1rem;
        }

        .insight-box {
            padding: 1rem 1.1rem;
            border-radius: 14px;
            background: rgba(59, 130, 246, 0.08);
            border: 1px solid rgba(59, 130, 246, 0.22);
            margin: 0.5rem 0 1rem 0;
        }

        div[data-testid="stMetric"] {
            background: rgba(148, 163, 184, 0.08);
            border: 1px solid rgba(148, 163, 184, 0.22);
            padding: 14px 16px;
            border-radius: 14px;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
        }

        .small-note {
            color: #64748b;
            font-size: 0.88rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# DATA LOADING
# =========================================================
@st.cache_data
def load_data():
    possible_files = ["train.csv", "train(1).csv"]

    csv_file = next((f for f in possible_files if os.path.exists(f)), None)

    if csv_file is None:
        st.error(
            "Dataset not found. Place the Superstore dataset in the project folder "
            "and name it `train.csv`."
        )
        st.stop()

    data = pd.read_csv(csv_file)

    # Dataset contains day-first dates such as 15/04/2018.
    for col in ["Order Date", "Ship Date"]:
        data[col] = pd.to_datetime(
            data[col],
            format="mixed",
            dayfirst=True,
            errors="coerce",
        )

    data = data.dropna(subset=["Order Date", "Sales"]).copy()

    data["Year"] = data["Order Date"].dt.year
    data["Month"] = data["Order Date"].dt.month
    data["Month Name"] = data["Order Date"].dt.month_name()
    data["Quarter"] = "Q" + data["Order Date"].dt.quarter.astype(str)

    return data


df = load_data()

# =========================================================
# SHARED HELPERS
# =========================================================
def money(value):
    return f"${value:,.0f}"


def hero(title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def monthly_series(data, filter_column=None, filter_value=None):
    temp = data.copy()

    if filter_column and filter_value:
        temp = temp[temp[filter_column] == filter_value]

    series = (
        temp.set_index("Order Date")["Sales"]
        .resample("ME")
        .sum()
        .asfreq("ME", fill_value=0)
    )

    return series


@st.cache_data(show_spinner=False)
def run_sarima_forecast(series_values, series_dates, horizon):
    ts = pd.Series(
        np.asarray(series_values, dtype=float),
        index=pd.DatetimeIndex(series_dates),
    ).sort_index()

    if len(ts) < 30:
        raise ValueError("Not enough monthly history for a reliable seasonal forecast.")

    # Back-test on the last 3 known months for MAE/RMSE.
    train = ts.iloc[:-3]
    test = ts.iloc[-3:]

    eval_model = SARIMAX(
        train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    eval_fit = eval_model.fit(disp=False, maxiter=200)
    test_pred = eval_fit.get_forecast(steps=3).predicted_mean

    mae = mean_absolute_error(test.values, test_pred.values)
    rmse = np.sqrt(mean_squared_error(test.values, test_pred.values))

    # Refit on all available history before forecasting the future.
    final_model = SARIMAX(
        ts,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    final_fit = final_model.fit(disp=False, maxiter=200)
    result = final_fit.get_forecast(steps=horizon)

    forecast = result.predicted_mean
    confidence = result.conf_int(alpha=0.05)

    return forecast, confidence, mae, rmse


@st.cache_data
def build_anomaly_data(data):
    weekly = (
        data.set_index("Order Date")["Sales"]
        .resample("W")
        .sum()
        .reset_index()
    )

    model = IsolationForest(contamination=0.05, random_state=42)
    weekly["Anomaly_IF"] = model.fit_predict(weekly[["Sales"]])

    anomalies = weekly[weekly["Anomaly_IF"] == -1].copy()
    return weekly, anomalies


@st.cache_data
def build_product_segments(data):
    monthly_product = (
        data.groupby(
            ["Sub-Category", pd.Grouper(key="Order Date", freq="ME")]
        )["Sales"]
        .sum()
        .reset_index()
    )

    features = (
        monthly_product.groupby("Sub-Category")["Sales"]
        .agg(Total_Sales="sum", Sales_Volatility="std")
    )

    avg_order_value = (
        data.groupby("Sub-Category")["Sales"]
        .mean()
        .rename("Average_Order_Value")
    )
    features = features.join(avg_order_value)

    yearly_sales = (
        data.groupby(["Sub-Category", data["Order Date"].dt.year])["Sales"]
        .sum()
        .unstack()
    )

    yoy_growth = yearly_sales.pct_change(axis=1) * 100
    features["YoY_Growth"] = yoy_growth.mean(axis=1)

    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)

    feature_columns = [
        "Total_Sales",
        "Sales_Volatility",
        "Average_Order_Value",
        "YoY_Growth",
    ]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features[feature_columns])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    features["Cluster"] = kmeans.fit_predict(scaled)

    # Labels derived from the cluster profiles in the completed notebook.
    cluster_labels = {
        0: "High Volume, High Volatility",
        1: "Low Volume, Stable Demand",
        2: "High Volume, Stable Demand",
        3: "Rapidly Growing Demand",
    }

    features["Demand Segment"] = features["Cluster"].map(cluster_labels)

    pca = PCA(n_components=2)
    pca_values = pca.fit_transform(scaled)
    features["PCA1"] = pca_values[:, 0]
    features["PCA2"] = pca_values[:, 1]

    return features.reset_index(), pca.explained_variance_ratio_


weekly_sales, anomalies = build_anomaly_data(df)
product_segments, pca_variance = build_product_segments(df)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 📈 SupplyPulse")
    st.caption("Sales Forecasting & Demand Intelligence")
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "🏠 Sales Overview",
            "🔮 Forecast Explorer",
            "🚨 Anomaly Report",
            "📦 Demand Segments",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("#### Dataset")
    st.write(f"**{len(df):,}** transactions")
    st.write(
        f"**{df['Order Date'].min():%b %Y} – "
        f"{df['Order Date'].max():%b %Y}**"
    )
    st.caption("Built from the completed Superstore sales analysis.")

# =========================================================
# PAGE 1 — SALES OVERVIEW
# =========================================================
if page == "🏠 Sales Overview":
    hero(
        "Sales Overview",
        "A business-level view of revenue, growth, seasonality, regions and product categories.",
    )

    # Global filters
    with st.expander("🎛️ Dashboard Filters", expanded=True):
        f1, f2, f3 = st.columns(3)

        selected_years = f1.multiselect(
            "Year",
            sorted(df["Year"].unique()),
            default=sorted(df["Year"].unique()),
        )

        selected_regions = f2.multiselect(
            "Region",
            sorted(df["Region"].unique()),
            default=sorted(df["Region"].unique()),
        )

        selected_categories = f3.multiselect(
            "Category",
            sorted(df["Category"].unique()),
            default=sorted(df["Category"].unique()),
        )

    filtered = df[
        df["Year"].isin(selected_years)
        & df["Region"].isin(selected_regions)
        & df["Category"].isin(selected_categories)
    ].copy()

    if filtered.empty:
        st.warning("No data matches the selected filters.")
        st.stop()

    total_sales = filtered["Sales"].sum()
    total_orders = filtered["Order ID"].nunique()
    avg_order_value = filtered.groupby("Order ID")["Sales"].sum().mean()
    top_category = (
        filtered.groupby("Category")["Sales"].sum().idxmax()
        if not filtered.empty else "N/A"
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Sales", money(total_sales))
    k2.metric("Orders", f"{total_orders:,}")
    k3.metric("Avg. Order Value", money(avg_order_value))
    k4.metric("Top Category", top_category)

    st.markdown("### Performance at a glance")
    c1, c2 = st.columns([1.05, 1.55])

    yearly = filtered.groupby("Year", as_index=False)["Sales"].sum()
    fig_year = px.bar(
        yearly,
        x="Year",
        y="Sales",
        text_auto=".3s",
        title="Total Sales by Year",
    )
    fig_year.update_traces(
        marker_color=yearly["Sales"],
        marker_colorscale="Blues",
        textposition="outside",
    )
    fig_year.update_layout(
        showlegend=False,
        yaxis_title="Sales ($)",
        xaxis_title="Year",
        hovermode="x unified",
    )
    c1.plotly_chart(fig_year, use_container_width=True)

    monthly = (
        filtered.set_index("Order Date")["Sales"]
        .resample("ME")
        .sum()
        .reset_index()
    )
    fig_month = px.area(
        monthly,
        x="Order Date",
        y="Sales",
        title="Monthly Sales Trend",
    )
    fig_month.update_traces(
        line=dict(width=3),
        fill="tozeroy",
        hovertemplate="<b>%{x|%b %Y}</b><br>Sales: $%{y:,.0f}<extra></extra>",
    )
    fig_month.update_layout(
        xaxis_title="Month",
        yaxis_title="Sales ($)",
        hovermode="x unified",
    )
    c2.plotly_chart(fig_month, use_container_width=True)

    c3, c4 = st.columns(2)

    region_category = (
        filtered.groupby(["Region", "Category"], as_index=False)["Sales"].sum()
    )
    fig_rc = px.bar(
        region_category,
        x="Region",
        y="Sales",
        color="Category",
        barmode="group",
        title="Sales by Region and Category",
    )
    fig_rc.update_layout(yaxis_title="Sales ($)", xaxis_title="")
    c3.plotly_chart(fig_rc, use_container_width=True)

    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    seasonality = (
        filtered.groupby(["Year", "Month Name"], as_index=False)["Sales"].sum()
    )
    seasonality["Month Name"] = pd.Categorical(
        seasonality["Month Name"], categories=month_order, ordered=True
    )
    seasonality = seasonality.sort_values(["Year", "Month Name"])

    fig_season = px.line(
        seasonality,
        x="Month Name",
        y="Sales",
        color="Year",
        markers=True,
        title="Seasonality Across Years",
    )
    fig_season.update_layout(
        xaxis_title="Month",
        yaxis_title="Sales ($)",
        legend_title="Year",
    )
    c4.plotly_chart(fig_season, use_container_width=True)

    category_sales = filtered.groupby("Category")["Sales"].sum().sort_values(ascending=False)
    best_category = category_sales.index[0]
    best_value = category_sales.iloc[0]

    st.markdown(
        f"""
        <div class="insight-box">
        <b>Business insight:</b> <b>{best_category}</b> is the strongest category in the
        current selection with <b>{money(best_value)}</b> in sales. Use the filters above
        to compare how the revenue mix changes across years and regions.
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# PAGE 2 — FORECAST EXPLORER
# =========================================================
elif page == "🔮 Forecast Explorer":
    hero(
        "Forecast Explorer",
        "Explore 1–3 month sales forecasts using the best-performing model from the analysis: SARIMA.",
    )

    s1, s2, s3 = st.columns([1, 1.4, 1])

    forecast_type = s1.selectbox(
        "Forecast by",
        ["Category", "Region"],
    )

    options = sorted(df[forecast_type].dropna().unique())
    selected_value = s2.selectbox(
        f"Select {forecast_type}",
        options,
    )

    horizon = s3.slider(
        "Select Forecast Horizon (Months Ahead)",
        min_value=1,
        max_value=3,
        value=3,
        step=1,
        format="%d month(s)",
    )

    ts = monthly_series(df, forecast_type, selected_value)

    with st.spinner("Training SARIMA and generating the forecast..."):
        try:
            forecast, confidence, mae, rmse = run_sarima_forecast(
                tuple(ts.values),
                tuple(ts.index),
                horizon,
            )
        except Exception as exc:
            st.error(f"Forecast could not be generated: {exc}")
            st.stop()

    # Interactive forecast chart
    fig = go.Figure()

    history = ts.iloc[-24:]
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history.values,
            mode="lines+markers",
            name="Historical Sales",
            line=dict(width=3),
        )
    )

    lower = confidence.iloc[:, 0]
    upper = confidence.iloc[:, 1]

    fig.add_trace(
        go.Scatter(
            x=list(forecast.index) + list(forecast.index[::-1]),
            y=list(upper.values) + list(lower.values[::-1]),
            fill="toself",
            fillcolor="rgba(99, 102, 241, 0.16)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="95% Confidence Range",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast.values,
            mode="lines+markers",
            name="SARIMA Forecast",
            line=dict(width=4, dash="dash"),
            marker=dict(size=10),
        )
    )

    fig.update_layout(
        title=f"{selected_value}: {horizon}-Month Sales Forecast",
        xaxis_title="Date",
        yaxis_title="Monthly Sales ($)",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.12),
        height=520,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Model metrics are intentionally shown below the chart to match Task 7.
    st.markdown("### Model Performance")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", money(mae))
    m2.metric("RMSE", money(rmse))
    m3.metric(
        f"{horizon}-Month Forecast Total",
        money(float(forecast.sum())),
    )

    forecast_table = pd.DataFrame(
        {
            "Month": forecast.index.strftime("%B %Y"),
            "Expected Sales": forecast.values,
            "Lower Range (95%)": lower.values,
            "Upper Range (95%)": upper.values,
        }
    )

    display_forecast = forecast_table.copy()
    for col in ["Expected Sales", "Lower Range (95%)", "Upper Range (95%)"]:
        display_forecast[col] = display_forecast[col].map(lambda x: f"${x:,.2f}")

    st.markdown("### Forecast details")
    st.dataframe(display_forecast, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="insight-box">
        <b>How to read this:</b> The forecast line is the model's expected sales level.
        The shaded band is the 95% confidence range and represents forecast uncertainty.
        For inventory planning, use the expected value as the base case and the upper range
        as a high-demand scenario.
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# PAGE 3 — ANOMALY REPORT
# =========================================================
elif page == "🚨 Anomaly Report":
    hero(
        "Anomaly Report",
        "Identify weeks with unusually high or low sales so the business can investigate promotions, large orders or operational issues.",
    )

    a1, a2, a3 = st.columns(3)
    a1.metric("Weeks Analysed", f"{len(weekly_sales):,}")
    a2.metric("Anomalies Detected", f"{len(anomalies):,}")
    a3.metric(
        "Largest Anomalous Week",
        money(anomalies["Sales"].max()),
    )

    fig_anomaly = go.Figure()

    fig_anomaly.add_trace(
        go.Scatter(
            x=weekly_sales["Order Date"],
            y=weekly_sales["Sales"],
            mode="lines",
            name="Weekly Sales",
            line=dict(width=2),
        )
    )

    fig_anomaly.add_trace(
        go.Scatter(
            x=anomalies["Order Date"],
            y=anomalies["Sales"],
            mode="markers",
            name="Detected Anomaly",
            marker=dict(size=11, symbol="diamond"),
            customdata=anomalies["Sales"],
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Sales: $%{y:,.2f}<extra></extra>",
        )
    )

    fig_anomaly.update_layout(
        title="Weekly Sales with Isolation Forest Anomalies",
        xaxis_title="Week",
        yaxis_title="Weekly Sales ($)",
        hovermode="x unified",
        height=520,
        legend=dict(orientation="h", y=1.12),
    )

    st.plotly_chart(fig_anomaly, use_container_width=True)

    st.markdown("### Detected anomaly dates")

    anomaly_table = anomalies[["Order Date", "Sales"]].copy()
    anomaly_table = anomaly_table.sort_values("Sales", ascending=False)
    anomaly_table["Order Date"] = anomaly_table["Order Date"].dt.strftime("%d %b %Y")
    anomaly_table["Sales"] = anomaly_table["Sales"].map(lambda x: f"${x:,.2f}")

    st.dataframe(
        anomaly_table.rename(
            columns={"Order Date": "Week Ending", "Sales": "Weekly Sales"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    top3 = anomalies.nlargest(3, "Sales")
    top3_text = "<br>".join(
        [
            f"• <b>{row['Order Date']:%d %b %Y}</b> — {money(row['Sales'])}"
            for _, row in top3.iterrows()
        ]
    )

    st.markdown(
        f"""
        <div class="insight-box">
        <b>Top three high-sales anomalies</b><br>
        {top3_text}<br><br>
        These dates should be checked against promotion calendars, festive periods and
        large customer orders before treating the spikes as recurring demand.
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# PAGE 4 — PRODUCT DEMAND SEGMENTS
# =========================================================
elif page == "📦 Demand Segments":
    hero(
        "Product Demand Segments",
        "Group product sub-categories by sales volume, growth, volatility and order value to support smarter stocking decisions.",
    )

    cluster_order = [
        "High Volume, High Volatility",
        "Low Volume, Stable Demand",
        "High Volume, Stable Demand",
        "Rapidly Growing Demand",
    ]

    selected_segments = st.multiselect(
        "Filter demand segments",
        cluster_order,
        default=cluster_order,
    )

    segment_view = product_segments[
        product_segments["Demand Segment"].isin(selected_segments)
    ].copy()

    fig_cluster = px.scatter(
        segment_view,
        x="PCA1",
        y="PCA2",
        color="Demand Segment",
        text="Sub-Category",
        size="Total_Sales",
        size_max=28,
        hover_data={
            "PCA1": False,
            "PCA2": False,
            "Total_Sales": ":,.0f",
            "Sales_Volatility": ":,.0f",
            "Average_Order_Value": ":,.0f",
            "YoY_Growth": ":.1f",
        },
        title="Product Demand Segmentation using K-Means and PCA",
    )

    fig_cluster.update_traces(
        textposition="top center",
        marker=dict(line=dict(width=1, color="white")),
    )

    fig_cluster.update_layout(
        xaxis_title=f"Principal Component 1 ({pca_variance[0]*100:.1f}% of variation)",
        yaxis_title=f"Principal Component 2 ({pca_variance[1]*100:.1f}% of variation)",
        legend_title="Demand Segment",
        height=650,
    )

    st.plotly_chart(fig_cluster, use_container_width=True)

    st.caption(
        "Bubble size represents total sales. Products close together have similar demand "
        "characteristics. PCA is used only to display the four clustering features in two dimensions."
    )

    st.markdown("### Segment membership and stocking strategy")

    strategy_map = {
        "High Volume, High Volatility":
            "Maintain safety stock and refresh forecasts frequently to manage demand swings.",
        "Low Volume, Stable Demand":
            "Keep lean inventory and replenish in smaller, regular quantities.",
        "High Volume, Stable Demand":
            "Prioritize availability with frequent or automated replenishment.",
        "Rapidly Growing Demand":
            "Increase stock gradually and monitor growth closely to avoid premature overstocking.",
    }

    table = segment_view[
        [
            "Sub-Category",
            "Demand Segment",
            "Total_Sales",
            "Sales_Volatility",
            "YoY_Growth",
        ]
    ].copy()

    table["Stocking Strategy"] = table["Demand Segment"].map(strategy_map)
    table["Total_Sales"] = table["Total_Sales"].map(lambda x: f"${x:,.0f}")
    table["Sales_Volatility"] = table["Sales_Volatility"].map(lambda x: f"${x:,.0f}")
    table["YoY_Growth"] = table["YoY_Growth"].map(lambda x: f"{x:.1f}%")

    st.dataframe(table, use_container_width=True, hide_index=True)

    st.markdown("### Recommended inventory actions")

    cols = st.columns(2)
    for i, segment in enumerate(cluster_order):
        with cols[i % 2]:
            st.markdown(f"**{segment}**")
            st.write(strategy_map[segment])

    st.markdown(
        """
        <div class="insight-box">
        <b>Decision takeaway:</b> A single inventory policy should not be applied to every
        product. Stable high-volume products need dependable availability, volatile products
        need flexible safety stock, low-volume products need lean inventory, and fast-growing
        products need controlled inventory expansion.
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# FOOTER
# =========================================================
st.divider()
st.caption(
    "SupplyPulse • Sales Forecasting, Anomaly Detection & Product Demand Intelligence"
)
