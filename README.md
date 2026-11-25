# SupplyPulse — Sales Forecasting & Demand Intelligence

An end-to-end retail analytics and forecasting project built on the Superstore sales dataset. The project combines exploratory data analysis, time-series forecasting, anomaly detection, product-demand segmentation, and an interactive Streamlit dashboard for business decision support.

## Project Highlights

- Analyzed four years of retail sales data and identified category, regional, seasonal, and shipping patterns.
- Compared three forecasting approaches: SARIMA, Prophet, and XGBoost.
- Selected SARIMA as the best-performing model based on evaluation metrics.
- Generated a 3-month sales forecast with 95% confidence ranges.
- Detected unusual weekly sales using Isolation Forest and rolling Z-score methods.
- Segmented product sub-categories using K-Means clustering and PCA.
- Built an interactive Streamlit dashboard with four business-focused pages.
- Prepared a two-page executive report for supply-chain and finance stakeholders.

## Dashboard Pages

### 1. Sales Overview
- Total sales by year
- Monthly sales trend
- Interactive year, region, and category filters
- Sales by region and category
- Seasonality comparison across years

### 2. Forecast Explorer
- Forecast by category or region
- Select a 1–3 month forecast horizon
- SARIMA forecast with 95% confidence range
- MAE and RMSE model-performance metrics
- Forecast details table

### 3. Anomaly Report
- Weekly sales anomaly visualization
- Isolation Forest anomaly detection
- Table of anomaly dates and sales values
- Highlighted high-sales anomalies

### 4. Product Demand Segments
- K-Means product-demand clustering
- PCA-based interactive cluster visualization
- Sub-category membership table
- Recommended stocking strategy for each demand segment

## Key Business Findings

- Technology generated the highest total revenue at approximately **$827,456**.
- The East region showed the most consistent sales growth across the analysis period.
- September, November, and December were consistently strong sales months.
- November recorded the highest average monthly sales at approximately **$87,540**.
- SARIMA achieved the best overall forecasting performance with an MAE of approximately **$18,031**, RMSE of **$19,009**, and MAPE of **18.97%**.
- The analysis detected **11 unusual sales weeks**.
- Product sub-categories were divided into four demand groups for differentiated stocking strategies.

## Project Structure

```text
SupplyPulse_Project/
├── app.py
├── analysis.ipynb
├── train.csv
├── summary.pdf
├── requirements.txt
├── README.md
├── .gitignore
└── charts/
    ├── 01_total_revenue_by_category.jpg
    ├── ...
    └── 19_product_demand_clusters_pca.jpg
```

## Run Locally

1. Clone the repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the Streamlit application:

```bash
streamlit run app.py
```

## Deployment

The application is designed for deployment on Streamlit Community Cloud. Ensure that `app.py`, `train.csv`, and `requirements.txt` are in the repository root before deploying.

## Tech Stack

Python, Pandas, NumPy, Matplotlib, Seaborn, Statsmodels, Prophet, XGBoost, Scikit-learn, Plotly, Streamlit

## Business Use Case

SupplyPulse helps retail teams understand historical sales performance, anticipate short-term demand, investigate unusual sales activity, and apply differentiated inventory strategies across product demand segments.

## Author

**Sachi Wadajkar**
