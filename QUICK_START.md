# Quick Start Guide - Dashboard

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Start the Server

```bash
python start_dashboard.py
```

## 3. Open Browser

Navigate to: **http://localhost:5000**

## 4. Explore the Dashboard

- **Data Overview**: See statistics for all your databases
- **Features**: Explore individual feature statistics
- **Covariance Matrix**: View feature relationships
- **Database Visualization**: See time series, distributions, and patterns
- **Forecasting Results**: Run and compare all three forecasting models

## Features Overview

### What You'll See:

1. **Database Statistics**
   - Row counts, column names, date ranges
   - Summary statistics for each database

2. **Feature Analysis**
   - Mean, std, min, max, median for each feature
   - Missing data information

3. **Covariance Matrix**
   - Heatmap showing feature relationships
   - Full covariance values table

4. **Database Visualizations**
   - PV power time series
   - Temperature and irradiance distributions
   - Feature correlations
   - Daily patterns

5. **Forecasting Results**
   - 24-hour forecast comparison (all models)
   - Performance metrics (MAE, RMSE, R²)
   - Side-by-side model comparison

## Troubleshooting

- **Server won't start**: Check that port 5000 is available
- **Data not loading**: Verify data file paths in `backend/app.py`
- **Forecasts not running**: Ensure XGBoost/TensorFlow are installed (optional)

Enjoy exploring your energy data! ⚡

