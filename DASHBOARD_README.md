# Energy Management System - Full-Stack Dashboard

A comprehensive web dashboard for visualizing energy data, analyzing features, and displaying PV forecasting results.

## Features

### 📊 Data Overview
- View statistics for all databases (PVGIS, OpenWeather, Merged)
- See row counts, column information, and date ranges
- Quick overview cards with key metrics

### 🔍 Feature Analysis
- Detailed statistics for each feature
- Mean, standard deviation, min, max, median
- Missing data information

### 📈 Covariance Matrix
- Interactive heatmap visualization
- Full covariance matrix table
- Shows relationships between all features

### 🖼️ Database Visualization
- Time series plots
- Distribution histograms
- Correlation matrices
- Daily patterns

### 🔮 Forecasting Results
- Run all three forecasting models (GradientBoosting, XGBoost, LSTM)
- Compare model performance
- Visualize 24-hour forecasts
- View detailed metrics (MAE, RMSE, nRMSE, R²)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your data files are in the correct location:
   - `/Users/mariabigonah/Desktop/thesis/building database/Timeseries_45.044_7.639_SA3_40deg_2deg_2005_2023.csv`
   - `/Users/mariabigonah/Desktop/thesis/building database/openweather_historical.csv`

## Running the Dashboard

Start the Flask backend server:

```bash
python start_dashboard.py
```

Or directly:

```bash
cd backend
python app.py
```

Then open your browser and navigate to:

```
http://localhost:5000
```

## API Endpoints

- `GET /api/data/overview` - Get database overview statistics
- `GET /api/data/features` - Get detailed feature information
- `GET /api/data/covariance` - Get covariance matrix and visualization
- `GET /api/data/visualization` - Get database visualization figures
- `GET /api/forecast/run` - Run all forecasting models
- `GET /api/forecast/visualization` - Get forecasting comparison visualization

## Project Structure

```
.
├── backend/
│   └── app.py              # Flask backend API
├── frontend/
│   └── index.html          # Dashboard frontend
├── src/
│   └── forecasting/        # Forecasting modules
├── start_dashboard.py      # Startup script
└── requirements.txt        # Python dependencies
```

## Notes

- The dashboard uses cached data for faster loading
- First-time data loading may take a few seconds
- Forecasting models may take several minutes to train
- XGBoost and LSTM require additional dependencies (see requirements.txt)

## Troubleshooting

**Port 5000 already in use:**
- Change the port in `backend/app.py`: `app.run(..., port=5001)`

**CORS errors:**
- Ensure `flask-cors` is installed: `pip install flask-cors`

**Data not loading:**
- Check that data file paths are correct in `backend/app.py`
- Verify data files exist and are readable

**Forecasting models not available:**
- Install missing dependencies: `pip install xgboost tensorflow`
- Check console output for specific error messages

