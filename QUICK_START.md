# Quick Start Guide - Torino Building LSTM Energy Forecasting

## 📋 Your Case

- **Location**: Torino, Italy
- **Building**: 1 building
- **Apartments**: 20 units
- **Archetypes**: 
  - Couple working (5 apartments)
  - Family with one child (5 apartments)
  - One-working couple (5 apartments)
  - Retired (5 apartments)
- **Data Sources**:
  - Hourly PV from PVGIS
  - Hourly load from Load Profile Generator
- **Forecast**: Next-hour load and PV using past 24 hours + calendar/season features

## 🚀 Quick Start (3 Steps)

### Step 1: Prepare Your Data

Ensure your CSV files have:
- **PVGIS file**: DateTime column + PV generation column
- **LPG file**: DateTime column + Load columns (for 20 apartments or total)

### Step 2: Run the Simple Example

```python
# Edit torino_lstm_example.py and update file paths:
# pv_df = load_pvgis_data('your_pvgis_data.csv')
# load_df = load_lpg_data('your_lpg_data.csv')

python torino_lstm_example.py
```

### Step 3: Use Full Stack Application

```bash
# Option 1: Docker
docker-compose up --build

# Option 2: Manual
cd backend && python app.py  # Terminal 1
cd frontend && npm start      # Terminal 2
```

## 📝 Code Structure

**Simple LSTM (Command Line):**
- `torino_lstm_example.py` - Simple script for quick forecasting
- `lstm_energy_forecast.py` - LSTM model class
- `data_loader_enhanced.py` - Enhanced data loading for your case

**Full Stack Application:**
- `backend/app.py` - Flask API
- `frontend/` - React dashboard

## 🎯 Key Features

1. **24-hour lookback**: Uses past 24 hours of data
2. **Calendar features**: Hour, day of week, month, season (sin/cos encoded)
3. **Dual output**: Forecasts both load and PV simultaneously
4. **Simple API**: Easy to use Python code

## 📊 Expected Output

```
Performance Metrics:

LOAD:
  MAE:  X.XXXX kW
  RMSE: X.XXXX kW
  R²:   X.XXXX
  MAPE: XX.XX%

PV:
  MAE:  X.XXXX kW
  RMSE: X.XXXX kW
  R²:   X.XXXX
  MAPE: XX.XX%

Next hour forecast:
  Load: XX.XX kW
  PV:   XX.XX kW
```

## 🔧 Customization

The model automatically:
- Creates calendar/season features
- Scales data (MinMaxScaler)
- Uses LSTM architecture (64 → 32 units)
- Forecasts next hour

To customize, edit `lstm_energy_forecast.py`.


