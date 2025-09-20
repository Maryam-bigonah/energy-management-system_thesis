# Forecasting & Integration Module

## Overview

This module adds **prediction capabilities** to the energy management system, enabling:
- **Tomorrow's 24-hour forecasts** with automatic optimization
- **Next year's full simulation** (365 days) with all strategies
- **Fast cost prediction** using surrogate models (no solver required)

## 🎯 Module Components

### **Core Forecasting Scripts**

1. **`train_forecast_load.py`** - Load forecasting training
   - SARIMAX and XGBoost models
   - Rolling-origin cross-validation
   - Features: time, weather, lagged load, rolling means

2. **`train_forecast_pv.py`** - PV forecasting training
   - Two-stage model: Physical PR + XGBoost residuals
   - PR model: `pv_hat = kWp * PR * (GHI / GHI_ref) * (1 + α_T * (temp_C − 25))`
   - XGBoost on residuals: `pv_kw - pv_hat`

3. **`forecast_next_day.py`** - 24-hour prediction
   - Generates load, PV, and TOU forecasts
   - Runs optimization for all strategies
   - Outputs cost summary per strategy

4. **`simulate_next_year.py`** - Annual simulation
   - 365-day forecasting and optimization
   - Parallel processing support
   - Annual KPI aggregation

### **Optional Surrogate Models**

5. **`train_surrogate.py`** - Fast cost prediction training
   - Generates 1000+ diverse scenarios
   - Trains XGBoost models to predict costs
   - No solver required for predictions

6. **`predict_surrogate.py`** - Surrogate model usage
   - Instant cost predictions from features
   - Sample scenario testing
   - Batch prediction support

## 📁 File Structure

```
models/
├── load/
│   ├── sarimax_model.joblib
│   ├── xgboost_model.joblib
│   ├── scaler.joblib
│   └── meta.json
├── pv/
│   ├── pr_model.joblib
│   ├── xgboost_model.joblib
│   ├── scaler.joblib
│   └── meta.json
└── surrogate/
    ├── surrogate_MSC.joblib
    ├── surrogate_TOU.joblib
    ├── surrogate_MMR.joblib
    ├── surrogate_DRP2P.joblib
    ├── scaler_MSC.joblib
    ├── scaler_TOU.joblib
    ├── scaler_MMR.joblib
    ├── scaler_DRP2P.joblib
    └── meta.json

forecast/
├── nextday_load_24h.csv
├── nextday_pv_24h.csv
├── nextyear_load_2025.csv
├── nextyear_pv_2025.csv
└── nextyear_weather_2025.csv

results/
├── figs_forecast/
│   ├── load_forecast_validation.png
│   ├── pv_forecast_validation.png
│   └── surrogate_validation.png
└── kpis_forecast_2025.csv
```

## 🚀 Usage Examples

### **1. Training Phase**

```bash
# Train load forecasting models
python3 train_forecast_load.py --cv-splits 5

# Train PV forecasting models  
python3 train_forecast_pv.py --cv-splits 5

# Train surrogate models (optional)
python3 train_surrogate.py --n-scenarios 1000
```

### **2. Forecasting Phase**

```bash
# Forecast tomorrow and run optimization
python3 forecast_next_day.py --date 2025-01-17

# Simulate entire next year
python3 simulate_next_year.py --year 2025 --save-hourly

# Use surrogate models for fast prediction
python3 predict_surrogate.py --sample
```

### **3. Integration with Existing System**

The forecasting module **seamlessly integrates** with your existing Step 2 optimizer:

- **No changes required** to `run_day.py`
- **Same data formats** (CSV files, YAML specs)
- **Same optimization strategies** (MSC, TOU, MMR-P2P, DR-P2P)
- **Same output formats** (hourly results, KPIs)

## 🔧 Technical Implementation

### **Load Forecasting Features**
- **Time features**: hour (sin/cos), day-of-week, month, weekend flag
- **Weather features**: temperature, lagged temperature, rolling means
- **Load features**: lagged load (1h, 24h, 168h), rolling means (24h, 7d)
- **Calendar features**: holiday flags, seasonal indicators

### **PV Forecasting Features**
- **Physical model**: Performance ratio with temperature correction
- **Time features**: hour, day-of-year, seasonal patterns
- **Weather features**: GHI, temperature, clear-sky index
- **Residual modeling**: XGBoost on physical model residuals

### **TOU Tariff Generation**
- **ARERA F1/F2/F3 bands** with proper weekday/weekend/holiday mapping
- **Peak hours**: 08:00-19:00 (F1)
- **Flat hours**: 07:00-08:00, 19:00-23:00 (F2)
- **Valley hours**: 23:00-07:00 (F3)

### **Surrogate Model Features**
- **PV features**: total, peak, ramps, peak hour
- **Load features**: total, peak, ramps, peak hour
- **Price features**: mean, std, ratio, temporal patterns
- **Battery features**: capacity, power limits
- **Derived features**: ratios, net loads, self-consumption potential

## 📊 Output Examples

### **Next Day Forecast Summary**
```
============================================================
FORECAST SUMMARY FOR 2025-01-17
============================================================
Strategy        Cost (€)     Status    
----------------------------------------
MSC             458.32       Success   
TOU             458.32       Success   
MMR             461.86       Success   
DRP2P           447.21       Success   
============================================================
```

### **Annual Simulation Summary**
```
================================================================================
ANNUAL SIMULATION SUMMARY FOR 2025
================================================================================
Strategy        Annual Cost (€)    Mean Daily (€)    Std Daily (€)
----------------------------------------------------------------------
MSC             167,286.80        458.32           45.23
TOU             167,286.80        458.32           45.23
MMR             168,578.90        461.86           47.12
DRP2P           163,231.65        447.21           42.18
================================================================================
```

### **Surrogate Model Predictions**
```
============================================================
SAMPLE SCENARIO PREDICTIONS
============================================================
Strategy        Predicted Cost (€)
----------------------------------------
MSC             458.32
TOU             458.32
MMR             461.86
DRP2P           447.21
============================================================
```

## 🎯 Key Benefits

### **1. Operational Planning**
- **Day-ahead optimization** with accurate forecasts
- **Automatic strategy selection** based on predicted conditions
- **Cost estimation** before actual implementation

### **2. Long-term Analysis**
- **Annual simulation** for investment planning
- **Scenario analysis** with different weather patterns
- **Strategy comparison** across full year

### **3. Fast Decision Making**
- **Surrogate models** provide instant cost estimates
- **No solver required** for preliminary analysis
- **Batch processing** for multiple scenarios

### **4. Integration Benefits**
- **Seamless integration** with existing optimizer
- **Same data formats** and interfaces
- **Consistent results** across all components

## 🔍 Validation & Quality

### **Cross-Validation**
- **Rolling-origin CV** for time series data
- **MAE, RMSE, MAPE** metrics for all models
- **Statistical significance** testing

### **Model Selection**
- **Automatic best model** selection by MAE
- **Fallback mechanisms** for failed models
- **Robust error handling** throughout

### **Data Quality**
- **Synthetic data generation** when real data unavailable
- **Missing value handling** with realistic defaults
- **Outlier detection** and treatment

## 🚀 Next Steps

### **For Production Use**
1. **Train models** with your historical data
2. **Validate forecasts** against actual results
3. **Tune parameters** for your specific location
4. **Integrate** with your operational systems

### **For Research**
1. **Extend features** with additional weather variables
2. **Add uncertainty quantification** with ensemble methods
3. **Implement online learning** for model updates
4. **Develop advanced strategies** using forecast information

---

**The forecasting module transforms your energy management system from reactive to predictive, enabling proactive optimization and strategic planning.**
