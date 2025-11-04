# Project Context - Energy Forecasting System

## Goal/Purpose
Build an LSTM-based energy forecasting system for a residential building in Torino, Italy.

## Project Details

### Building Configuration
- **Location**: Torino, Italy
- **Units**: 20 apartments
- **Family Archetypes** (4 types):
  1. Couple working
  2. Family with one child
  3. One-working couple
  4. Retired couple

### Data Sources
- **PV Generation**: Hourly PV data from PVGIS API
- **Load Profiles**: Hourly load data from Load Profile Generator (LPG)
- **Time Resolution**: Hourly data
- **Time Period**: Full year 2024 (2024-01-01 to 2024-12-31 23:00)

### Forecasting Model
- **Model Type**: LSTM (Long Short-Term Memory) neural network
- **Framework**: TensorFlow/Keras
- **Forecasting Target**: 
  - Next-hour load
  - Next-hour PV generation
- **Input Features**:
  - Past 24 hours of historical data (load and PV)
  - Calendar features: hour, dayofweek, month, is_weekend
  - Season features: 0=winter, 1=spring, 2=summer, 3=autumn

### Technical Stack
- **Language**: Python
- **Data Processing**: Pandas
- **Machine Learning**: TensorFlow/Keras
- **Code Style**: Simple, readable, well-documented

### Additional Components
- **Battery System**: Shared building-level battery with allocation methods
- **Economic Analysis**: Italian grid tariffs (ARERA) and Feed-in-Tariff (GME)
- **Web Application**: Flask backend + React frontend for visualization

## Master Dataset Structure

The master dataset (`master_dataset_2024.csv`) contains:
- **20 apartment load columns**: `apartment_01` to `apartment_20`
- **PV column**: `pv_1kw` (1kW system generation)
- **Calendar columns**: `hour`, `dayofweek`, `month`, `is_weekend`
- **Season column**: `season` (0=winter, 1=spring, 2=summer, 3=autumn)
- **Index**: Hourly datetime index for full year 2024

## Code Guidelines
- Always use Python with pandas for data manipulation
- Use TensorFlow/Keras for LSTM implementation
- Keep code simple, readable, and well-commented
- Follow consistent naming conventions
- Include error handling and data validation

