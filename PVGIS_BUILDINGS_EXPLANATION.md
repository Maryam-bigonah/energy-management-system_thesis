# PVGIS & Buildings Integration - What Was Done

## 📍 Overview

This document explains what was implemented for the **PVGIS and Buildings integration** part of the energy forecasting system.

---

## 🎯 What Was Done

### 1. **Building Data Integration** (`query_pvgis_for_buildings.py`)

**Purpose**: Query PVGIS API for each building using their coordinates to get hourly PV generation data.

**What it does**:
- ✅ Reads building CSV with Latitude/Longitude coordinates
- ✅ Groups buildings by similar coordinates (~100m radius) to avoid duplicate queries
- ✅ Queries PVGIS API for hourly PV generation data (2024)
- ✅ Saves individual CSV files for each building with PV data
- ✅ Can use existing Torino PVGIS file if all buildings are in same area

**Key Features**:
```python
# Main function: process_buildings_from_csv()
- Reads: /Users/mariabigonah/Desktop/thesis/building database/building-data-All-2025-10-06T12-26-43 3.csv
- Extracts: Latitude, Longitude for each building
- Queries: PVGIS API (https://re.jrc.ec.europa.eu/api/hourly)
- Outputs: CSV files in data/pvgis_buildings/
```

### 2. **PVGIS API Integration**

**API Endpoint Used**:
```
https://re.jrc.ec.europa.eu/api/hourly
```

**Parameters Sent**:
- `lat`, `lon`: Building coordinates
- `startyear`, `endyear`: 2024
- `peakpower`: 1 kWp (default)
- `loss`: 14% system loss
- `angle`: 30° slope
- `aspect`: 180° azimuth (south-facing)
- `pvtechchoice`: Crystalline Silicon

**Response Format**:
```json
{
  "outputs": {
    "hourly": [
      {
        "time": "20240101:00",
        "P": 0.0  // Power in Watts
      },
      ...
    ]
  }
}
```

### 3. **Data Processing**

**What happens to PVGIS data**:
1. **Parse time strings**: Convert "YYYYMMDD:HH" to pandas datetime
2. **Convert units**: W → kW (divide by 1000)
3. **Create DataFrame**: Hourly index with `pv_power` column
4. **Save per building**: Individual CSV file for each building

**Output Format**:
```csv
time,pv_power
2024-01-01 00:00:00,0.0
2024-01-01 01:00:00,0.0
...
2024-01-01 12:00:00,0.85
...
```

### 4. **Integration with Master Dataset**

**How it connects**:
- PVGIS data files are used in `build_master_dataset.py`
- Each building gets its own PV generation data
- Combined with Load Profile Generator data
- Creates complete dataset for LSTM training

---

## 📊 Building Data Structure

**Input CSV** (`building-data-All-2025-10-06T12-26-43 3.csv`):
```
Columns:
- OSM ID: Building identifier (e.g., way/49062146)
- Latitude: 45.0447177 (Torino area)
- Longitude: 7.6367993 (Torino area)
- Roof Area (m²): Available roof space for PV
- Height (m): Building height
- Apartments: Number of apartments
- ... other building metadata
```

**10 Buildings Currently**:
- All in Torino, Italy
- Coordinates range: 45.0436° - 45.0450° N, 7.6359° - 7.6399° E
- Similar coordinates → can use one PVGIS query for all

---

## 🔄 Process Flow

```
1. Read Building CSV
   ↓
2. Extract Coordinates (Latitude, Longitude)
   ↓
3. Group Buildings by Similar Coordinates (~100m)
   ↓
4. For Each Group:
   - Query PVGIS API (or use existing Torino file)
   - Get hourly PV generation data
   ↓
5. Save Individual CSV Files
   - data/pvgis_buildings/pvgis_building_1.csv
   - data/pvgis_buildings/pvgis_building_2.csv
   - ...
   ↓
6. Ready for Master Dataset Integration
```

---

## 🎨 Visualization

**Created Visualization Script**: `visualize_buildings_pvgis.py`

**What it shows**:
1. **Building Map**: Scatter plot of all buildings with coordinates
2. **Process Flow**: Diagram showing PVGIS query process
3. **Sample PVGIS Data**: Hourly PV generation patterns
4. **Integration Summary**: Complete overview of building-PVGIS connection

**To generate visualizations**:
```bash
python visualize_buildings_pvgis.py
```

**Output Images**:
- `data/buildings_map.png` - Building locations
- `data/pvgis_process_flow.png` - Process diagram
- `data/pvgis_data_sample.png` - Sample PV data
- `data/buildings_pvgis_integration.png` - Complete overview

---

## ✅ What Was Pushed to Git

### Files in GitHub Repository:

1. **`query_pvgis_for_buildings.py`** ✅
   - Main PVGIS integration script
   - API querying functions
   - Building processing functions

2. **`build_master_dataset.py`** ✅
   - Uses PVGIS data in master dataset
   - Combines PVGIS + Load Profile Generator data

3. **`data_loader_enhanced.py`** ✅
   - Enhanced data loading utilities
   - PVGIS file reading functions

4. **`GET_PVGIS_DATA.md`** ✅
   - Documentation for PVGIS integration
   - Usage instructions

5. **`visualize_buildings_pvgis.py`** ⚠️ (Just created - needs to be pushed)

### Git Status Check:

```bash
# Check if PVGIS files are in git
git ls-files | grep pvgis
```

**Current Status**:
- ✅ `query_pvgis_for_buildings.py` - **PUSHED**
- ✅ `build_master_dataset.py` - **PUSHED**
- ✅ `data_loader_enhanced.py` - **PUSHED**
- ✅ `GET_PVGIS_DATA.md` - **PUSHED**
- ⏳ `visualize_buildings_pvgis.py` - **NEW, needs push**
- ⏳ `PVGIS_BUILDINGS_EXPLANATION.md` - **NEW, needs push**

---

## 🔧 How to Use

### Option 1: Use Existing Torino PVGIS File

If you have `pvgis_torino_hourly.csv`:

```python
# In query_pvgis_for_buildings.py
use_existing_torino = True
torino_pvgis_path = 'data/pvgis_torino_hourly.csv'

python query_pvgis_for_buildings.py
```

### Option 2: Query PVGIS API for Each Building

```python
# In query_pvgis_for_buildings.py
use_existing_torino = False

python query_pvgis_for_buildings.py
```

**Note**: PVGIS API has rate limits, so querying many buildings takes time.

### Option 3: Visualize

```bash
python visualize_buildings_pvgis.py
```

---

## 📈 Results

**What you get**:
- ✅ Individual PVGIS CSV files for each building
- ✅ Hourly PV generation data (2024)
- ✅ Data ready for LSTM forecasting
- ✅ Visualizations showing integration

**Integration Points**:
- Building coordinates → PVGIS API → Hourly PV data
- PVGIS data → Master dataset → LSTM model training
- Combined data → Energy forecasting

---

## 🚀 Next Steps

1. **Generate Visualizations**:
   ```bash
   python visualize_buildings_pvgis.py
   ```

2. **Query PVGIS** (if needed):
   ```bash
   python query_pvgis_for_buildings.py
   ```

3. **Use in Master Dataset**:
   ```bash
   python build_master_dataset.py
   ```

4. **Train LSTM Model**:
   ```bash
   python torino_lstm_example.py
   ```

---

## 📝 Summary

**What was implemented**:
- ✅ PVGIS API integration for building energy data
- ✅ Building coordinate processing
- ✅ Individual PVGIS data files per building
- ✅ Integration with master dataset builder
- ✅ Visualization tools

**Status**: 
- ✅ **Code pushed to GitHub** (except new visualization script)
- ✅ **Documentation created**
- ✅ **Ready to use**

---

**Repository**: https://github.com/Maryam-bigonah/energy-management-system-

All PVGIS integration code is in the repository! 🎉

