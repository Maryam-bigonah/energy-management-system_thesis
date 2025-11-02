# How to View PVGIS & Buildings Visualizations

## 📊 Generated Visualizations

The visualization script creates 4 image files showing different aspects of the PVGIS-Buildings integration:

### 1. Building Locations Map
**File**: `data/buildings_map.png`
- Shows all 10 buildings on a map
- Color = Building height
- Size = Roof area
- Labels = Building IDs

### 2. PVGIS Process Flow
**File**: `data/pvgis_process_flow.png`
- Diagram showing the complete PVGIS query process
- Step-by-step flow from CSV to PVGIS data

### 3. Sample PVGIS Data
**File**: `data/pvgis_data_sample.png`
- Shows hourly PV generation pattern (one week)
- Daily average pattern
- Demonstrates solar generation cycles

### 4. Integration Summary
**File**: `data/buildings_pvgis_integration.png`
- Complete overview with 4 panels:
  - Building locations
  - Building statistics
  - Integration explanation
  - Height distribution

---

## 🚀 How to Generate Visualizations

### Option 1: Run the Script

```bash
cd /Users/mariabigonah/Desktop/thesis/code
python3 visualize_buildings_pvgis.py
```

**Requirements:**
```bash
pip install matplotlib pandas
```

### Option 2: View Generated Images

After running the script, images are saved in `data/` folder:
```bash
# View in Finder
open data/

# Or list files
ls -lh data/*.png
```

---

## 📝 What Was Done - Summary

### ✅ Code Created

1. **`query_pvgis_for_buildings.py`** - Main PVGIS integration
   - Queries PVGIS API for each building
   - Processes coordinates
   - Saves individual CSV files

2. **`build_master_dataset.py`** - Dataset builder
   - Combines PVGIS + Load Profile Generator data
   - Creates master dataset for LSTM

3. **`visualize_buildings_pvgis.py`** - Visualization script
   - Generates 4 types of visualizations
   - Shows building locations, PVGIS process, and data

4. **Data Loaders** - Enhanced utilities
   - `data_loader.py` - Basic PVGIS loading
   - `data_loader_enhanced.py` - Enhanced with archetype support

### ✅ Documentation Created

1. **`PVGIS_BUILDINGS_EXPLANATION.md`** - Complete explanation
2. **`GET_PVGIS_DATA.md`** - Usage guide
3. **`HOW_TO_VIEW.md`** - Viewing instructions

---

## 🔍 Git Status

### What Was Pushed to GitHub ✅

All PVGIS integration code is in the repository:

```bash
# Check what's in git
git ls-files | grep -i pvgis
```

**Files in GitHub:**
- ✅ `query_pvgis_for_buildings.py` - **PUSHED**
- ✅ `build_master_dataset.py` - **PUSHED**  
- ✅ `data_loader.py` - **PUSHED**
- ✅ `data_loader_enhanced.py` - **PUSHED**
- ✅ `GET_PVGIS_DATA.md` - **PUSHED**

**New Files (Just Created):**
- ⏳ `visualize_buildings_pvgis.py` - **Ready to push**
- ⏳ `PVGIS_BUILDINGS_EXPLANATION.md` - **Ready to push**
- ⏳ `HOW_TO_VIEW.md` - **Ready to push**

---

## 📖 Quick Explanation

### What the PVGIS Integration Does:

1. **Reads Building CSV**
   - Extracts Latitude/Longitude for each building
   - Your file: `building-data-All-2025-10-06T12-26-43 3.csv`
   - 10 buildings in Torino area

2. **Queries PVGIS API**
   - API: https://re.jrc.ec.europa.eu/api/hourly
   - Gets hourly PV generation data (2024)
   - Parameters: coordinates, PV system specs

3. **Processes Data**
   - Converts time format (YYYYMMDD:HH → datetime)
   - Converts units (W → kW)
   - Creates hourly DataFrame

4. **Saves Results**
   - Individual CSV per building
   - Format: time, pv_power columns
   - Saved to `data/pvgis_buildings/`

5. **Integrates with Master Dataset**
   - PVGIS data → Master dataset builder
   - Combines with load data
   - Ready for LSTM training

---

## 🎯 Integration Flow

```
Building CSV (with coordinates)
         ↓
Extract Lat/Lon for each building
         ↓
Group buildings by similar coordinates
         ↓
Query PVGIS API (or use existing file)
         ↓
Get hourly PV generation data
         ↓
Save individual CSV files
         ↓
Use in master dataset builder
         ↓
Train LSTM model
```

---

## 📸 View Visualizations

**After running the visualization script:**

```bash
# Images are saved in data/ folder
open data/buildings_map.png
open data/pvgis_process_flow.png
open data/pvgis_data_sample.png
open data/buildings_pvgis_integration.png
```

**Or view in GitHub** (after pushing):
- Go to: https://github.com/Maryam-bigonah/energy-management-system-
- Navigate to `data/` folder
- Click on any `.png` file

---

**Everything is documented and ready to visualize!** 🎉

