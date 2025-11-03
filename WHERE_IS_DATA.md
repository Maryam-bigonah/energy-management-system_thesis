# 📊 Where is Your Data?

## Current Data Storage

**Important:** Your system uses **in-memory storage** (no traditional database yet). Here's where everything is:

### 1. **Master Dataset (CSV Files)**
**Location:** `data/master_dataset_2024.csv` (when created)

**Contains:**
- 20 apartment loads (apartment_01 to apartment_20)
- PV generation (pv_1kw)
- Calendar features (hour, dayofweek, month, is_weekend, season)

**How to create:**
```bash
python build_master_dataset_final.py
```

### 2. **Runtime Data (In Memory)**
**Location:** Backend server memory (when running)

**Stored in:**
- `df_master` - Master dataset (loaded from CSV or created)
- `battery_results` - Battery simulation results
- `economic_results` - Economic analysis results

**Note:** Data is lost when backend server stops!

### 3. **Source CSV Files**
**Location:** Your Desktop (`/Users/mariabigonah/Desktop/thesis/`)

- LPG files: `CHR54 Retired Couple, no work/Results/DeviceProfiles_3600s.Electricity.csv`
- PVGIS file: (you need to provide path)
- Building data: `building database/building-data-All-2025-10-06T12-26-43 3.csv`

---

## 🔍 How to See All Data

### Option 1: Web Interface (Complete View)

1. **Start servers:**
   ```bash
   # Terminal 1
   cd backend && python3 app.py
   
   # Terminal 2
   cd frontend && npm start
   ```

2. **Open in browser:**
   - Go to: http://localhost:3000/complete
   - Click tabs to see:
     - **Family Load Profiles** - Consumption by family type
     - **Battery & PV** - Energy storage from PV
     - **Economics** - Prices and costs

### Option 2: API Endpoints

**Family Consumption:**
```
GET http://localhost:5000/api/data/family-consumption
```

**Energy Storage:**
```
GET http://localhost:5000/api/data/storage-energy
```

**Complete Summary:**
```
GET http://localhost:5000/api/data/all-data-summary
```

### Option 3: Load Master Dataset from CSV

**Load your master dataset:**
```bash
POST http://localhost:5000/api/data/load-master-csv
Body: {"csv_path": "data/master_dataset_2024.csv"}
```

---

## 📋 What Data You'll See

### 1. **Family Consumption**
- Couple working (apartments 01-05)
- Family one child (apartments 06-10)
- One working (apartments 11-15)
- Retired (apartments 16-20)

**Shows:**
- Total consumption per family (kWh)
- Average consumption (kW)
- Hourly load profiles

### 2. **PV Energy Storage**
- Total PV generation (kWh)
- Excess PV (that can be stored)
- Self-consumption (PV used directly)
- Battery storage (if simulated)

### 3. **Battery Storage**
- State of Charge (SOC)
- Charge/discharge power
- Total energy stored (kWh)
- Grid import/export

### 4. **Prices**
- Grid import costs (€)
- Grid export revenues (€)
- Net cost (€)

---

## 💾 To Save Data Permanently

### Save Master Dataset:
```python
df_master.to_csv('data/master_dataset_2024.csv')
```

### Save Battery Results:
```python
battery_results.to_csv('data/battery_results.csv')
```

### Save Economic Results:
```python
economic_results.to_csv('data/economic_results.csv')
```

---

## 🎯 Quick Access

**See everything at once:**
- URL: http://localhost:3000/complete
- Tab: "Summary" - Shows all data in one table
- Tab: "Family Load Profiles" - Consumption breakdown
- Tab: "Battery & PV" - Energy storage details
- Tab: "Economics" - Price analysis

---

**Your data is ready to view! Open the Complete View page to see everything.** 🚀

