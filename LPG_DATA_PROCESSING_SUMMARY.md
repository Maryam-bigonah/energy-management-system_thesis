# Load Profile Generator (LPG) Data Processing Summary

## ✅ What Was Done

### 1. **Created Script to Process LPG Data** (`build_master_dataset_from_lpg.py`)

This script processes your Load Profile Generator CSV files and creates the master dataset.

**Key Features:**
- ✅ Reads LPG CSV files (semicolon-separated format)
- ✅ Processes minute-level data (1440 minutes = 24 hours)
- ✅ Aggregates to hourly load (averages 60 minutes per hour)
- ✅ Sums all appliance power values to get total load
- ✅ Repeats daily pattern for full year (if data is one day)
- ✅ Combines with PVGIS data
- ✅ Creates 20 apartments (4 family types × 5 apartments)
- ✅ Adds calendar columns (hour, dayofweek, month, is_weekend)
- ✅ Adds season column (0=winter, 1=spring, 2=summer, 3=autumn)

### 2. **Analyzed Your LPG Data Structure**

**From your file** (`TimeOfUseProfiles.Electricity.csv`):
- ✅ **Format**: Semicolon-separated (`;`)
- ✅ **Data Type**: Minute-level (1440 rows = 24 hours)
- ✅ **Time Column**: `Electricity.Time` (minute index 0-1439)
- ✅ **Calendar Column**: `Calender` (HH:MM:SS format)
- ✅ **Appliance Columns**: 39 device/appliance columns
- ✅ **Date Range**: One day pattern (will repeat for full year)

**Data Structure:**
```
Electricity.Time | Calender | Appliance1 | Appliance2 | ... | Appliance39
0                | 00:00:00 | 25 W       | 1 W        | ... | 369 W
1                | 00:01:00 | 23 W       | 3 W        | ... | 369 W
...
1439             | 23:59:00 | 24 W       | 0 W        | ... | 369 W
```

**Load Statistics (from your file):**
- Min: 4490 W (4.49 kW)
- Max: 5356 W (5.36 kW)
- Mean: 4716 W (4.72 kW)
- This represents hourly average power for a retired couple

### 3. **Date Range Information**

**Your LPG Data:**
- **Start**: 2024-01-01 00:00:00 (minute 0)
- **End**: 2024-01-01 23:59:00 (minute 1439)
- **Coverage**: **One day (24 hours)**
- **Will be repeated** for full year 2024 (365 days × 24 hours = 8,760 hours)

**Master Dataset Output:**
- **Start**: 2024-01-01 00:00:00
- **End**: 2024-12-31 23:00:00
- **Total Hours**: 8,760 hours (full year)
- **Frequency**: Hourly

---

## 📊 What the Script Does

### Step-by-Step Process:

1. **Load LPG CSV Files**
   - Reads semicolon-separated CSV
   - Identifies appliance columns (39 columns)
   - Sums all appliances → total load per minute

2. **Aggregate to Hourly**
   - Groups 60 minutes → 1 hour
   - Calculates average power per hour
   - Creates 24 hourly values (one day)

3. **Extend to Full Year**
   - Repeats daily pattern 365 times
   - Creates full year dataset (8,760 hours)

4. **Assign to 20 Apartments**
   - 4 family types:
     - `couple_working` → 5 apartments
     - `family_one_child` → 5 apartments
     - `one_working` → 5 apartments
     - `retired` → 5 apartments
   - Creates columns: `apartment_01` to `apartment_20`

5. **Combine with PVGIS**
   - Loads PVGIS hourly data
   - Aligns with master dataset dates
   - Creates `pv_1kw` column

6. **Add Calendar Features**
   - `hour`: 0-23
   - `dayofweek`: 0-6 (Monday=0, Sunday=6)
   - `month`: 1-12
   - `is_weekend`: 0 or 1

7. **Add Season**
   - `season`: 0=winter, 1=spring, 2=summer, 3=autumn
   - Based on month:
     - Dec, Jan, Feb → 0 (winter)
     - Mar, Apr, May → 1 (spring)
     - Jun, Jul, Aug → 2 (summer)
     - Sep, Oct, Nov → 3 (autumn)

---

## 🔧 How to Use

### 1. Update File Paths

In `build_master_dataset_from_lpg.py`, update these paths:

```python
# PVGIS file path
pvgis_path = 'path/to/your/pvgis_torino_hourly.csv'

# Load Profile Generator file paths (4 family types)
lpg_paths = {
    'couple_working': 'path/to/couple_working/TimeOfUseProfiles.Electricity.csv',
    'family_one_child': 'path/to/family_one_child/TimeOfUseProfiles.Electricity.csv',
    'one_working': 'path/to/one_working/TimeOfUseProfiles.Electricity.csv',
    'retired': '/Users/mariabigonah/Desktop/thesis/CHR54 Retired Couple, no work/Reports/TimeOfUseProfiles.Electricity.csv'
}
```

### 2. Run the Script

```bash
cd /Users/mariabigonah/Desktop/thesis/code
python3 build_master_dataset_from_lpg.py
```

### 3. Output

**Created Dataset:**
- `data/master_dataset_2024.csv`
- Shape: (8760, 27) - 8760 hours, 27 columns
- Columns:
  - 20 apartment loads (`apartment_01` to `apartment_20`)
  - `pv_1kw` (from PVGIS)
  - `hour`, `dayofweek`, `month`, `is_weekend`, `season`

---

## 📁 File Structure Expected

```
thesis/
├── code/
│   └── build_master_dataset_from_lpg.py
├── CHR54 Retired Couple, no work/
│   └── Reports/
│       └── TimeOfUseProfiles.Electricity.csv  ← Found ✅
├── [Other Family Types]/
│   └── Reports/
│       └── TimeOfUseProfiles.Electricity.csv  ← Need to find
└── pvgis_torino_hourly.csv  ← Need to provide path
```

---

## ❓ What You Need to Provide

1. **PVGIS File Path**:
   - Where is your `pvgis_torino_hourly.csv` file?
   - Should have columns: `time`, `pv_power`

2. **Other 3 Family Type Files**:
   - You have: **Retired couple** ✅
   - Need to find:
     - Couple working
     - Family with one child
     - One-working couple
   
   These should be in similar folders like:
   ```
   CHR[XX] [Family Type Name]/Reports/TimeOfUseProfiles.Electricity.csv
   ```

---

## ✅ What's Already Working

1. ✅ Script can read your LPG CSV format
2. ✅ Processes minute-level data correctly
3. ✅ Aggregates to hourly correctly
4. ✅ Extends one day to full year
5. ✅ Creates 20 apartments from 4 types
6. ✅ Adds calendar and season columns

**Once you provide the file paths, it will work!**

---

## 🎯 Final Output Format

The master dataset will have:

```python
Index: 2024-01-01 00:00:00 to 2024-12-31 23:00:00 (hourly)

Columns:
- apartment_01, apartment_02, ..., apartment_20  (load in kW)
- pv_1kw  (PV generation in kW)
- hour  (0-23)
- dayofweek  (0-6)
- month  (1-12)
- is_weekend  (0 or 1)
- season  (0, 1, 2, or 3)
```

**Total**: 27 columns, 8760 rows (full year hourly data)

---

Everything is ready! Just provide the file paths and run the script! 🚀

