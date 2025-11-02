# Load Profile Generator File Analysis

## ✅ Your CSV File Analysis

**File**: `DeviceProfiles_3600s.Electricity.csv`  
**Path**: `/Users/mariabigonah/Desktop/thesis/CHR54 Retired Couple, no work/Results/DeviceProfiles_3600s.Electricity.csv`

### 📅 Date Range Information

**Your CSV File:**
- **Start Date**: **2016-01-01 00:00:00**
- **End Date**: **2016-12-31 23:00:00**
- **Total Hours**: **8,784 hours**
- **Days**: **366 days** (2016 was a leap year)
- **Year**: **2016**

✅ **This is a FULL YEAR of hourly data!**

### 📊 File Structure

- **Format**: Semicolon-separated (`;`)
- **Columns**: 
  - `Electricity.Timestep` - Timestep index (0, 60, 120, ... representing minutes)
  - `Time` - Date/time in format `MM/DD/YYYY HH:MM` (e.g., `1/1/2016 00:00`)
  - **73 device/appliance columns** - Each with `[kWh]` units
- **Data Type**: Hourly energy consumption (kWh per hour)
- **Load Range**: 0.0598 - 3.0550 kWh/hour (≈kW average power)

### 🔄 What the Script Does

1. **Reads the CSV** with semicolon separator
2. **Parses Time column** to datetime (format: `%m/%d/%Y %H:%M`)
3. **Sums all 73 device columns** to get total hourly load
4. **Converts year** from 2016 to 2024 (if needed) by matching day-of-year and hour
5. **Creates 20 apartments** by assigning 4 family types (5 apartments each)
6. **Adds PVGIS data** for PV generation
7. **Adds calendar features**: hour, dayofweek, month, is_weekend, season

---

## 📝 Master Dataset Output

**Target Year**: 2024  
**Output Date Range**:
- **Start**: **2024-01-01 00:00:00**
- **End**: **2024-12-31 23:00:00**
- **Total Hours**: **8,760 hours** (365 days, 2024 is not a leap year)

### Columns Created

1. **20 Apartment Loads**: `apartment_01` to `apartment_20`
   - Each with load values in kWh/hour (≈kW)

2. **PV Generation**: `pv_1kw`
   - PVGIS hourly power generation in kW

3. **Calendar Features**:
   - `hour`: 0-23
   - `dayofweek`: 0-6 (Monday=0, Sunday=6)
   - `month`: 1-12
   - `is_weekend`: 0 or 1

4. **Season**: `season`
   - 0 = winter (Dec, Jan, Feb)
   - 1 = spring (Mar, Apr, May)
   - 2 = summer (Jun, Jul, Aug)
   - 3 = autumn (Sep, Oct, Nov)

**Total**: 27 columns, 8,760 rows

---

## 🚀 How to Use

### 1. Update File Paths

In `build_master_dataset_final.py`, update:

```python
# PVGIS file path (you need to provide this)
pvgis_path = 'path/to/pvgis_torino_hourly.csv'

# Load Profile Generator file paths (4 family types)
lpg_paths = {
    'couple_working': 'path/to/couple_working/Results/DeviceProfiles_3600s.Electricity.csv',
    'family_one_child': 'path/to/family_one_child/Results/DeviceProfiles_3600s.Electricity.csv',
    'one_working': 'path/to/one_working/Results/DeviceProfiles_3600s.Electricity.csv',
    'retired': '/Users/mariabigonah/Desktop/thesis/CHR54 Retired Couple, no work/Results/DeviceProfiles_3600s.Electricity.csv'  # ✅
}
```

### 2. Run the Script

```bash
cd /Users/mariabigonah/Desktop/thesis/code
python3 build_master_dataset_final.py
```

### 3. Output

- Creates: `data/master_dataset_2024.csv`
- Ready for LSTM training!

---

## 📋 Summary

✅ **Found your LPG file format**  
✅ **Analyzed date range**: 2016-01-01 to 2016-12-31 (full year)  
✅ **Created script** to process it  
✅ **Script handles year conversion** (2016 → 2024)  
✅ **Creates 20 apartments** from 4 family types  
✅ **Adds all required features**

**Next Step**: Provide paths to:
1. PVGIS file (`pvgis_torino_hourly.csv`)
2. Other 3 family type files (DeviceProfiles_3600s.Electricity.csv)

Then run the script! 🚀

