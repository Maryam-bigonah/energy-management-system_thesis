# Load Profile Generator (LPG) - Real Data Guide

## 🚨 **CRITICAL: Current Load Data is NOT from Real LPG**

The current load data is **GENERATED/SIMULATED** and must be replaced with **REAL LPG data**.

## ✅ **REQUIRED: Use Real LPG Software**

### **Step 1: Download LPG Software**
1. **Go to**: https://www.loadprofilegenerator.de
2. **Download**: Windows desktop application (LPG is Windows-only)
3. **Install**: Run the installer and follow setup instructions
4. **Launch**: Start the LPG application

### **Step 2: Configure 4 Household Types**

#### **Household Type 1: Working Couple**
- **Occupants**: 2 adults, both working
- **Appliances**: Standard apartment appliances
- **Schedule**: Both work 9-17h, home evenings/weekends
- **Location**: Rome, Italy (closest to Turin)

#### **Household Type 2: Mixed Work**
- **Occupants**: 2 adults, one working, one at home
- **Appliances**: Standard + home office equipment
- **Schedule**: One works 9-17h, other at home all day
- **Location**: Rome, Italy

#### **Household Type 3: Family with Children**
- **Occupants**: 2 adults + 2 children (school age)
- **Appliances**: Standard + children's appliances
- **Schedule**: Children at school 8-16h, adults work
- **Location**: Rome, Italy

#### **Household Type 4: Elderly Couple**
- **Occupants**: 2 adults, both retired
- **Appliances**: Standard + medical equipment
- **Schedule**: Home most of the time
- **Location**: Rome, Italy

### **Step 3: LPG Configuration**
- **Building Type**: Apartment
- **Resolution**: 1 hour
- **Simulation Period**: 1 full year (8760 hours)
- **Location**: Rome, Italy (45.4°N, 12.5°E)
- **Weather Data**: Use LPG's built-in weather data

### **Step 4: Export Data**
For each household type:
1. **Run simulation** for 1 year
2. **Export as CSV** format
3. **Save as**:
   - `LPG_outputs/household_type1_working_couple.csv`
   - `LPG_outputs/household_type2_mixed_work.csv`
   - `LPG_outputs/household_type3_family_children.csv`
   - `LPG_outputs/household_type4_elderly_couple.csv`

### **Step 5: Process Real LPG Data**
After getting real LPG outputs, run:
```bash
python3 process_lpg_outputs.py
```

This will:
- Load the real LPG CSV files
- Aggregate to 20 units (6×Type1, 4×Type2, 5×Type3, 5×Type4)
- Create `load_24h_real.csv` and `load_8760_real.csv`

## 🔍 **Current Status**

❌ **Current data is FAKE/GENERATED**
✅ **Need to replace with REAL LPG data**

## 📋 **Action Required**

1. **Download LPG**: https://www.loadprofilegenerator.de
2. **Configure 4 household types** as specified above
3. **Run simulations** for 1 year each
4. **Export CSV files** for each type
5. **Process with** `process_lpg_outputs.py`

## ⚠️ **Important Notes**

- LPG is Windows-only software
- You need to manually configure each household type
- The simulation takes time (several minutes per household)
- Export format must be CSV with hourly data
- Data should be in kW (not W)

## 🎯 **Expected Output**

After using real LPG:
- `project/data/load_24h_real.csv` - Real 24-hour building load
- `project/data/load_8760_real.csv` - Real yearly building load
- Data will be from actual LPG simulations, not generated patterns

## 📊 **Validation**

Real LPG data should show:
- **Peak load**: 20-40 kW for 20 units
- **Daily pattern**: Clear morning/evening peaks
- **Seasonal variation**: Higher consumption in winter
- **Weekend patterns**: Different from weekday patterns

---

**This is the ONLY way to get real load data for your thesis!**

