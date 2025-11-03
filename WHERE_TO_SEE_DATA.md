# 📊 Where to See All Your Data

## 🎯 What You Want to See

1. ✅ **Consumption of each family** (4 family types)
2. ✅ **Energy stored from PV** (PV generation)
3. ✅ **Battery** (SOC, charge/discharge)
4. ✅ **Price** (tariffs, costs, revenues)

---

## 🌐 Where to View: Complete Visualization Page

### **Main Page - Everything in One Place:**

**URL:** http://localhost:3000/complete

**OR:** Click "Complete View" in the navbar

---

## 📋 What Each Tab Shows

### Tab 1: **Overview**
- System summary statistics
- Total PV generation
- Total load consumption
- Battery summary

### Tab 2: **Family Load Profiles** ⭐
**This shows consumption of each family!**

- **Couple Working** (apartments 01-05)
- **Family One Child** (apartments 06-10)
- **One Working** (apartments 11-15)
- **Retired** (apartments 16-20)

**Shows:**
- Individual charts per family type
- Combined comparison chart
- Hourly/daily consumption patterns

### Tab 3: **Battery & PV** ⭐
**This shows PV energy storage and battery!**

- **PV Generation** (how much energy from PV)
- **Battery SOC** (State of Charge)
- **Battery Charge/Discharge** (how much energy stored/released)
- **Grid Import/Export** (net energy flow)
- **Energy stored from PV** = Battery charge when PV > Load

**To see this:**
1. Click "Run Battery Simulation" button if needed
2. View the charts showing PV vs Load vs Battery

### Tab 4: **Economics** ⭐
**This shows prices!**

- **Grid Import Cost** (€/kWh paid for importing)
- **Grid Export Revenue** (€/kWh earned for exporting)
- **Net Cost** (total cost - revenue)
- **Tariffs** (from ARERA/GME if loaded)

**Note:** Need to load tariffs CSV files first (see below)

### Tab 5: **Summary**
- Complete data table with all statistics
- All metrics in one place

---

## 💾 Where is the Database/Data?

### Current Setup:

**No traditional database** - Data is loaded in memory:

1. **Backend (Python):**
   - Data loaded when backend starts
   - Stored in memory as pandas DataFrames
   - Sample data created automatically

2. **Master Dataset File (if created):**
   - Location: `data/master_dataset_2024.csv`
   - Format: CSV file with all 20 apartments + PV + calendar features

3. **To Create Master Dataset:**

Run this script:
```bash
cd /Users/mariabigonah/Desktop/thesis/code
python3 build_master_dataset_final.py
```

This will create: `data/master_dataset_2024.csv`

---

## 🔍 How to See Each Family's Consumption

### Step 1: Go to Complete View
http://localhost:3000/complete

### Step 2: Click "Family Load Profiles" Tab

You'll see:
- **Chart 1:** All 4 family types together (comparison)
- **Chart 2:** Couple Working (individual)
- **Chart 3:** Family One Child (individual)
- **Chart 4:** One Working (individual)
- **Chart 5:** Retired (individual)

**Each chart shows:**
- Hourly consumption in kW
- Daily patterns
- Weekly patterns
- Peak consumption times

---

## 🔋 How to See PV Energy Storage

### Step 1: Go to Complete View
http://localhost:3000/complete

### Step 2: Click "Battery & PV" Tab

### Step 3: Click "Run Battery Simulation" Button

**You'll see:**
- **PV Generation Chart:** How much energy generated (green area)
- **Total Load Chart:** How much energy consumed (blue area)
- **Battery Charge:** How much energy stored from excess PV (green line)
- **Battery Discharge:** How much energy released when needed (red line)
- **Battery SOC:** Current battery level (0-100%)
- **Grid Import:** Energy bought from grid (when PV + Battery < Load)
- **Grid Export:** Energy sold to grid (when PV > Load + Battery capacity)

**Energy Stored from PV = Battery Charge (when PV > Load)**

---

## 💰 How to See Prices

### Step 1: Load Tariff Data (if available)

If you have tariff CSV files:
1. Place them in: `data/tariffs_arera.csv` and `data/fit_gme.csv`
2. Backend will use them automatically

### Step 2: Go to Complete View
http://localhost:3000/complete

### Step 3: Click "Economics" Tab

**You'll see:**
- **Import Cost Chart:** Cost per hour for buying from grid (€)
- **Export Revenue Chart:** Revenue per hour for selling to grid (€)
- **Net Cost:** Total cost - revenue (€)
- **Summary:** Total costs and revenues for the period

---

## 📊 All Data in One View

**Best page for everything:** http://localhost:3000/complete

**Tabs show:**
1. **Overview** - Quick stats
2. **Family Load Profiles** - Family consumption ⭐
3. **Battery & PV** - Energy storage ⭐
4. **Economics** - Prices ⭐
5. **Summary** - All data table

---

## 🔧 If Data Not Showing

### Issue: No family data shown

**Fix:** Master dataset needs to be created:
```bash
python3 build_master_dataset_final.py
```

### Issue: Battery simulation not working

**Fix:** Click "Run Battery Simulation" button in Battery & PV tab

### Issue: No price data

**Fix:** Load tariffs CSV files (see GET_TARIFFS_DATA.md)

---

## 📍 Quick Links

- **Complete View (All Data):** http://localhost:3000/complete
- **Dashboard:** http://localhost:3000/
- **Forecasts:** http://localhost:3000/forecasts
- **Training:** http://localhost:3000/training

---

## ✅ Summary

**To see everything:**
1. Open: http://localhost:3000/complete
2. Switch between tabs to see:
   - Family consumption → "Family Load Profiles" tab
   - PV energy storage → "Battery & PV" tab  
   - Prices → "Economics" tab
   - All together → "Summary" tab

**Data location:**
- In-memory (backend) or CSV files in `data/` folder
- Master dataset: `data/master_dataset_2024.csv` (if created)

**Everything is ready to view!** 🎉

