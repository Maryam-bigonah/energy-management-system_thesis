# LoadProfileGenerator (LPG) Step-by-Step Checklist

## ✅ Pre-Setup
- [ ] Download LPG from https://www.loadprofilegenerator.de/download/
- [ ] Install on Windows 64-bit system
- [ ] Ensure 2GB+ RAM and several GB free space
- [ ] Create `LPG_outputs/` folder in project directory

## ✅ Household Type 1: Working Couple with Appliances
- [ ] Open LPG → Households → Add Household
- [ ] Name: "Working Couple with Appliances"
- [ ] **Persons**: 2 adults (both working)
- [ ] **Work Schedule**: 8:00-17:00 (Monday-Friday)
- [ ] **Appliances**:
  - [ ] High-efficiency dishwasher
  - [ ] Washing machine (evening use)
  - [ ] LED lighting throughout
  - [ ] Modern HVAC system
  - [ ] Electric vehicle charger (optional)
- [ ] **Behavior**: Peak consumption 6-8h and 18-22h
- [ ] **Location**: Rome, Italy
- [ ] **Building Type**: Apartment

## ✅ Household Type 2: Mixed Work Couple (One Working, One Home)
- [ ] Add Household → Name: "Mixed Work Couple"
- [ ] **Persons**: 2 adults (1 working, 1 stay-at-home)
- [ ] **Work Schedule**: 1 person 8:00-17:00, 1 person home all day
- [ ] **Appliances**:
  - [ ] Standard appliances with daytime usage
  - [ ] Continuous HVAC operation
  - [ ] Home office equipment
  - [ ] Kitchen appliances (daytime cooking)
- [ ] **Behavior**: Steady daytime consumption + evening peaks
- [ ] **Location**: Rome, Italy
- [ ] **Building Type**: Apartment

## ✅ Household Type 3: Family with Children
- [ ] Add Household → Name: "Family with Children"
- [ ] **Persons**: 2 adults + 2 children (ages 6-12)
- [ ] **Work Schedule**: Adults 8:00-17:00, children school 8:00-15:00
- [ ] **Appliances**:
  - [ ] Multiple TVs and gaming consoles
  - [ ] Larger HVAC system
  - [ ] Frequent laundry/dishwashing
  - [ ] Children's electronic devices
  - [ ] Kitchen appliances (more cooking)
- [ ] **Behavior**: High consumption morning, afternoon, evening
- [ ] **Location**: Rome, Italy
- [ ] **Building Type**: Apartment

## ✅ Household Type 4: Elderly Couple
- [ ] Add Household → Name: "Elderly Couple"
- [ ] **Persons**: 2 adults (retired)
- [ ] **Work Schedule**: Home most of the day
- [ ] **Appliances**:
  - [ ] Traditional appliances
  - [ ] Conservative energy usage
  - [ ] Medical equipment (if applicable)
  - [ ] Older, less efficient devices
- [ ] **Behavior**: Steady, moderate consumption throughout day
- [ ] **Location**: Rome, Italy
- [ ] **Building Type**: Apartment

## ✅ Simulation Configuration
- [ ] Go to Calculation section
- [ ] **Calculation Type**: Modular Household
- [ ] **Time Resolution**: 1 hour
- [ ] **Simulation Period**: 1 full year (8760 hours)
- [ ] **Location**: Rome, Italy
- [ ] **Weather Data**: Rome weather file
- [ ] **Output Format**: CSV
- [ ] **Output Directory**: `LPG_outputs/`

## ✅ Run Individual Simulations
- [ ] Run simulation for Household Type 1
  - [ ] Export as: `household_type1_working_couple.csv`
- [ ] Run simulation for Household Type 2
  - [ ] Export as: `household_type2_mixed_work.csv`
- [ ] Run simulation for Household Type 3
  - [ ] Export as: `household_type3_family_children.csv`
- [ ] Run simulation for Household Type 4
  - [ ] Export as: `household_type4_elderly_couple.csv`

## ✅ Data Processing
- [ ] Run Python script: `python3 process_lpg_outputs.py`
- [ ] Verify output files created:
  - [ ] `load_24h_lpg.csv` (24-hour profile)
  - [ ] `load_8760h_lpg.csv` (full year profile)
  - [ ] `plots/` directory with visualizations
- [ ] Check validation results in console output

## ✅ Quality Checks
- [ ] **Daily Consumption**: 300-600 kWh total (15-30 kWh per unit)
- [ ] **Peak Demand**: 40-80 kW total (2-4 kW per unit)
- [ ] **Load Factor**: 0.3-0.6 (typical residential)
- [ ] **Peak Times**: Morning (6-8h) and Evening (18-22h)
- [ ] **Seasonal Variation**: Higher in summer/winter

## ✅ Final Outputs
- [ ] `load_24h_lpg.csv` - Ready for optimizer
- [ ] `load_8760h_lpg.csv` - For annual analysis
- [ ] Validation plots for review
- [ ] Documentation of household distribution:
  - 6x Working Couples
  - 5x Mixed Work Couples
  - 6x Families with Children
  - 3x Elderly Couples

## 🔧 Troubleshooting
- **LPG won't start**: Check Windows 64-bit compatibility
- **Missing weather data**: Download Rome weather file from LPG
- **CSV format issues**: Ensure CSV export with timestamp and power columns
- **Unrealistic values**: Check household appliance configurations
- **Processing errors**: Verify all 4 CSV files exist in `LPG_outputs/`

## 📊 Expected Results Summary
- **Total Building Load**: ~475 kWh/day average
- **Peak Demand**: ~60 kW in evening
- **Load Diversity**: Good variation between household types
- **Seasonal Patterns**: Higher consumption in summer/winter
- **Realistic Profiles**: Based on actual European residential behavior

