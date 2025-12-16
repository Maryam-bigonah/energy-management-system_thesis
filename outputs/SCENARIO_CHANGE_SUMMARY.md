# Household Profile Replacement Summary

**Date**: 2025-12-16  
**Change**: Replaced household profile #4 (family with 2 children) with a 3-children family profile (1 adult at work, 1 at home)

---

## What Changed

### Old Profile (Removed)
- **Type**: Family with 2 children  
- **Data file**: `SumProfiles_Family with 2 childrens.HH1.Apparent.csv` (Apparent power)  
- **Annual energy**: 4,320.4 kWh/year  
- **Peak hour**: 9:00 AM (0.883 kWh)  
- **Behavior**: Both adults working, children at school during day

### New Profile (Added)
- **Type**: Family with 3 children (1 adult at work, 1 at home)  
- **Data file**: `SumProfiles_3 children 1 at home,1 at work.HH1.Electricity.csv` (Electricity measurement)  
- **Annual energy**: 4,691.4 kWh/year  
- **Peak hour**: 7:00 PM / 19:00 (1.528 kWh)  
- **Behavior**: One adult home during day → higher daytime and evening consumption

---

## Impact on Building

### Energy Impact
- **Per household**: +371 kWh/year (+8.6%)  
- **Affected apartments**: 5 out of 20 (ap4, ap8, ap12, ap16, ap20)  
- **Building total**: +1,855 kWh/year across affected apartments

### Load Pattern Impact
- **Peak shift**: Morning (9:00) → Evening (19:00)  
- **Daytime demand**: Increased by 0.2–0.5 kWh/hour (10:00–16:00)  
- **Evening peak**: Increased by 0.6–0.8 kWh/hour (18:00–21:00)  
- **Nighttime**: Minimal change

### Implications for Energy Management
1. **Forecasting**: Peak hour shift affects model training; evening peak now more pronounced
2. **PV self-consumption**: Daytime load increase → better PV utilization
3. **Battery dispatch**: Evening peak requires battery to discharge during 18:00–21:00
4. **DR flexibility**: Higher peak load → ±10% band has larger absolute range
5. **Grid imports**: Evening peak may increase grid dependency if battery undersized

---

## Data Quality Improvements

### Why This Change Improves the Dataset
1. **Better measurement type**: Electricity (real power) vs Apparent power → more accurate
2. **Behavioral diversity**: Adds daytime-occupancy household → more realistic community
3. **Data source consistency**: All 4 profiles now from LPG Electricity measurements

---

## Pipeline Updates (All Completed)

✅ **Step 1**: Rebuilt master dataset with new profile  
✅ **Step 2**: Retrained forecasting models (24-hour ahead, load + PV)  
✅ **Step 3**: Re-ran day-ahead optimization (July 2023, 31 days)  
✅ **Step 4**: Regenerated all thesis figures (5 figures)  
✅ **Step 5**: Regenerated optimization figures (5 figures)  
✅ **Step 6**: Generated household comparison figures (2 figures)  
✅ **Step 7**: Validated entire pipeline (all checks passed)

---

## Figures Generated

### Household Comparison Figures (NEW)
1. `household_profile_comparison_daily.png` → Average daily load pattern comparison
2. `household_profile_comparison_difference.png` → Hourly difference (new - old)

### Thesis Figures (UPDATED)
1. `figure1_system_diagram.png` → System architecture
2. `figure2_timeseries_alignment.png` → Load vs PV vs Net Load (updated data)
3. `figure3_pv_relationships.png` → PV feature correlations (updated data)
4. `figure4_load_by_family_type.png` → **⚠️ NEW family type included**
5. `figure5_battery_operation_logic.png` → Battery constraints diagram

### Optimization Figures (UPDATED)
1. `figure_O1_baseline_vs_optimized_load.png`
2. `figure_O2_battery_operation_soc.png`
3. `figure_O3_grid_import_export.png`
4. `figure_O4_cost_comparison.png`
5. `figure_O5_supply_breakdown_priority.png`

---

## How to Document This in Your Thesis

### Recommended Paragraph (Copy/Paste)

> One household archetype was replaced to better represent families with daytime occupancy. 
> Specifically, the previous profile (family with 2 children, both adults working, measured as 
> apparent power) was replaced by a 3-children family profile where one adult works and one 
> adult remains at home (measured as electricity consumption). This replacement modifies the 
> temporal demand characteristics by increasing daytime consumption and shifting the peak 
> load from morning (9:00) to evening (19:00) hours. The new profile exhibits 8.6% higher 
> annual energy consumption (4,691 kWh/year vs 4,320 kWh/year), altering building-level 
> demand patterns and battery dispatch requirements. All downstream steps (forecasting, 
> optimization, MPC simulation, and KPI evaluation) were recomputed using the updated 
> household composition to ensure consistency and comparability across scenarios.

---

## Validation Results

✅ **All checks passed**:
- Forecast files: No NaNs, all horizons h1–h24 present
- Optimization: 24 hourly rows per day, non-negative flows
- DR constraints: 0 violations (±10% envelope respected)
- Energy balance: Max residual 3.8e-14 kWh (numerical precision)
- SoC bounds: Respected (5.0 → 25.0 kWh observed range)
- KPI integrity: All required fields present

---

## Dashboard Access

View all updated figures at:  
**http://localhost:5001/all-figures**

(Make sure Flask backend is running: `python3 backend/app.py`)

---

## Files Modified

### Code
- `src/forecasting/community_load_preprocessing.py` → Updated PATH_LPG_FAMILY_TWO_CHILDREN_DEFAULT
- `src/analysis/compare_household_profiles.py` → NEW script for comparison

### Data
- `/Users/mariabigonah/Desktop/thesis/building database/MASTER_20_APARTMENTS_2022_2023.csv` → Rebuilt
- `outputs/MASTER_20_APARTMENTS_2022_2023.csv` → Backup copy updated
- `outputs/forecasting_results/load_forecasts_2023.csv` → Regenerated
- `outputs/forecasting_results/pv_forecasts_2023.csv` → Regenerated
- `outputs/optimization/optimization_results.csv` → Regenerated
- `outputs/optimization/optimization_kpis.csv` → Regenerated

### Figures
- `outputs/figures/` → All 5 thesis figures regenerated
- `outputs/optimization/figures/` → All 5 optimization figures regenerated
- `outputs/figures/household_profile_comparison_*.png` → 2 new comparison figures

---

## Next Steps (If Needed)

If you want to test with different household compositions:
1. Modify the `FAMILY_ASSIGNMENTS` dictionary in `community_load_preprocessing.py`
2. Rerun: `python3 src/validation/run_all.py` (full pipeline)
3. Compare KPIs across scenarios

---

**Status**: ✅ Scenario change complete and validated. All results are consistent and ready for thesis.

