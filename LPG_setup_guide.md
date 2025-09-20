# LoadProfileGenerator (LPG) Setup Guide

## Overview
Using the official [LoadProfileGenerator](https://www.loadprofilegenerator.de) from Forschungszentrum Jülich to create realistic household load profiles for our 20-unit apartment building.

## Step 1: Download and Install LPG

### Download
- Visit: https://www.loadprofilegenerator.de/download/
- Download the Windows 64-bit version
- Requirements: Windows 64-bit, 2GB+ RAM, several GB free space

### Installation
- Run the installer or extract ZIP archive
- Install to default location (recommended)

## Step 2: Define 4 Household Types

### Household Type 1: Working Couple with Appliances
**Configuration:**
- **Persons**: 2 adults (both working)
- **Work Schedule**: 8:00-17:00 (Monday-Friday)
- **Appliances**: 
  - High-efficiency appliances (dishwasher, washing machine, dryer)
  - Modern HVAC system
  - LED lighting
  - Electric vehicle (if applicable)
- **Behavior**: Peak consumption in morning (6-8h) and evening (18-22h)

### Household Type 2: Mixed Work Couple (One Working, One Home)
**Configuration:**
- **Persons**: 2 adults (1 working, 1 stay-at-home)
- **Work Schedule**: 1 person 8:00-17:00, 1 person home all day
- **Appliances**:
  - Standard appliances with daytime usage
  - Continuous HVAC operation
  - Home office equipment
- **Behavior**: Steady daytime consumption with evening peaks

### Household Type 3: Family with Children
**Configuration:**
- **Persons**: 2 adults + 2 children (ages 6-12)
- **Work Schedule**: Adults 8:00-17:00, children school 8:00-15:00
- **Appliances**:
  - Multiple devices (TVs, gaming consoles, computers)
  - Larger HVAC system
  - More frequent laundry/dishwashing
- **Behavior**: High consumption in morning, afternoon, and evening

### Household Type 4: Elderly Couple
**Configuration:**
- **Persons**: 2 adults (retired)
- **Work Schedule**: Home most of the day
- **Appliances**:
  - Traditional appliances
  - Conservative energy usage
  - Medical equipment (if needed)
- **Behavior**: Steady, moderate consumption throughout day

## Step 3: LPG Configuration Settings

### Location Settings
- **Location**: Rome, Italy (closest to Turin)
- **Weather Data**: Use Rome weather file for accurate solar/thermal calculations
- **Time Zone**: CET (Central European Time)

### Simulation Settings
- **Time Resolution**: 1 hour (for optimizer compatibility)
- **Simulation Period**: 1 full year (8760 hours)
- **Household Type**: Apartment
- **Building Type**: Multi-family residential

### Output Settings
- **Format**: CSV
- **Units**: kW (kilowatts)
- **File Naming**: Include household type and date range

## Step 4: Running Simulations

### Individual Household Simulations
1. Run each of the 4 household types separately
2. Generate 1-year profiles for each type
3. Export as CSV files

### Scaling to 20 Units
- **Distribution**:
  - 6x Working Couples (Type 1)
  - 5x Mixed Work Couples (Type 2) 
  - 6x Families with Children (Type 3)
  - 3x Elderly Couples (Type 4)

## Step 5: Data Processing

### File Structure
```
LPG_outputs/
├── household_type1_working_couple.csv
├── household_type2_mixed_work.csv
├── household_type3_family_children.csv
├── household_type4_elderly_couple.csv
├── aggregated_20units_8760h.csv
└── aggregated_20units_24h.csv
```

### Aggregation Process
1. Load all 4 household CSV files
2. Scale each type by its distribution (6, 5, 6, 3)
3. Sum hourly values across all 20 units
4. Create final aggregated files

## Expected Results

### Daily Consumption Patterns
- **Morning Peak**: 6:00-8:00 (breakfast, getting ready)
- **Daytime**: 8:00-17:00 (varies by household type)
- **Evening Peak**: 17:00-22:00 (cooking, entertainment)
- **Night**: 22:00-6:00 (minimal consumption)

### Seasonal Variations
- **Summer**: Higher cooling loads
- **Winter**: Higher heating loads
- **Spring/Fall**: Moderate consumption

## Quality Assurance

### Validation Checks
- Total daily consumption within realistic ranges (15-25 kWh per household)
- Peak demand timing matches typical residential patterns
- Seasonal variations align with heating/cooling needs
- No unrealistic spikes or negative values

### Comparison with Literature
- Compare with European residential consumption studies
- Validate against Building Data Genome Project data
- Check against utility company load profiles

