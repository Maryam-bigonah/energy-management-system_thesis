# Updated Feature List for Each Timestamp

## Master Dataset Features

### Target Variables (Forecast Output)

| Feature | Type | Why it helps |
|---------|------|-------------|
| `load` (building total) | numeric | Total building load (sum of 20 apartments) |
| `pv_1kw` | numeric | PV generation for forecasting |

### Input Features (Past 24 Hours)

#### Apartment Loads (20 apartments)
| Feature | Type | Why it helps |
|---------|------|-------------|
| `apartment_01` to `apartment_20` | numeric | Individual apartment loads for 20 units (4 family types × 5 each) |

#### PV Generation
| Feature | Type | Why it helps |
|---------|------|-------------|
| `pv_1kw` | numeric | Past PV generation data |

### Calendar Features

| Feature | Type | Why it helps |
|---------|------|-------------|
| `hour` | numeric | Captures daily cycle |
| `dayofweek` | numeric | Captures weekly pattern |
| `is_weekend` | binary | Differentiates weekend load |
| `month` | numeric | Month trend |
| `season` | categorical (0-3) | Captures seasonal pattern |

### Season Mapping

| Season Code | Season Name | Months |
|-------------|-------------|--------|
| 0 | Winter | December, January, February |
| 1 | Spring | March, April, May |
| 2 | Summer | June, July, August |
| 3 | Autumn | September, October, November |

## Complete Feature Summary

### Total Features: 27

1. **20 Apartment Loads** (`apartment_01` to `apartment_20`)
   - 4 family types: couple_working, family_one_child, one_working, retired
   - 5 apartments per type
   - Units: kWh/hour (≈kW average power)

2. **PV Generation** (`pv_1kw`)
   - PVGIS hourly generation
   - Units: kW

3. **Calendar Features** (5 features)
   - `hour`: 0-23 (hour of day)
   - `dayofweek`: 0-6 (Monday=0, Sunday=6)
   - `month`: 1-12
   - `is_weekend`: 0 or 1
   - `season`: 0-3 (winter, spring, summer, autumn)

## Feature Engineering Details

### Calendar Feature Encoding

**Hour** (0-23):
- Numeric representation of hour of day
- Captures daily consumption patterns

**Day of Week** (0-6):
- 0 = Monday
- 6 = Sunday
- Captures weekly patterns (workdays vs weekends)

**Month** (1-12):
- Numeric representation of month
- Captures monthly/seasonal trends

**Is Weekend** (0 or 1):
- Binary flag: 1 if Saturday/Sunday, 0 otherwise
- Differentiates weekend consumption patterns

**Season** (0-3):
- Categorical encoding:
  - 0 = Winter (Dec, Jan, Feb)
  - 1 = Spring (Mar, Apr, May)
  - 2 = Summer (Jun, Jul, Aug)
  - 3 = Autumn (Sep, Oct, Nov)
- Captures seasonal consumption patterns

## LSTM Input Configuration

### Lookback Window: 24 hours

**Input Shape**: (samples, 24, 27)
- 24 time steps (past 24 hours)
- 27 features per time step:
  - 20 apartment loads
  - 1 PV generation
  - 5 calendar features
  - 1 season

### Output: Next Hour Forecast

**Output Shape**: (samples, 2)
- `load`: Next hour building load (kW)
- `pv`: Next hour PV generation (kW)

## Usage in Model

```python
# Features used for LSTM
features = [
    # Past 24 hours of apartment loads
    'apartment_01', 'apartment_02', ..., 'apartment_20',
    
    # Past 24 hours of PV
    'pv_1kw',
    
    # Calendar features (for current hour)
    'hour', 'dayofweek', 'month', 'is_weekend', 'season'
]
```

## Data Range

- **Time Period**: 2024-01-01 00:00:00 to 2024-12-31 23:00:00
- **Frequency**: Hourly
- **Total Records**: 8,760 hours
- **Total Features**: 27 columns

