# How to Get Tariffs and Prices Data (Italy)

## 📋 Economic Inputs Required

Your model needs three economic inputs:

1. **Grid electricity tariffs** (ARERA)
2. **Feed-in-Tariff (FiT)** (GME)
3. **P2P pricing model** (Custom/Research)

---

## 1. Grid Electricity Tariffs (ARERA)

### Source
**ARERA** - Italian Regulatory Authority for Energy, Networks and Environment
- **Website**: https://www.arera.it
- **English**: https://www.arera.it/en/index.htm

### How to Get Data

1. **Visit ARERA website**
2. **Navigate to**: Electricity Market → Tariffs
3. **Look for**:
   - Residential electricity tariffs
   - Time-of-use (TOU) pricing
   - Hourly/day-ahead prices (if available)

### Expected CSV Format

Save as: `data/tariffs_arera.csv`

```csv
time,tariff_euro_per_kwh
2024-01-01 00:00:00,0.20
2024-01-01 01:00:00,0.20
2024-01-01 02:00:00,0.20
2024-01-01 03:00:00,0.20
2024-01-01 08:00:00,0.25
2024-01-01 09:00:00,0.25
...
```

**Required columns:**
- `time`: Datetime (will be used as index)
- `tariff_euro_per_kwh`: Grid electricity tariff in €/kWh

### Typical Values (Italy, 2024)
- **Off-peak** (night): ~0.18-0.20 €/kWh
- **Peak** (day): ~0.25-0.30 €/kWh
- **Average**: ~0.22 €/kWh

### Alternative Sources
If ARERA doesn't provide hourly data:
- **Terna** (Italian grid operator): https://www.terna.it
- **GME** (Energy market operator): https://www.mercatoelettrico.org

---

## 2. Feed-in-Tariff (FiT) - GME

### Source
**GME** - Gestore dei Mercati Energetici (Italian Energy Market Operator)
- **Website**: https://www.mercatoelettrico.org
- **English**: Available on website

### How to Get Data

1. **Visit GME website**
2. **Navigate to**: Market Data → Feed-in-Tariffs or PV Incentives
3. **Look for**:
   - Current FiT rates
   - Historical FiT data
   - Hourly/day-ahead prices for renewable energy

### Expected CSV Format

Save as: `data/fit_gme.csv`

```csv
time,fit_euro_per_kwh
2024-01-01 00:00:00,0.12
2024-01-01 01:00:00,0.12
2024-01-01 02:00:00,0.12
2024-01-01 03:00:00,0.12
...
```

**Required columns:**
- `time`: Datetime (will be used as index)
- `fit_euro_per_kwh`: Feed-in-Tariff in €/kWh

### Typical Values (Italy, 2024)
- **FiT rate**: ~0.10-0.15 €/kWh (typically lower than grid tariff)
- Can vary by:
  - Installation date
  - System size
  - Market conditions

### Notes
- FiT may be constant or time-varying
- Check if FiT applies to all exported energy or only excess after self-consumption

---

## 3. P2P Pricing Model

### Description
**Peer-to-peer pricing** for energy trading between units within the building.

### Options

#### Option A: Constant P2P Price
Simple model: fixed price between grid tariff and FiT

```python
p2p_price = (grid_tariff + fit_rate) / 2
```

Example: (0.22 + 0.12) / 2 = 0.17 €/kWh

#### Option B: Dynamic P2P (Research Model)
Based on Liao et al. (2024) or similar P2P trading models:

```python
# Can be implemented as a function
def p2p_price(seller_unit, buyer_unit, timestamp, net_power_seller, net_power_buyer):
    # Custom logic based on:
    # - Supply/demand balance
    # - Unit characteristics
    # - Time of day
    # - Negotiation model
    ...
```

#### Option C: CSV File
Save P2P prices as CSV (if you have historical P2P market data):

```csv
time,p2p_price_euro_per_kwh
2024-01-01 00:00:00,0.15
2024-01-01 01:00:00,0.15
...
```

---

## 📂 File Structure

Once you have the data, place CSV files in `data/` directory:

```
data/
├── tariffs_arera.csv          # Grid electricity tariffs (ARERA)
├── fit_gme.csv                # Feed-in-Tariff (GME)
└── p2p_prices.csv             # P2P prices (optional)
```

---

## 💻 How to Use

### Step 1: Generate Templates

```python
from tariffs_model import create_tariffs_template, create_fit_template

# Create template files showing expected format
create_tariffs_template('data/tariffs_arera.csv')
create_fit_template('data/fit_gme.csv')
```

### Step 2: Fill Templates with Real Data

1. Download data from ARERA and GME
2. Format according to templates
3. Save as CSV files

### Step 3: Load in Model

```python
from tariffs_model import TariffsModel

# Load tariffs
tariffs = TariffsModel(
    tariffs_csv='data/tariffs_arera.csv',
    fit_csv='data/fit_gme.csv',
    p2p_pricing_model={'price_per_kwh': 0.15}  # or custom function
)

# Use with energy simulation
results, summary = tariffs.calculate_cost_revenue(df_energy_results)
```

---

## 📊 Integration with Energy Model

The tariffs model integrates with:

1. **Battery simulation**: Calculate costs/revenues
2. **Grid interaction**: Cost for imports, revenue for exports
3. **P2P trading**: Price for unit-to-unit transactions
4. **Economic analysis**: Total system costs and benefits

---

## ✅ Checklist

- [ ] Download grid tariffs from ARERA
- [ ] Download FiT from GME
- [ ] Format as CSV (matching templates)
- [ ] Save in `data/` directory
- [ ] Test loading with `TariffsModel`
- [ ] Integrate with energy simulation

---

## 🔗 Useful Links

- **ARERA**: https://www.arera.it
- **GME**: https://www.mercatoelettrico.org
- **Terna** (Grid operator): https://www.terna.it
- **Italian Energy Market**: https://www.mercatoelettrico.org

---

## 📝 Notes

1. **Time alignment**: Ensure tariff data matches your energy data time index (hourly, same year)
2. **Missing data**: The model handles missing timestamps by using nearest neighbor
3. **Currency**: All prices in Euro (€)
4. **Units**: All tariffs in €/kWh

---

**Once you provide the CSV files, the model will automatically use real Italian market prices!** 🇮🇹

