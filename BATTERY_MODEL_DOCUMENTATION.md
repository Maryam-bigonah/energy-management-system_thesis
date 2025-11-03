# Battery Model Documentation

## 📋 Battery Parameters (Table A2 - Residential Building)

### Specifications

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Capacity (E_b)** | Configurable | Battery capacity in kWh |
| **Charge Efficiency (η_ch)** | 0.9 | Charging efficiency (90%) |
| **Discharge Efficiency (η_dis)** | 0.9 | Discharging efficiency (90%) |
| **Life Cycles** | 10,000 | Number of charge/discharge cycles |
| **Cost (xb)** | $510/kWh | Battery cost per kWh |
| **SOCmin** | 0.2 | Minimum State of Charge (20%) |
| **SOCmax** | 0.95 | Maximum State of Charge (95%) |
| **P_ch,max** | Capacity × C-rate | Maximum charging power (kW) |
| **P_dis,max** | Capacity × C-rate | Maximum discharging power (kW) |

**Default C-rate**: 0.75 (conservative for residential)

---

## 🔋 SOC Update Equation

### Formula from Research Paper

```
SOC(t+1) = SOC(t) + ξ(t) * (P_b,ch(t) * Δt * η_ch / E_b)
          - (1 - ξ(t)) * (P_b,dis(t) * Δt / (E_b * η_dis))
```

### Parameters

- **SOC(t)**: State of charge at time t (0.0 to 1.0)
- **SOC(t+1)**: State of charge at next time step
- **ξ(t)**: Binary variable
  - 1 if charging
  - 0 if discharging
- **P_b,ch(t)**: Charging power at time t (kW)
- **P_b,dis(t)**: Discharging power at time t (kW)
- **Δt**: Time step (1 hour for hourly data)
- **η_ch = 0.9**: Charge efficiency
- **η_dis = 0.9**: Discharge efficiency
- **E_b**: Battery capacity (kWh)

### Constraint

**Cannot charge and discharge simultaneously**:
- Either ξ(t) = 1 (charging, P_b,dis = 0)
- Or ξ(t) = 0 (discharging, P_b,ch = 0)

---

## 💻 Implementation

### Files Created

1. **`battery_parameters.py`**
   - Stores battery parameters configuration
   - Function to get parameters for given capacity

2. **`battery_model.py`**
   - `BatteryModel` class implementing SOC equation
   - Functions for battery simulation
   - Integration with master dataset

### Usage Example

```python
from battery_model import BatteryModel

# Initialize 10 kWh battery at 50% SOC
battery = BatteryModel(capacity_kwh=10.0, initial_soc=0.5)

# Update SOC for one hour (charging at 2 kW)
soc_after = battery.update_soc(p_charge=2.0, p_discharge=0.0, delta_t=1.0)

# Or simulate over full dataset
from battery_model import simulate_battery_for_building
df_with_battery, battery = simulate_battery_for_building(
    df_master=df_master,
    battery_capacity_kwh=10.0,
    initial_soc=0.5
)
```

### Output Columns

When simulating with master dataset, adds:
- `battery_soc`: State of charge over time
- `battery_charge`: Charging power (kW)
- `battery_discharge`: Discharging power (kW)
- `net_power`: PV - Load (kW)
- `grid_import`: Power imported from grid (kW)
- `grid_export`: Power exported to grid (kW)

---

## 📊 Integration with Master Dataset

### Energy Flow

```
PV Generation → [Net Power] → Battery (if excess) OR Grid
Load Demand ← Battery (if deficit) OR Grid
```

### Logic

1. **Calculate Net Power**: `net_power = pv_1kw - total_building_load`

2. **If Net Power > 0** (Excess PV):
   - Charge battery (up to SOCmax)
   - Export remaining to grid

3. **If Net Power < 0** (Load > PV):
   - Discharge battery (down to SOCmin)
   - Import remaining from grid

4. **SOC Update**: Uses equation above

---

## 🔧 Configuration

### Default Parameters (Residential)

```python
BATTERY_PARAMS = {
    'charge_efficiency': 0.9,
    'discharge_efficiency': 0.9,
    'life_cycles': 10000,
    'cost_per_kwh': 510,
    'soc_min': 0.2,
    'soc_max': 0.95,
}
```

### Typical Capacities

- **Small residential**: 5-10 kWh
- **Medium residential**: 10-15 kWh
- **Large residential**: 15-20 kWh
- **Building scale**: 20-50 kWh+

---

## 📈 Battery Summary Statistics

After simulation, get summary:

```python
summary = battery.get_summary()
# Returns:
# {
#     'initial_soc': 0.5,
#     'final_soc': 0.68,
#     'min_soc': 0.20,
#     'max_soc': 0.95,
#     'avg_soc': 0.65,
#     'total_charge_energy': 1250.5 kWh,
#     'total_discharge_energy': 1125.3 kWh,
#     'energy_efficiency': 0.90
# }
```

---

## 📚 References

- **Table A2**: Battery specifications for residential building
- **SOC Equation**: Standard battery state update with efficiency losses
- **Citation**: "Table A2 – Specifications of the battery used in this study."

---

## ✅ Status

- ✅ Battery parameters stored
- ✅ SOC equation implemented
- ✅ Battery model class created
- ✅ Integration function for master dataset
- ✅ Ready for energy management system

**Files are ready to use with your master dataset!** 🚀

