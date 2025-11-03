# Shared Battery Model Explanation

## 🏢 Building-Level Battery System

### Concept

**In the Torino case, we assume the PV-battery system is installed at building level and shared between the 20 units; allocation of battery power to each unit follows a metric (e.g. energy share, floor area, or P2P SDR), in analogy with the P2P intra-community trading model proposed by Liao et al. (2024).**

### Why Shared Battery?

1. **Cost Efficiency**: One larger battery is more cost-effective than 20 small batteries
2. **Space Optimization**: Centralized installation saves space
3. **Better Utilization**: Aggregated demand/supply reduces battery cycling
4. **Grid Benefits**: Single point of grid interaction

### Architecture

```
Building with 20 Units
│
├── PV Generation (Building Level)
│
├── Shared Battery (Building Level)
│   └── Capacity: E_b kWh (configurable)
│   └── SOC: Tracked centrally
│
└── Allocation to Units
    ├── Unit 01 → Gets share based on allocation method
    ├── Unit 02 → Gets share based on allocation method
    ├── ...
    └── Unit 20 → Gets share based on allocation method
```

---

## 📊 Allocation Methods

### 1. Energy Share (`energy_share`)

**Principle**: Units with higher energy consumption get larger battery share.

**Calculation**:
```
Weight_i = Unit_i Load / Total Building Load
```

**Example**:
- Unit 01: 2.0 kW load
- Unit 02: 1.5 kW load
- Unit 03: 1.0 kW load
- Total: 4.5 kW

- Unit 01 weight: 2.0 / 4.5 = 0.444 (44.4%)
- Unit 02 weight: 1.5 / 4.5 = 0.333 (33.3%)
- Unit 03 weight: 1.0 / 4.5 = 0.222 (22.2%)

**When charging (excess PV)**:
- Total charge power: 5.0 kW
- Unit 01 gets: 5.0 × 0.444 = 2.22 kW
- Unit 02 gets: 5.0 × 0.333 = 1.67 kW
- Unit 03 gets: 5.0 × 0.222 = 1.11 kW

**When discharging (load > PV)**:
- Total discharge power: 3.0 kW
- Unit 01 gets: 3.0 × 0.444 = 1.33 kW
- Unit 02 gets: 3.0 × 0.333 = 1.00 kW
- Unit 03 gets: 3.0 × 0.222 = 0.67 kW

---

### 2. Floor Area (`floor_area`)

**Principle**: Units with larger floor area get larger battery share.

**Calculation**:
```
Weight_i = Unit_i Floor Area / Total Building Floor Area
```

**Example**:
- Unit 01: 50 m²
- Unit 02: 60 m²
- Unit 03: 70 m²
- Total: 180 m²

- Unit 01 weight: 50 / 180 = 0.278 (27.8%)
- Unit 02 weight: 60 / 180 = 0.333 (33.3%)
- Unit 03 weight: 70 / 180 = 0.389 (38.9%)

**Use Case**: Fair allocation based on physical space, independent of consumption patterns.

---

### 3. P2P SDR (`p2p_sdr`)

**Principle**: Peer-to-peer self-discharge ratio - units with higher energy deficits get higher battery priority.

**Reference**: Similar to P2P intra-community trading model by Liao et al. (2024)

**Calculation**:
```
Weight_i = Unit_i Energy Deficit / Total Energy Deficit
```

Where Energy Deficit = Unit Load when Load > PV

**Example**:
- Unit 01: 2.5 kW load, 1.0 kW PV → Deficit: 1.5 kW
- Unit 02: 1.8 kW load, 1.0 kW PV → Deficit: 0.8 kW
- Unit 03: 1.2 kW load, 1.0 kW PV → Deficit: 0.2 kW
- Total Deficit: 2.5 kW

- Unit 01 weight: 1.5 / 2.5 = 0.600 (60.0%)
- Unit 02 weight: 0.8 / 2.5 = 0.320 (32.0%)
- Unit 03 weight: 0.2 / 2.5 = 0.080 (8.0%)

**Use Case**: Prioritizes units with higher energy needs, similar to P2P energy trading.

---

## 🔋 Battery Operation

### SOC Update (Same as Individual Battery)

```
SOC(t+1) = SOC(t) + ξ(t) * (P_b,ch(t) * Δt * η_ch / E_b)
          - (1 - ξ(t)) * (P_b,dis(t) * Δt / (E_b * η_dis))
```

**Difference**: P_b,ch and P_b,dis are **building-level totals**, then allocated to units.

### Process Flow

1. **Calculate Building Net Power**:
   ```
   Net Power = Total PV - Total Building Load
   ```

2. **Determine Battery Operation** (building level):
   - If Net Power > 0 → Charge battery (excess PV)
   - If Net Power < 0 → Discharge battery (deficit)

3. **Allocate to Units**:
   - Calculate allocation weights based on method
   - Distribute charge/discharge power proportionally

4. **Update SOC**:
   - Use building-level total charge/discharge
   - Enforce SOC limits (20% - 95%)

5. **Grid Interaction**:
   - Grid Import = Remaining deficit after battery
   - Grid Export = Remaining excess after battery

---

## 💻 Implementation

### File: `shared_battery_model.py`

**Class**: `SharedBatteryModel`

**Key Methods**:
- `calculate_allocation_weights()`: Compute weights for each unit
- `allocate_battery_power()`: Distribute power to units
- `simulate_building()`: Full building simulation

### Usage Example

```python
from shared_battery_model import simulate_shared_battery_torino

# Load master dataset
df_master = pd.read_csv('data/master_dataset_2024.csv', index_col=0, parse_dates=True)

# Simulate with energy share allocation
results, model = simulate_shared_battery_torino(
    df_master=df_master,
    battery_capacity_kwh=20.0,  # Building-level: 20 kWh
    allocation_method='energy_share',
    initial_soc=0.5
)

# Results include:
# - Building-level: battery_soc, battery_charge_total, battery_discharge_total
# - Unit-level: apartment_XX_battery_charge, apartment_XX_battery_discharge
# - Allocation: apartment_XX_allocation_weight
# - Grid: grid_import, grid_export
```

### With Floor Area

```python
# Define floor areas for 20 units (m²)
floor_areas = [50, 60, 55, 65, 50,  # Units 1-5
               55, 60, 70, 65, 60,  # Units 6-10
               55, 50, 65, 60, 55,  # Units 11-15
               70, 65, 60, 55, 50, 65]  # Units 16-20

results, model = simulate_shared_battery_torino(
    df_master=df_master,
    battery_capacity_kwh=20.0,
    allocation_method='floor_area',
    floor_areas=floor_areas,
    initial_soc=0.5
)
```

---

## 📈 Output Columns

### Building-Level
- `building_total_load`: Total load of all 20 units (kW)
- `building_pv`: PV generation (kW)
- `building_net_power`: Net power (PV - Load) (kW)
- `battery_soc`: State of charge (0.0 to 1.0)
- `battery_charge_total`: Total charging power (kW)
- `battery_discharge_total`: Total discharging power (kW)
- `grid_import`: Power imported from grid (kW)
- `grid_export`: Power exported to grid (kW)

### Unit-Level (for each apartment_XX)
- `apartment_XX_battery_charge`: Charging power allocated to unit (kW)
- `apartment_XX_battery_discharge`: Discharging power allocated to unit (kW)
- `apartment_XX_allocation_weight`: Allocation weight (0.0 to 1.0)

---

## 📚 References

- **Battery Parameters**: Table A2 - Specifications of the battery used in this study (Residential Building)
- **SOC Equation**: Standard battery state update with efficiency losses
- **P2P Trading Model**: Liao et al. (2024) - P2P intra-community trading model

---

## ✅ Status

- ✅ Shared battery model implemented
- ✅ Three allocation methods available
- ✅ Building-level SOC tracking
- ✅ Unit-level allocation tracking
- ✅ Grid interaction calculation
- ✅ Ready for Torino building simulation

**Ready to use with your master dataset!** 🚀

