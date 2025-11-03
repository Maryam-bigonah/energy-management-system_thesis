# Shared Battery Model Documentation - Building-Level PV-Battery System

## 📋 Overview

In the Torino case, we assume the **PV-battery system is installed at building level** and **shared between the 20 units**. Allocation of battery power to each unit follows a metric (e.g. energy share, floor area, or P2P SDR), in analogy with the **P2P intra-community trading model proposed by Liao et al. (2024)**.

---

## 🏢 Building-Level Shared Battery System

### Concept

- **Single battery** installed at building level
- **Shared** among all 20 units
- **Allocation** of battery power follows a specified metric
- **Analogous** to P2P intra-community energy trading

### Architecture

```
                     ┌─────────────────┐
                     │  Building-Level │
                     │  Shared Battery │
                     │   (10-50 kWh)   │
                     └────────┬────────┘
                              │
                ┌─────────────┼─────────────┐
                │             │             │
          ┌─────▼──┐    ┌────▼──┐    ┌────▼──┐
          │ Unit 1 │    │Unit 2 │ ... │Unit 20│
          └────────┘    └───────┘    └───────┘
```

---

## 🔋 Battery Parameters (Table A2 - Residential)

Same as individual battery model:

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Capacity (E_b)** | Configurable | Building-level battery capacity (kWh) |
| **Charge Efficiency (η_ch)** | 0.9 | Charging efficiency (90%) |
| **Discharge Efficiency (η_dis)** | 0.9 | Discharging efficiency (90%) |
| **Life Cycles** | 10,000 | Number of charge/discharge cycles |
| **Cost (xb)** | $510/kWh | Battery cost per kWh |
| **SOCmin** | 0.2 | Minimum State of Charge (20%) |
| **SOCmax** | 0.95 | Maximum State of Charge (95%) |
| **P_ch,max / P_dis,max** | Capacity × C-rate | Maximum power (default C-rate: 0.75) |

---

## 📊 Allocation Methods

### 1. Energy Share (`energy_share`)

**Allocation based on unit's energy consumption share**

- Higher consumption → larger share of battery
- Weights proportional to load

**Formula:**
```
weight_i = load_i / Σ(load_j)
```

**Use Case:**
- Fair allocation based on actual energy needs
- Units with higher loads get more battery access

**Example:**
- Unit 1: 2 kW load → weight = 0.10 (10%)
- Unit 2: 3 kW load → weight = 0.15 (15%)
- ...

---

### 2. Floor Area (`floor_area`)

**Allocation based on unit's floor area share**

- Larger floor area → larger share of battery
- Weights proportional to area

**Formula:**
```
weight_i = area_i / Σ(area_j)
```

**Use Case:**
- Fair allocation based on apartment size
- Reflects property ownership structure

**Example:**
- Unit 1: 50 m² → weight = 0.05 (5%)
- Unit 2: 80 m² → weight = 0.08 (8%)
- ...

---

### 3. P2P SDR (`p2p_sdr`)

**Allocation based on P2P SDR (Self-Discharge Rate) metric**

- Higher SDR → higher priority for battery access
- Reflects P2P trading behavior (Liao et al., 2024)

**Formula:**
```
weight_i = SDR_i / Σ(SDR_j)
```

**Use Case:**
- P2P energy trading scenarios
- Reward units with better trading behavior
- Analogous to P2P intra-community trading model

**Example:**
- Unit 1: SDR = 0.8 → weight = 0.04 (4%)
- Unit 2: SDR = 1.2 → weight = 0.06 (6%)
- ...

---

### 4. Equal Allocation (`equal`)

**Equal allocation among all units**

- Each unit gets equal share (1/20 = 5%)

**Formula:**
```
weight_i = 1 / N
```

**Use Case:**
- Simple, fair baseline
- Equal rights for all units

---

## 💻 Implementation

### Files Created

1. **`shared_battery_model.py`**
   - `SharedBatteryModel` class
   - Implements allocation methods
   - Integrates with master dataset

### Usage Example

```python
from shared_battery_model import SharedBatteryModel, simulate_shared_battery_for_building

# Option 1: Use convenience function
results, battery = simulate_shared_battery_for_building(
    df_master=df_master,
    battery_capacity_kwh=20.0,  # Building-level capacity
    allocation_method='energy_share',
    initial_soc=0.5
)

# Option 2: Manual initialization
shared_battery = SharedBatteryModel(
    capacity_kwh=20.0,
    num_units=20,
    initial_soc=0.5,
    allocation_method='energy_share'
)

# With floor areas
floor_areas = [50, 60, 70, ...]  # m² per unit
results = shared_battery.simulate_for_building(
    df_master,
    floor_areas=floor_areas,
    allocation_method='floor_area'
)

# With P2P SDR
p2p_sdr = [0.8, 1.0, 1.2, ...]  # P2P SDR per unit
results = shared_battery.simulate_for_building(
    df_master,
    p2p_sdr=p2p_sdr,
    allocation_method='p2p_sdr'
)
```

---

## 📈 Output Columns

### Building-Level Columns

- `battery_soc`: State of charge over time
- `battery_charge_total`: Total charging power (kW)
- `battery_discharge_total`: Total discharging power (kW)
- `grid_import_total`: Total grid import (kW)
- `grid_export_total`: Total grid export (kW)

### Per-Unit Columns (20 units)

For each unit (01 to 20):
- `unit_XX_charge`: Charging power allocated to unit (kW)
- `unit_XX_discharge`: Discharging power allocated to unit (kW)
- `unit_XX_net_power`: Net power (PV share - Load) (kW)
- `unit_XX_grid_import`: Grid import for unit (kW)
- `unit_XX_grid_export`: Grid export for unit (kW)

---

## 🔄 Energy Flow Logic

### For Each Time Step:

1. **Calculate Net Power Per Unit**:
   ```
   net_power_i = (PV_total / 20) - load_i
   ```

2. **Calculate Total Building Net Power**:
   ```
   total_net_power = Σ(net_power_i)
   ```

3. **Determine Building-Level Battery Operation**:
   - If `total_net_power > 0`: Charge battery
   - If `total_net_power < 0`: Discharge battery

4. **Allocate Battery Power to Units**:
   - Based on allocation method (energy share, floor area, P2P SDR)
   - Units with more excess → more charging allocation
   - Units with more deficit → more discharging allocation

5. **Update SOC**:
   - Uses SOC equation from paper
   - Single building-level SOC tracking

---

## 📊 Allocation Summary

Get allocation statistics:

```python
summary = shared_battery.get_allocation_summary()
# Returns:
# {
#     'method': 'energy_share',
#     'unit_total_charge': [array of 20 values],
#     'unit_total_discharge': [array of 20 values],
#     'unit_avg_charge': [array of 20 values],
#     'unit_avg_discharge': [array of 20 values],
#     'unit_weights': [array of 20 values]
# }
```

---

## 📚 References

### Paper Citation

**Liao et al. (2024)**: P2P intra-community trading model
- Similar allocation concept
- P2P energy trading among community members
- Shared resources allocation

### Battery Parameters

**Table A2**: Specifications of the battery used in this study (residential building)

---

## ✅ Key Features

- ✅ Building-level shared battery
- ✅ Multiple allocation methods (energy share, floor area, P2P SDR, equal)
- ✅ Dynamic weight updates (optional)
- ✅ Per-unit battery power tracking
- ✅ Integration with master dataset
- ✅ Grid import/export per unit
- ✅ SOC equation implementation
- ✅ Allocation statistics

---

## 🎯 Torino Case Setup

### Configuration

```python
# Torino building: 20 units, shared battery
torino_battery = SharedBatteryModel(
    capacity_kwh=20.0,  # Building-level capacity
    num_units=20,
    initial_soc=0.5,
    allocation_method='energy_share'  # Or 'floor_area', 'p2p_sdr', 'equal'
)

# Simulate with master dataset
results = torino_battery.simulate_for_building(
    df_master=df_master_torino,
    update_weights_dynamically=True  # Update weights each hour
)
```

### Statement for Paper

> "In the Torino case, we assume the PV-battery system is installed at building level and shared between the 20 units; allocation of battery power to each unit follows a metric (e.g. energy share, floor area, or P2P SDR), in analogy with the P2P intra-community trading model proposed by Liao et al. (2024)."

---

## 🚀 Ready to Use

- ✅ Shared battery model implemented
- ✅ All allocation methods working
- ✅ Integration with master dataset
- ✅ Documentation complete
- ✅ Ready for Torino case study

**Everything is ready for your building-level shared battery system!** 🏢🔋

