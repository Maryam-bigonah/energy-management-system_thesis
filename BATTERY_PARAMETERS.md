# Battery Parameters Documentation

## Battery Specifications (Table A2 - Residential Building)

Based on research paper specifications for residential building applications.

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Battery Type | - | Residential | Application type |
| Battery Capacity | E_b | To be specified | kWh |
| Charge Efficiency | η_ch | 0.9 | 90% efficiency |
| Discharge Efficiency | η_dis | 0.9 | 90% efficiency |
| Life Cycle Number | - | 10,000 | Battery life cycles |
| Battery Cost | xb | $510/kWh | Cost per kWh capacity |
| SOC Minimum | SOC_min | 0.2 | 20% minimum state of charge |
| SOC Maximum | SOC_max | 0.95 | 95% maximum state of charge |
| Maximum Charge Power | P_ch,max | To be specified | kW |
| Maximum Discharge Power | P_dis,max | To be specified | kW |

## SOC Update Equation

The State of Charge (SOC) is updated using the following equation for 1-hour time steps:

```
SOC(t+1) = SOC(t) + ξ(t) * P_b,ch(t) * Δt * η_ch / E_b 
           - (1 - ξ(t)) * P_b,dis(t) * Δt / (E_b * η_dis)
```

### Variables:
- **SOC(t)**: State of Charge at time t (0-1 or kWh)
- **SOC(t+1)**: State of Charge at time t+1
- **ξ(t)**: Charging indicator (1 if charging, 0 if discharging)
- **P_b,ch(t)**: Battery charge power at time t (kW, positive when charging)
- **P_b,dis(t)**: Battery discharge power at time t (kW, positive when discharging)
- **E_b**: Battery capacity (kWh)
- **Δt**: Time step (1 hour for hourly data)
- **η_ch**: Charge efficiency = 0.9
- **η_dis**: Discharge efficiency = 0.9

### Equation Breakdown:

**Charging term** (when ξ(t) = 1):
```
SOC_increase = P_b,ch(t) * Δt * η_ch / E_b
```

**Discharging term** (when ξ(t) = 0):
```
SOC_decrease = P_b,dis(t) * Δt / (E_b * η_dis)
```

## Implementation

The battery parameters and SOC update equation are implemented in:
- **File**: `battery_parameters.py`
- **Functions**:
  - `get_battery_specs()` - Get battery specifications
  - `calculate_soc_update()` - Calculate SOC update for one time step
  - `simulate_battery_soc()` - Simulate battery SOC over time series
  - `apply_soc_bounds()` - Apply SOC constraints (SOCmin, SOCmax)
  - `apply_power_bounds()` - Apply power constraints (Pch,max, Pdis,max)

## Usage Example

```python
from battery_parameters import get_battery_specs, calculate_soc_update, simulate_battery_soc

# Get battery specs
specs = get_battery_specs(
    capacity_kwh=10.0,
    pch_max_kw=5.0,
    pdis_max_kw=5.0
)

# Calculate SOC update for one hour
soc_next, xi = calculate_soc_update(
    soc_t=0.5,  # Current SOC (50%)
    pb_ch_t=2.0,  # Charge power (kW)
    pb_dis_t=0.0,  # Discharge power (kW)
    battery_capacity_kwh=10.0
)

# Simulate over time series
result = simulate_battery_soc(
    initial_soc=0.5,
    charge_power=[2.0, 3.0, 4.0, ...],
    discharge_power=[0.0, 0.0, 1.0, ...],
    battery_capacity_kwh=10.0
)
```

## Constraints

### SOC Constraints:
- **SOC_min ≤ SOC(t) ≤ SOC_max**
- Default: 0.2 ≤ SOC ≤ 0.95 (20% to 95%)

### Power Constraints:
- **0 ≤ P_b,ch(t) ≤ P_ch,max**
- **0 ≤ P_b,dis(t) ≤ P_dis,max**

## Notes

- **Efficiency**: Both charge and discharge have 0.9 (90%) efficiency
- **Time Step**: Equation assumes Δt = 1 hour for hourly data
- **Charging Indicator**: ξ(t) automatically determined from power direction
- **Bounds**: SOC must be maintained within [SOC_min, SOC_max]

## Citation

When using these parameters, cite:
> "Table A2 – Specifications of the battery used in this study (Residential Building row)"

