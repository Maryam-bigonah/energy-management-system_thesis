"""
Battery Parameters Configuration
Based on Table A2 - Residential Building Specifications
"""

# Battery Specifications (Residential Building)
BATTERY_PARAMS = {
    # Capacity
    'capacity_kwh': None,  # Will be set based on building needs
    
    # Efficiency
    'charge_efficiency': 0.9,      # η_ch
    'discharge_efficiency': 0.9,  # η_dis
    
    # Life cycle
    'life_cycles': 10000,          # Number of charge/discharge cycles
    
    # Cost
    'cost_per_kwh': 510,           # xb ($/kWh)
    
    # State of Charge limits
    'soc_min': 0.2,                # SOCmin (20%)
    'soc_max': 0.95,               # SOCmax (95%)
    
    # Power limits
    'charge_power_max_kw': None,   # P_ch,max (kW) - depends on capacity
    'discharge_power_max_kw': None,  # P_dis,max (kW) - depends on capacity
}

# Typical power limits based on capacity (C-rate)
# For residential batteries, typical C-rate is 0.5-1.0
# Meaning: max charge/discharge power = capacity × C-rate
DEFAULT_C_RATE = 0.75  # 0.75 C-rate (conservative)


def get_battery_params(capacity_kwh, c_rate=DEFAULT_C_RATE):
    """
    Get battery parameters for given capacity
    
    Parameters:
    -----------
    capacity_kwh : float
        Battery capacity in kWh
    c_rate : float
        C-rate for power limits (default: 0.75)
        Power limit = capacity × c_rate
    
    Returns:
    --------
    dict: Complete battery parameters
    """
    params = BATTERY_PARAMS.copy()
    params['capacity_kwh'] = capacity_kwh
    params['charge_power_max_kw'] = capacity_kwh * c_rate
    params['discharge_power_max_kw'] = capacity_kwh * c_rate
    
    return params


def print_battery_params(params):
    """
    Print battery parameters in formatted way
    """
    print("=" * 70)
    print("Battery Parameters (Residential Building)")
    print("=" * 70)
    print(f"Capacity: {params['capacity_kwh']:.2f} kWh")
    print(f"Charge Efficiency (η_ch): {params['charge_efficiency']}")
    print(f"Discharge Efficiency (η_dis): {params['discharge_efficiency']}")
    print(f"Life Cycles: {params['life_cycles']}")
    print(f"Cost: ${params['cost_per_kwh']}/kWh")
    print(f"SOC Range: {params['soc_min']*100:.0f}% - {params['soc_max']*100:.0f}%")
    print(f"Max Charge Power: {params['charge_power_max_kw']:.2f} kW")
    print(f"Max Discharge Power: {params['discharge_power_max_kw']:.2f} kW")
    print("=" * 70)


# Example usage
if __name__ == "__main__":
    # Example: 10 kWh battery for residential building
    battery_10kwh = get_battery_params(10.0)
    print_battery_params(battery_10kwh)
