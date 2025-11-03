"""
Battery Parameters and SOC Update Model
Based on Table A2 - Specifications of the battery used in this study (Residential Building)

This module stores battery specifications and implements the SOC update equation
for 1-hour time steps, as used in the research paper.
"""

import pandas as pd
import numpy as np


# ============================================================================
# Battery Specifications (Table A2 - Residential Building)
# ============================================================================

BATTERY_SPECS = {
    'battery_type': 'residential',
    'battery_capacity_kwh': None,  # To be specified per building
    'charge_discharge_efficiency': 0.9,  # η_ch = η_dis = 0.9
    'life_cycle_number': 10000,
    'xb_usd_per_kwh': 510,  # Battery cost in $/kWh
    'soc_min': 0.2,  # SOCmin = 20%
    'soc_max': 0.95,  # SOCmax = 95%
    'pch_max_kw': None,  # Maximum charge power (to be specified)
    'pdis_max_kw': None,  # Maximum discharge power (to be specified)
}


def get_battery_specs(capacity_kwh=None, pch_max_kw=None, pdis_max_kw=None):
    """
    Get battery specifications for residential building
    
    Parameters:
    -----------
    capacity_kwh : float, optional
        Battery capacity in kWh (if None, needs to be specified)
    
    pch_max_kw : float, optional
        Maximum charge power in kW
    
    pdis_max_kw : float, optional
        Maximum discharge power in kW
    
    Returns:
    --------
    dict: Battery specifications dictionary
    """
    specs = BATTERY_SPECS.copy()
    
    if capacity_kwh is not None:
        specs['battery_capacity_kwh'] = capacity_kwh
    if pch_max_kw is not None:
        specs['pch_max_kw'] = pch_max_kw
    if pdis_max_kw is not None:
        specs['pdis_max_kw'] = pdis_max_kw
    
    return specs


def print_battery_specs(specs=None):
    """
    Print battery specifications in a formatted table
    """
    if specs is None:
        specs = BATTERY_SPECS
    
    print("=" * 70)
    print("Battery Specifications (Table A2 - Residential Building)")
    print("=" * 70)
    print()
    print(f"Battery Type: {specs['battery_type']}")
    print(f"Battery Capacity: {specs['battery_capacity_kwh']:.2f} kWh" if specs['battery_capacity_kwh'] else "Battery Capacity: Not specified")
    print(f"Charge/Discharge Efficiency: {specs['charge_discharge_efficiency']}")
    print(f"Life Cycle Number: {specs['life_cycle_number']:,}")
    print(f"Battery Cost (xb): ${specs['xb_usd_per_kwh']} per kWh")
    print(f"SOC Minimum: {specs['soc_min']} ({specs['soc_min']*100:.0f}%)")
    print(f"SOC Maximum: {specs['soc_max']} ({specs['soc_max']*100:.0f}%)")
    if specs['pch_max_kw']:
        print(f"Maximum Charge Power (Pch,max): {specs['pch_max_kw']:.2f} kW")
    if specs['pdis_max_kw']:
        print(f"Maximum Discharge Power (Pdis,max): {specs['pdis_max_kw']:.2f} kW")
    print()


# ============================================================================
# SOC Update Equation
# ============================================================================

def calculate_soc_update(soc_t, pb_ch_t, pb_dis_t, battery_capacity_kwh, 
                        dt_hours=1.0, eta_ch=0.9, eta_dis=0.9, 
                        soc_min=0.2, soc_max=0.95):
    """
    Calculate SOC update for 1-hour time step
    
    SOC Equation from research paper:
    SOC(t+1) = SOC(t) + ξ(t) * P_b,ch(t) * Δt * η_ch / E_b 
               - (1 - ξ(t)) * P_b,dis(t) * Δt / (E_b * η_dis)
    
    Where:
    - ξ(t) = 1 if charging (P_b,ch > 0), 0 if discharging (P_b,dis > 0)
    - η_ch = η_dis = 0.9 (charge/discharge efficiency)
    - E_b = battery capacity in kWh
    - Δt = 1 hour (for hourly data)
    
    Note: SOC is normalized (0-1) in this implementation
    
    Parameters:
    -----------
    soc_t : float or array
        Current SOC at time t (0-1, normalized)
    
    pb_ch_t : float or array
        Battery charge power at time t (kW, positive = charging)
    
    pb_dis_t : float or array
        Battery discharge power at time t (kW, positive = discharging)
    
    battery_capacity_kwh : float
        Battery capacity E_b (kWh)
    
    dt_hours : float
        Time step in hours (default: 1.0 for hourly data)
    
    eta_ch : float
        Charge efficiency (default: 0.9)
    
    eta_dis : float
        Discharge efficiency (default: 0.9)
    
    soc_min : float
        Minimum SOC bound (default: 0.2)
    
    soc_max : float
        Maximum SOC bound (default: 0.95)
    
    Returns:
    --------
    soc_t_plus_1 : float or array
        Updated SOC at time t+1 (0-1, normalized, bounded)
    
    xi_t : float or array
        Charging indicator: 1 if charging, 0 if discharging
    """
    
    # Convert to numpy arrays for vectorized operations
    soc_t = np.array(soc_t) if not isinstance(soc_t, np.ndarray) else soc_t
    pb_ch_t = np.array(pb_ch_t) if not isinstance(pb_ch_t, np.ndarray) else pb_ch_t
    pb_dis_t = np.array(pb_dis_t) if not isinstance(pb_dis_t, np.ndarray) else pb_dis_t
    
    # Determine charging/discharging mode
    # ξ(t) = 1 if charging, 0 if discharging
    # Note: If both are zero, xi_t = 0 (idle)
    xi_t = (pb_ch_t > 0).astype(float)
    
    # Calculate SOC update (in normalized form 0-1)
    # Charging term: ξ(t) * P_b,ch(t) * Δt * η_ch / E_b
    # This gives fraction of capacity added (0-1)
    charging_term = xi_t * pb_ch_t * dt_hours * eta_ch / battery_capacity_kwh
    
    # Discharging term: (1 - ξ(t)) * P_b,dis(t) * Δt / (E_b * η_dis)
    # This gives fraction of capacity removed (0-1)
    discharging_term = (1 - xi_t) * pb_dis_t * dt_hours / (battery_capacity_kwh * eta_dis)
    
    # Update SOC
    soc_t_plus_1 = soc_t + charging_term - discharging_term
    
    # Apply SOC bounds (SOC_min ≤ SOC ≤ SOC_max)
    soc_t_plus_1 = apply_soc_bounds(soc_t_plus_1, soc_min, soc_max)
    
    return soc_t_plus_1, xi_t


def simulate_battery_soc(initial_soc, charge_power, discharge_power, 
                         battery_capacity_kwh, dt_hours=1.0,
                         soc_min=0.2, soc_max=0.95):
    """
    Simulate battery SOC over time
    
    Parameters:
    -----------
    initial_soc : float
        Initial SOC (0-1, normalized)
    
    charge_power : array-like
        Charge power time series (kW)
    
    discharge_power : array-like
        Discharge power time series (kW)
    
    battery_capacity_kwh : float
        Battery capacity (kWh)
    
    dt_hours : float
        Time step in hours (default: 1.0)
    
    soc_min : float
        Minimum SOC bound (default: 0.2)
    
    soc_max : float
        Maximum SOC bound (default: 0.95)
    
    Returns:
    --------
    pd.DataFrame with columns: soc, xi, charge_power, discharge_power, soc_kwh
    """
    
    charge_power = np.array(charge_power)
    discharge_power = np.array(discharge_power)
    
    # Ensure initial SOC is within bounds
    initial_soc = np.clip(initial_soc, soc_min, soc_max)
    
    n = len(charge_power)
    soc = np.zeros(n)
    xi = np.zeros(n)
    
    soc[0] = initial_soc
    
    for t in range(n - 1):
        soc[t+1], xi[t] = calculate_soc_update(
            soc[t], 
            charge_power[t], 
            discharge_power[t],
            battery_capacity_kwh,
            dt_hours,
            soc_min=soc_min,
            soc_max=soc_max
        )
    
    # Last time step charging indicator
    xi[-1] = (charge_power[-1] > 0).astype(float)
    
    # Calculate SOC in kWh for reference
    soc_kwh = soc * battery_capacity_kwh
    
    df = pd.DataFrame({
        'soc': soc,  # Normalized (0-1)
        'soc_kwh': soc_kwh,  # In kWh
        'xi': xi,
        'charge_power': charge_power,
        'discharge_power': discharge_power
    })
    
    return df


# ============================================================================
# Battery Constraints
# ============================================================================

def apply_soc_bounds(soc, soc_min=0.2, soc_max=0.95):
    """
    Apply SOC bounds (SOCmin and SOCmax)
    
    Parameters:
    -----------
    soc : float or array
        SOC values
    
    soc_min : float
        Minimum SOC (default: 0.2)
    
    soc_max : float
        Maximum SOC (default: 0.95)
    
    Returns:
    --------
    soc_bounded : float or array
        SOC values constrained to [soc_min, soc_max]
    """
    return np.clip(soc, soc_min, soc_max)


def apply_power_bounds(charge_power, discharge_power, pch_max, pdis_max):
    """
    Apply power bounds (maximum charge and discharge power)
    
    Parameters:
    -----------
    charge_power : float or array
        Charge power (kW)
    
    discharge_power : float or array
        Discharge power (kW)
    
    pch_max : float
        Maximum charge power (kW)
    
    pdis_max : float
        Maximum discharge power (kW)
    
    Returns:
    --------
    charge_power_bounded, discharge_power_bounded
    """
    charge_power = np.clip(charge_power, 0, pch_max)
    discharge_power = np.clip(discharge_power, 0, pdis_max)
    
    return charge_power, discharge_power


# ============================================================================
# Example Usage
# ============================================================================

def example_battery_simulation():
    """
    Example battery simulation with hourly data
    """
    print("=" * 70)
    print("Battery Simulation Example")
    print("=" * 70)
    
    # Battery specs
    battery_capacity = 10.0  # kWh (example)
    initial_soc = 0.5  # 50%
    
    # Example hourly power data (24 hours)
    # Positive = excess energy available for charging
    # Negative = energy deficit requiring discharge
    net_energy = np.array([
        -2.0, -1.5, -1.0, -0.5, 0.0,  # Night (discharging)
        0.5, 1.0, 1.5, 2.0, 2.5, 3.0,  # Morning (PV generation starts)
        3.5, 4.0, 3.5, 3.0, 2.5,  # Noon (peak PV)
        2.0, 1.5, 1.0, 0.5, 0.0,  # Afternoon (PV decreasing)
        -0.5, -1.0, -1.5  # Evening (load, no PV)
    ])
    
    # Separate into charge and discharge
    charge_power = np.maximum(net_energy, 0)  # Positive = charge
    discharge_power = np.maximum(-net_energy, 0)  # Negative net = discharge
    
    # Simulate
    result = simulate_battery_soc(
        initial_soc=initial_soc,
        charge_power=charge_power,
        discharge_power=discharge_power,
        battery_capacity_kwh=battery_capacity
    )
    
    print("\nBattery SOC Simulation Results (24 hours):")
    print(result)
    print(f"\nInitial SOC: {initial_soc:.2%}")
    print(f"Final SOC: {result['soc'].iloc[-1]:.2%}")
    print(f"Min SOC: {result['soc'].min():.2%}")
    print(f"Max SOC: {result['soc'].max():.2%}")
    
    return result


if __name__ == "__main__":
    # Print battery specs
    print_battery_specs()
    
    # Run example
    example_battery_simulation()

