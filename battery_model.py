"""
Battery Model with SOC Update Equation

Implements the SOC (State of Charge) update equation from the research paper:
SOC(t+1) = SOC(t) + ξ(t) * (P_b,ch(t) * Δt * η_ch / E_b) 
          - (1 - ξ(t)) * (P_b,dis(t) * Δt / (E_b * η_dis))

Where:
- SOC(t): State of charge at time t
- ξ(t): Binary variable (1 if charging, 0 if discharging)
- P_b,ch(t): Charging power at time t (kW)
- P_b,dis(t): Discharging power at time t (kW)
- Δt: Time step (1 hour for hourly data)
- η_ch = η_dis = 0.9: Charge/discharge efficiency
- E_b: Battery capacity (kWh)
"""

import pandas as pd
import numpy as np
from battery_parameters import get_battery_params


class BatteryModel:
    """
    Battery model with SOC tracking and energy management
    """
    
    def __init__(self, capacity_kwh, initial_soc=0.5):
        """
        Initialize battery model
        
        Parameters:
        -----------
        capacity_kwh : float
            Battery capacity in kWh
        initial_soc : float
            Initial state of charge (0.0 to 1.0), default: 0.5 (50%)
        """
        self.params = get_battery_params(capacity_kwh)
        self.capacity_kwh = capacity_kwh
        self.eta_ch = self.params['charge_efficiency']
        self.eta_dis = self.params['discharge_efficiency']
        self.soc_min = self.params['soc_min']
        self.soc_max = self.params['soc_max']
        self.p_ch_max = self.params['charge_power_max_kw']
        self.p_dis_max = self.params['discharge_power_max_kw']
        
        self.initial_soc = np.clip(initial_soc, self.soc_min, self.soc_max)
        self.current_soc = self.initial_soc
        
        self.soc_history = []
        self.charge_history = []
        self.discharge_history = []
        
    def update_soc(self, p_charge, p_discharge, delta_t=1.0):
        """
        Update SOC using the equation from the paper
        
        Parameters:
        -----------
        p_charge : float
            Charging power at time t (kW), 0 if not charging
        p_discharge : float
            Discharging power at time t (kW), 0 if not discharging
        delta_t : float
            Time step in hours (default: 1.0 for hourly data)
        
        Returns:
        --------
        float: Updated SOC (0.0 to 1.0)
        """
        # Ensure only one direction (charging OR discharging)
        if p_charge > 0 and p_discharge > 0:
            raise ValueError("Cannot charge and discharge simultaneously")
        
        # Binary variable: 1 if charging, 0 if discharging
        xi = 1 if p_charge > 0 else 0
        
        # Limit power to maximum
        p_charge = min(p_charge, self.p_ch_max)
        p_discharge = min(p_discharge, self.p_dis_max)
        
        # SOC update equation from the paper
        # SOC(t+1) = SOC(t) + ξ(t) * (P_b,ch(t) * Δt * η_ch / E_b)
        #           - (1 - ξ(t)) * (P_b,dis(t) * Δt / (E_b * η_dis))
        
        if xi == 1:  # Charging
            delta_soc = (p_charge * delta_t * self.eta_ch) / self.capacity_kwh
        else:  # Discharging
            delta_soc = - (p_discharge * delta_t) / (self.capacity_kwh * self.eta_dis)
        
        # Update SOC
        new_soc = self.current_soc + delta_soc
        
        # Enforce SOC limits
        new_soc = np.clip(new_soc, self.soc_min, self.soc_max)
        
        # Store history
        self.current_soc = new_soc
        self.soc_history.append(self.current_soc)
        self.charge_history.append(p_charge)
        self.discharge_history.append(p_discharge)
        
        return self.current_soc
    
    def calculate_optimal_power(self, net_power, delta_t=1.0):
        """
        Calculate optimal charge/discharge power based on net power
        
        Net power = PV generation - Load consumption
        - Positive: Excess power (can charge battery)
        - Negative: Deficit power (need to discharge battery)
        
        Parameters:
        -----------
        net_power : float
            Net power available (PV - Load) in kW
            Positive = excess, Negative = deficit
        delta_t : float
            Time step in hours (default: 1.0)
        
        Returns:
        --------
        tuple: (charge_power, discharge_power, updated_soc)
        """
        charge_power = 0.0
        discharge_power = 0.0
        
        if net_power > 0:  # Excess power - charge battery
            # Can we charge?
            if self.current_soc < self.soc_max:
                # Charge up to max power or until battery is full
                available_capacity = (self.soc_max - self.current_soc) * self.capacity_kwh / (self.eta_ch * delta_t)
                charge_power = min(net_power, self.p_ch_max, available_capacity)
        else:  # Deficit power - discharge battery
            # Can we discharge?
            if self.current_soc > self.soc_min:
                # Discharge up to max power or until battery is empty
                available_energy = (self.current_soc - self.soc_min) * self.capacity_kwh * self.eta_dis / delta_t
                discharge_power = min(abs(net_power), self.p_dis_max, available_energy)
        
        # Update SOC
        updated_soc = self.update_soc(charge_power, discharge_power, delta_t)
        
        return charge_power, discharge_power, updated_soc
    
    def simulate(self, net_power_series, initial_soc=None, delta_t=1.0):
        """
        Simulate battery operation over a time series
        
        Parameters:
        -----------
        net_power_series : pd.Series or array-like
            Series of net power (PV - Load) for each time step
        initial_soc : float, optional
            Initial SOC (uses current SOC if None)
        delta_t : float
            Time step in hours (default: 1.0)
        
        Returns:
        --------
        pd.DataFrame with columns:
            - soc: State of charge over time
            - charge_power: Charging power
            - discharge_power: Discharging power
            - net_power: Input net power
        """
        if initial_soc is not None:
            self.current_soc = np.clip(initial_soc, self.soc_min, self.soc_max)
        
        self.soc_history = [self.current_soc]
        self.charge_history = []
        self.discharge_history = []
        
        charge_powers = []
        discharge_powers = []
        
        for net_power in net_power_series:
            charge_p, discharge_p, _ = self.calculate_optimal_power(net_power, delta_t)
            charge_powers.append(charge_p)
            discharge_powers.append(discharge_p)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'soc': self.soc_history,
            'charge_power': charge_powers,
            'discharge_power': discharge_powers,
            'net_power': net_power_series.values if isinstance(net_power_series, pd.Series) else net_power_series
        })
        
        if isinstance(net_power_series, pd.Series):
            results.index = net_power_series.index
        
        return results
    
    def reset(self):
        """Reset battery to initial SOC"""
        self.current_soc = self.initial_soc
        self.soc_history = []
        self.charge_history = []
        self.discharge_history = []
    
    def get_summary(self):
        """Get battery summary statistics"""
        if len(self.soc_history) == 0:
            return None
        
        return {
            'initial_soc': self.initial_soc,
            'final_soc': self.current_soc,
            'min_soc': min(self.soc_history),
            'max_soc': max(self.soc_history),
            'avg_soc': np.mean(self.soc_history),
            'total_charge_energy': sum(self.charge_history),
            'total_discharge_energy': sum(self.discharge_history),
            'energy_efficiency': sum(self.discharge_history) / sum(self.charge_history) if sum(self.charge_history) > 0 else 0
        }


def simulate_battery_for_building(df_master, battery_capacity_kwh=10.0, initial_soc=0.5):
    """
    Simulate battery operation for building master dataset
    
    Parameters:
    -----------
    df_master : pd.DataFrame
        Master dataset with columns: pv_1kw, apartment_01 to apartment_20
    battery_capacity_kwh : float
        Battery capacity in kWh
    initial_soc : float
        Initial SOC (default: 0.5)
    
    Returns:
    --------
    pd.DataFrame with battery operation results
    """
    # Calculate total building load
    apt_cols = [col for col in df_master.columns if col.startswith('apartment_')]
    total_load = df_master[apt_cols].sum(axis=1)
    
    # Calculate net power (PV - Load)
    net_power = df_master['pv_1kw'] - total_load
    
    # Initialize battery
    battery = BatteryModel(battery_capacity_kwh, initial_soc)
    
    # Simulate
    results = battery.simulate(net_power, delta_t=1.0)
    
    # Add to master dataset
    df_with_battery = df_master.copy()
    df_with_battery['battery_soc'] = results['soc'].values
    df_with_battery['battery_charge'] = results['charge_power'].values
    df_with_battery['battery_discharge'] = results['discharge_power'].values
    df_with_battery['net_power'] = net_power.values
    df_with_battery['grid_import'] = net_power - results['charge_power'] + results['discharge_power']
    df_with_battery['grid_import'] = df_with_battery['grid_import'].clip(lower=0)  # Only import (positive)
    df_with_battery['grid_export'] = (net_power - results['charge_power']).clip(lower=0)  # Only export (positive)
    
    return df_with_battery, battery


def main():
    """
    Example usage
    """
    print("=" * 70)
    print("Battery Model - SOC Update Equation")
    print("=" * 70)
    
    # Example: 10 kWh battery
    battery = BatteryModel(capacity_kwh=10.0, initial_soc=0.5)
    
    print("\nBattery Parameters:")
    print_battery_params(battery.params)
    
    print("\nExample SOC Updates:")
    print("Time | Charge (kW) | Discharge (kW) | SOC")
    print("-" * 50)
    
    # Simulate a few hours
    scenarios = [
        (2.0, 0.0, "Excess PV - Charging"),
        (1.5, 0.0, "Excess PV - Charging"),
        (0.0, 3.0, "Load > PV - Discharging"),
        (0.0, 2.0, "Load > PV - Discharging"),
    ]
    
    for i, (p_ch, p_dis, desc) in enumerate(scenarios, 1):
        old_soc = battery.current_soc
        new_soc = battery.update_soc(p_ch, p_dis, delta_t=1.0)
        print(f"  {i}  |     {p_ch:4.1f}    |      {p_dis:4.1f}     | {new_soc:.3f} ({desc})")
    
    print("\n" + "=" * 70)
    print("Battery simulation complete!")
    print("=" * 70)


if __name__ == "__main__":
    from battery_parameters import print_battery_params
    main()

