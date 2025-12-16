"""
Day-ahead operational cost optimization.

Implements the optimization problem as specified in the thesis:
- Objective: Minimize total operational cost
- Constraints: Demand response, energy balance, battery, P2P trading
- Horizon: 24 hours (hourly resolution)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Tuple, Optional
import warnings


class DayAheadOptimizer:
    """
    Day-ahead operational cost optimization for building energy management.
    
    Solves the optimization problem:
    - Minimize total electricity cost (grid + P2P)
    - Subject to: demand response, energy balance, battery constraints
    - Horizon: 24 hours
    """
    
    def __init__(
        self,
        battery_capacity_kwh: float = 50.0,
        battery_max_power_kw: float = 25.0,
        battery_efficiency_charge: float = 0.95,
        battery_efficiency_discharge: float = 0.95,
        battery_soc_min: float = 0.1,
        battery_soc_max: float = 0.9,
        battery_soc_initial: float = 0.5,
        dr_flexibility: float = 0.10,
        grid_price_import: float = 0.20,  # €/kWh
        grid_price_export: float = 0.10,  # €/kWh
        p2p_price_margin: float = 0.05,  # Margin between grid prices
    ):
        """
        Initialize optimizer with system parameters.
        
        Parameters:
        -----------
        battery_capacity_kwh : float
            Battery capacity in kWh
        battery_max_power_kw : float
            Maximum charging/discharging power in kW
        battery_efficiency_charge : float
            Charging efficiency (0-1)
        battery_efficiency_discharge : float
            Discharging efficiency (0-1)
        battery_soc_min : float
            Minimum state of charge (fraction)
        battery_soc_max : float
            Maximum state of charge (fraction)
        battery_soc_initial : float
            Initial state of charge (fraction)
        dr_flexibility : float
            Demand response flexibility (±fraction, e.g., 0.10 = ±10%)
        grid_price_import : float
            Grid import price (€/kWh)
        grid_price_export : float
            Grid export price (€/kWh)
        p2p_price_margin : float
            P2P price margin between grid import and export
        """
        # Battery parameters
        self.battery_capacity = battery_capacity_kwh
        self.battery_max_power = battery_max_power_kw
        self.eta_ch = battery_efficiency_charge
        self.eta_dis = battery_efficiency_discharge
        self.soc_min = battery_soc_min
        self.soc_max = battery_soc_max
        self.soc_initial = battery_soc_initial
        
        # Demand response
        self.dr_flexibility = dr_flexibility
        
        # Pricing
        self.grid_price_import = grid_price_import
        self.grid_price_export = grid_price_export
        self.p2p_price_margin = p2p_price_margin
        
        # Horizon
        self.horizon = 24  # hours
        # Hourly resolution
        self.dt_hours = 1.0

        # Soft penalties to discourage physically-implausible simultaneous actions
        # (strict exclusivity would require mixed-integer variables).
        self.penalty_throughput = 1e-3  # discourages unnecessary cycling
        self.penalty_overlap = 1e-2     # discourages charge & discharge at same hour
        
    def _calculate_p2p_prices(
        self, 
        pv_forecast: np.ndarray, 
        load_forecast: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate dynamic P2P prices based on supply/demand ratio.
        
        P2P prices are between grid import and export prices,
        adjusted by local supply/demand conditions.
        
        Parameters:
        -----------
        pv_forecast : np.ndarray
            PV generation forecast (kW) for 24 hours
        load_forecast : np.ndarray
            Load forecast (kW) for 24 hours
            
        Returns:
        --------
        p2p_price_in : np.ndarray
            P2P import price (€/kWh) for each hour
        p2p_price_out : np.ndarray
            P2P export price (€/kWh) for each hour
        """
        # Calculate supply/demand ratio
        net_supply = pv_forecast - load_forecast
        supply_demand_ratio = np.clip(
            net_supply / (load_forecast + 1e-6), 
            -1.0, 1.0
        )
        
        # P2P prices: between grid export and import
        # When excess supply: lower export price, higher import price
        # When deficit: higher export price, lower import price
        p2p_price_out = (
            self.grid_price_export 
            + (self.grid_price_import - self.grid_price_export) 
            * (0.5 - 0.5 * supply_demand_ratio)
            - self.p2p_price_margin
        )
        p2p_price_in = (
            self.grid_price_export 
            + (self.grid_price_import - self.grid_price_export) 
            * (0.5 - 0.5 * supply_demand_ratio)
            + self.p2p_price_margin
        )
        
        # Ensure P2P prices are between grid prices
        p2p_price_out = np.clip(
            p2p_price_out,
            self.grid_price_export,
            self.grid_price_import - self.p2p_price_margin
        )
        p2p_price_in = np.clip(
            p2p_price_in,
            self.grid_price_export + self.p2p_price_margin,
            self.grid_price_import
        )
        
        return p2p_price_in, p2p_price_out
    
    def _objective_function(self, x: np.ndarray, *args) -> float:
        """
        Objective function: minimize total operational cost.
        
        Cost = grid_import_cost - grid_export_revenue 
             + p2p_import_cost - p2p_export_revenue
        
        Parameters:
        -----------
        x : np.ndarray
            Decision variables vector
        *args : tuple
            (load_forecast, pv_forecast, grid_price_import, grid_price_export,
             p2p_price_in, p2p_price_out)
            
        Returns:
        --------
        float
            Total cost (to minimize)
        """
        load_forecast, pv_forecast, grid_price_import, grid_price_export, \
            p2p_price_in, p2p_price_out = args
        
        # Unpack decision variables (SoC is computed from P_ch/P_dis, not optimized directly)
        # x = [L_opt (24), P_ch (24), P_dis (24),
        #      P_grid_in (24), P_grid_out (24), P_p2p_in (24), P_p2p_out (24)]
        n = 24
        L_opt = x[0:n]
        P_ch = x[n:2 * n]
        P_dis = x[2 * n:3 * n]
        P_grid_in = x[3 * n:4 * n]
        P_grid_out = x[4 * n:5 * n]
        P_p2p_in = x[5 * n:6 * n]
        P_p2p_out = x[6 * n:7 * n]

        # Calculate cost (€/kWh) * (kW) * dt(hours) -> €
        grid_cost = np.sum(
            (grid_price_import * P_grid_in - grid_price_export * P_grid_out) * self.dt_hours
        )
        p2p_cost = np.sum(
            (p2p_price_in * P_p2p_in - p2p_price_out * P_p2p_out) * self.dt_hours
        )

        # Soft penalties:
        # - throughput discourages unnecessary cycling
        # - overlap discourages simultaneous charge/discharge
        penalty = (
            self.penalty_throughput * np.sum((P_ch + P_dis) * self.dt_hours)
            + self.penalty_overlap * np.sum((P_ch * P_dis) * (self.dt_hours ** 2))
        )

        return grid_cost + p2p_cost + penalty
    
    def _soc_trajectory(self, P_ch: np.ndarray, P_dis: np.ndarray) -> np.ndarray:
        """
        Compute SoC trajectory (kWh) from charging/discharging decisions.

        Returns SoC with length 25: SoC[0]...SoC[24].
        """
        n = self.horizon
        soc = np.zeros(n + 1, dtype=float)
        soc[0] = self.soc_initial * self.battery_capacity
        for t in range(n):
            soc[t + 1] = soc[t] + (
                self.eta_ch * P_ch[t] - (1.0 / self.eta_dis) * P_dis[t]
            ) * self.dt_hours
        return soc

    def _constraints(self, load_forecast: np.ndarray, pv_forecast: np.ndarray) -> list:
        """
        Build constraint functions for optimization.
        
        Constraints:
        1. Hourly demand response flexibility
        2. Daily energy conservation
        3. Energy balance (each hour)
        4. Battery SoC dynamics
        5. Battery SoC bounds
        6. Battery power limits
        7. Battery mutual exclusivity (simplified)
        8. Non-negativity
        
        Parameters:
        -----------
        load_forecast : np.ndarray
            Baseline load forecast (kW) for 24 hours
            
        Returns:
        --------
        list
            List of constraint dictionaries for scipy.optimize
        """
        n = 24
        constraints = []
        
        # 1. Hourly demand response flexibility: (1-α)L_base ≤ L_opt ≤ (1+α)L_base
        for t in range(n):
            # Lower bound
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, t=t: x[t] - (1 - self.dr_flexibility) * load_forecast[t]
            })
            # Upper bound
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, t=t: (1 + self.dr_flexibility) * load_forecast[t] - x[t]
            })
        
        # 2. Daily energy conservation: sum(L_opt) = sum(L_base)
        constraints.append({
            'type': 'eq',
            'fun': lambda x: np.sum(x[0:n]) - np.sum(load_forecast)
        })
        
        # 3. Energy balance (each hour):
        #    L_opt + P_ch + P_p2p_out + P_grid_out = PV + P_dis + P_p2p_in + P_grid_in
        for t in range(n):
            constraints.append({
                'type': 'eq',
                'fun': lambda x, t=t: self._energy_balance_constraint(x, pv_forecast)[t]
            })

        # 4. Battery SoC bounds applied to the *computed* trajectory
        # SoC[0] is fixed by soc_initial; constrain SoC[1]..SoC[24] (and also SoC[0])
        for t in range(n + 1):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, t=t: self._soc_trajectory(
                    x[n:2 * n],
                    x[2 * n:3 * n]
                )[t] - self.soc_min * self.battery_capacity
            })
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, t=t: self.soc_max * self.battery_capacity - self._soc_trajectory(
                    x[n:2 * n],
                    x[2 * n:3 * n]
                )[t]
            })

        # 5. Battery power limits: 0 ≤ P_ch ≤ P_max, 0 ≤ P_dis ≤ P_max
        for t in range(n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, t=t: self.battery_max_power - x[n + t]  # P_ch ≤ P_max
            })
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, t=t: self.battery_max_power - x[2*n + t]  # P_dis ≤ P_max
            })
        
        # 6. Non-negativity for all power variables
        for t in range(n):
            # P_ch, P_dis, P_grid_in, P_grid_out, P_p2p_in, P_p2p_out ≥ 0
            for idx in [n + t, 2 * n + t, 3 * n + t, 4 * n + t, 5 * n + t, 6 * n + t]:
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, idx=idx: x[idx]
                })
        
        return constraints
    
    def _energy_balance_constraint(
        self, 
        x: np.ndarray, 
        pv_forecast: np.ndarray
    ) -> np.ndarray:
        """
        Energy balance constraint for each hour.
        
        L_opt + P_ch + P_p2p_out + P_grid_out = PV + P_dis + P_p2p_in + P_grid_in
        
        Returns array of constraint violations (should be zero).
        """
        n = 24
        L_opt = x[0:n]
        P_ch = x[n:2 * n]
        P_dis = x[2 * n:3 * n]
        P_grid_in = x[3 * n:4 * n]
        P_grid_out = x[4 * n:5 * n]
        P_p2p_in = x[5 * n:6 * n]
        P_p2p_out = x[6 * n:7 * n]
        
        # Energy balance: supply = demand
        balance = (
            pv_forecast + P_dis + P_p2p_in + P_grid_in
            - (L_opt + P_ch + P_p2p_out + P_grid_out)
        )
        
        return balance
    
    def optimize(
        self,
        load_forecast: np.ndarray,
        pv_forecast: np.ndarray,
        initial_guess: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Solve the day-ahead optimization problem.
        
        Parameters:
        -----------
        load_forecast : np.ndarray
            Baseline load forecast (kW) for 24 hours
        pv_forecast : np.ndarray
            PV generation forecast (kW) for 24 hours
        initial_guess : np.ndarray, optional
            Initial guess for optimization (8*24 = 192 variables)
            
        Returns:
        --------
        dict
            Dictionary containing:
            - 'L_opt': Optimized load (kW)
            - 'P_ch': Battery charging power (kW)
            - 'P_dis': Battery discharging power (kW)
            - 'SoC': Battery state of charge (kWh)
            - 'P_grid_in': Grid import (kW)
            - 'P_grid_out': Grid export (kW)
            - 'P_p2p_in': P2P import (kW)
            - 'P_p2p_out': P2P export (kW)
            - 'total_cost': Total operational cost (€)
            - 'success': Optimization success flag
        """
        n = self.horizon
        
        # Calculate P2P prices
        p2p_price_in, p2p_price_out = self._calculate_p2p_prices(
            pv_forecast, load_forecast
        )
        
        # Initial guess
        if initial_guess is None:
            x0 = np.zeros(7 * n)
            # Initialize with baseline load
            x0[0:n] = load_forecast
        else:
            x0 = initial_guess.copy()
        
        # Bounds: all variables non-negative (SoC is computed)
        bounds = [(0, None) for _ in range(7 * n)]

        # Constraints (includes energy balance and SoC bounds)
        constraints = self._constraints(load_forecast, pv_forecast)
        
        # Objective function arguments
        args = (
            load_forecast,
            pv_forecast,
            np.full(n, self.grid_price_import),
            np.full(n, self.grid_price_export),
            p2p_price_in,
            p2p_price_out
        )
        
        # Solve optimization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                self._objective_function,
                x0,
                args=args,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-6}
            )
        
        # Extract results
        L_opt = result.x[0:n]
        P_ch = result.x[n:2 * n]
        P_dis = result.x[2 * n:3 * n]
        P_grid_in = result.x[3 * n:4 * n]
        P_grid_out = result.x[4 * n:5 * n]
        P_p2p_in = result.x[5 * n:6 * n]
        P_p2p_out = result.x[6 * n:7 * n]

        soc_traj = self._soc_trajectory(P_ch, P_dis)  # length 25
        
        # Calculate total cost
        total_cost = result.fun
        
        return {
            'L_opt': L_opt,
            'P_ch': P_ch,
            'P_dis': P_dis,
            'SoC': soc_traj,
            'P_grid_in': P_grid_in,
            'P_grid_out': P_grid_out,
            'P_p2p_in': P_p2p_in,
            'P_p2p_out': P_p2p_out,
            'total_cost': total_cost,
            'success': result.success,
            'message': result.message
        }

