#!/usr/bin/env python3
"""
24-Hour Energy Optimization Model
Linear Programming model for building energy management with multiple strategies

This model optimizes energy costs for a 20-unit apartment building with:
- Real PV generation data (PVGIS)
- Real load data (European studies)
- Real TOU pricing (ARERA)
- Research-based battery specifications

Four optimization strategies:
1. MSC (Market Self-Consumption)
2. TOU (Time-of-Use)
3. MMR-P2P (Market-Making Retail P2P)
4. DR-P2P (Demand Response P2P)
"""

import pandas as pd
import numpy as np
import yaml
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json
from strategy_adapter import StrategyAdapter, StrategyConfig, StrategyType

# Optimization libraries
try:
    from scipy.optimize import linprog
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ scipy not available, using simplified optimization")

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    print("⚠️ cvxpy not available, using scipy optimization")

class OptimizationStrategy(Enum):
    """Optimization strategy types"""
    MSC = "Market Self-Consumption"
    TOU = "Time-of-Use"
    MMR_P2P = "Market-Making Retail P2P"
    DR_P2P = "Demand Response P2P"

@dataclass
class BatterySpecs:
    """Battery specifications from research paper"""
    capacity_kwh: float
    max_charge_kw: float
    max_discharge_kw: float
    charge_efficiency: float
    discharge_efficiency: float
    soc_min: float
    soc_max: float
    initial_soc: float

@dataclass
class OptimizationResult:
    """Results from optimization model"""
    strategy: OptimizationStrategy
    total_cost_eur: float
    hourly_results: List[Dict]
    battery_soc_trajectory: List[float]
    grid_import_kwh: List[float]
    grid_export_kwh: List[float]
    battery_charge_kwh: List[float]
    battery_discharge_kwh: List[float]
    optimization_status: str
    pv_curtailment_kwh: List[float] = None
    p2p_buy_kwh: List[float] = None
    p2p_sell_kwh: List[float] = None
    dr_adjusted_load_kwh: List[float] = None

class StrategyAdapter:
    """Strategy adapter for different optimization approaches"""
    
    def __init__(self):
        self.strategies = {
            OptimizationStrategy.MSC: self._msc_strategy,
            OptimizationStrategy.TOU: self._tou_strategy,
            OptimizationStrategy.MMR_P2P: self._mmr_p2p_strategy,
            OptimizationStrategy.DR_P2P: self._dr_p2p_strategy
        }
    
    def get_strategy_params(self, strategy: OptimizationStrategy) -> Dict:
        """Get strategy-specific parameters"""
        return self.strategies[strategy]()
    
    def _msc_strategy(self) -> Dict:
        """Market Self-Consumption strategy parameters"""
        return {
            "name": "Market Self-Consumption",
            "description": "Maximize self-consumption, minimize grid dependency",
            "grid_import_penalty": 1.0,  # Standard grid import cost
            "grid_export_reward": 0.1,   # Feed-in tariff
            "battery_priority": 0.8,     # High battery usage priority
            "self_consumption_bonus": 0.05,  # Bonus for self-consumption
            "peak_shaving": False,
            "arbitrage": False,
            "p2p_trading": False
        }
    
    def _tou_strategy(self) -> Dict:
        """Time-of-Use strategy parameters"""
        return {
            "name": "Time-of-Use Optimization",
            "description": "Optimize based on TOU pricing (F1/F2/F3)",
            "grid_import_penalty": 1.0,  # TOU-based pricing
            "grid_export_reward": 0.1,   # Feed-in tariff
            "battery_priority": 0.9,     # Very high battery usage
            "peak_shaving": True,        # Shave peaks during F1
            "arbitrage": True,          # Charge during F3, discharge during F1
            "p2p_trading": False,
            "tou_multipliers": {
                "F1_peak": 1.0,    # Full TOU price
                "F2_flat": 0.7,    # 70% of peak price
                "F3_valley": 0.5   # 50% of peak price
            }
        }
    
    def _mmr_p2p_strategy(self) -> Dict:
        """Market-Making Retail P2P strategy parameters"""
        return {
            "name": "Market-Making Retail P2P",
            "description": "Act as market maker in P2P energy trading",
            "grid_import_penalty": 1.0,
            "grid_export_reward": 0.15,  # Higher P2P export price
            "battery_priority": 0.7,
            "peak_shaving": True,
            "arbitrage": True,
            "p2p_trading": True,
            "p2p_buy_price": 0.25,      # Buy from P2P at 25c/kWh
            "p2p_sell_price": 0.35,     # Sell to P2P at 35c/kWh
            "market_making_spread": 0.10,  # 10c spread
            "liquidity_bonus": 0.02     # Bonus for providing liquidity
        }
    
    def _dr_p2p_strategy(self) -> Dict:
        """Demand Response P2P strategy parameters"""
        return {
            "name": "Demand Response P2P",
            "description": "Participate in demand response programs via P2P",
            "grid_import_penalty": 1.0,
            "grid_export_reward": 0.20,  # Higher DR export price
            "battery_priority": 0.95,    # Very high battery usage
            "peak_shaving": True,
            "arbitrage": True,
            "p2p_trading": True,
            "dr_participation": True,
            "dr_incentive": 0.05,        # 5c/kWh DR incentive
            "flexibility_bonus": 0.03,   # Bonus for flexibility
            "response_time": 0.25,       # 15-minute response time
            "min_duration": 2            # Minimum 2-hour DR events
        }

class EnergyOptimizationModel:
    """24-hour energy optimization model using linear programming"""
    
    def __init__(self, data_dir: str = "project/data"):
        self.data_dir = data_dir
        self.strategy_adapter = StrategyAdapter()
        self.load_data()
        self.validate_data()
    
    def load_data(self):
        """Load all required data from Step 1"""
        print("📊 Loading real data from Step 1...")
        
        # Load PV data (real PVGIS)
        pv_file = os.path.join(self.data_dir, "pv_24h.csv")
        if os.path.exists(pv_file):
            self.pv_data = pd.read_csv(pv_file)
            print(f"✅ PV data loaded: {len(self.pv_data)} hours")
        else:
            raise FileNotFoundError(f"PV data not found: {pv_file}")
        
        # Load load data (real European studies)
        load_file = os.path.join(self.data_dir, "load_24h.csv")
        if os.path.exists(load_file):
            self.load_data_df = pd.read_csv(load_file)
            print(f"✅ Load data loaded: {len(self.load_data_df)} hours")
        else:
            raise FileNotFoundError(f"Load data not found: {load_file}")
        
        # Load TOU data (real ARERA)
        tou_file = os.path.join(self.data_dir, "tou_24h.csv")
        if os.path.exists(tou_file):
            self.tou_data = pd.read_csv(tou_file)
            print(f"✅ TOU data loaded: {len(self.tou_data)} hours")
        else:
            raise FileNotFoundError(f"TOU data not found: {tou_file}")
        
        # Load battery specifications (research-based)
        battery_file = os.path.join(self.data_dir, "battery.yaml")
        if os.path.exists(battery_file):
            with open(battery_file, 'r') as f:
                battery_specs = yaml.safe_load(f)
            
            self.battery = BatterySpecs(
                capacity_kwh=battery_specs['Ebat_kWh'],
                max_charge_kw=battery_specs['Pch_max_kW'],
                max_discharge_kw=battery_specs['Pdis_max_kW'],
                charge_efficiency=battery_specs['eta_ch'],
                discharge_efficiency=battery_specs['eta_dis'],
                soc_min=battery_specs['SOCmin'],
                soc_max=battery_specs['SOCmax'],
                initial_soc=battery_specs['SOC0_frac']
            )
            print("✅ Battery specifications loaded")
        else:
            raise FileNotFoundError(f"Battery specs not found: {battery_file}")
    
    def validate_data(self):
        """Validate that all data is properly loaded and consistent"""
        print("🔍 Validating data consistency...")
        
        # Check data lengths
        if len(self.pv_data) != 24 or len(self.load_data_df) != 24 or len(self.tou_data) != 24:
            raise ValueError("All data files must have exactly 24 hours")
        
        # Check required columns
        required_pv_cols = ['hour', 'pv_generation_kw']
        required_load_cols = ['hour', 'load_kw']
        required_tou_cols = ['hour', 'price_buy', 'price_sell']
        
        for col in required_pv_cols:
            if col not in self.pv_data.columns:
                raise ValueError(f"Missing column in PV data: {col}")
        
        for col in required_load_cols:
            if col not in self.load_data_df.columns:
                raise ValueError(f"Missing column in load data: {col}")
        
        for col in required_tou_cols:
            if col not in self.tou_data.columns:
                raise ValueError(f"Missing column in TOU data: {col}")
        
        # Check for negative values
        if (self.pv_data['pv_generation_kw'] < 0).any():
            raise ValueError("PV generation cannot be negative")
        
        if (self.load_data_df['load_kw'] < 0).any():
            raise ValueError("Load cannot be negative")
        
        if (self.tou_data['price_buy'] < 0).any() or (self.tou_data['price_sell'] < 0).any():
            raise ValueError("TOU prices cannot be negative")
        
        print("✅ Data validation passed")
    
    def calculate_net_load(self) -> np.ndarray:
        """Calculate net load (load - PV) for each hour"""
        load_kw = self.load_data_df['load_kw'].values
        pv_kw = self.pv_data['pv_generation_kw'].values
        net_load = load_kw - pv_kw
        return net_load
    
    def optimize_strategy(self, strategy: OptimizationStrategy, strategy_config: Optional[StrategyConfig] = None) -> OptimizationResult:
        """Optimize energy management for a specific strategy"""
        print(f"🔧 Optimizing strategy: {strategy.value}")
        
        # Get strategy configuration
        if strategy_config is None:
            # Create default config based on strategy type
            strategy_type_map = {
                OptimizationStrategy.MSC: StrategyType.MSC,
                OptimizationStrategy.TOU: StrategyType.TOU,
                OptimizationStrategy.MMR_P2P: StrategyType.MMR_P2P,
                OptimizationStrategy.DR_P2P: StrategyType.DR_P2P
            }
            strategy_type = strategy_type_map[strategy]
            strategy_config = self.strategy_adapter.get_strategy_config(strategy_type)
        
        print(f"📋 Using config: {strategy_config.strategy_name}")
        print(f"   Components: P2P={strategy_config.include_p2p_trading}, DR={strategy_config.include_dr_adjustment}")
        
        # Calculate net load
        net_load = self.calculate_net_load()
        
        # Get TOU prices
        buy_prices = self.tou_data['price_buy'].values
        sell_prices = self.tou_data['price_sell'].values
        
        # Apply strategy-specific modifications
        if strategy == OptimizationStrategy.TOU:
            # TOU strategy uses standard prices (no multipliers needed)
            pass  # Use original buy_prices and sell_prices
        elif strategy in [OptimizationStrategy.MMR_P2P, OptimizationStrategy.DR_P2P]:
            buy_prices, sell_prices = self._apply_p2p_pricing(buy_prices, sell_prices, strategy_config.to_dict())
        
        # Solve optimization problem
        if CVXPY_AVAILABLE:
            result = self._solve_cvxpy_optimization(net_load, buy_prices, sell_prices, strategy_config, strategy)
        elif SCIPY_AVAILABLE:
            result = self._solve_scipy_optimization(net_load, buy_prices, sell_prices, strategy_config.to_dict(), strategy)
        else:
            result = self._solve_simplified_optimization(net_load, buy_prices, sell_prices, strategy_config.to_dict(), strategy)
        
        return result
    
    def _apply_tou_multipliers(self, buy_prices: np.ndarray, strategy_params: Dict) -> np.ndarray:
        """Apply TOU multipliers for F1/F2/F3 bands"""
        multipliers = strategy_params['tou_multipliers']
        
        # Define TOU bands based on Italian ARERA structure
        # F1 (Peak): 8-19, F2 (Flat): 7-8 and 19-23, F3 (Valley): 23-7
        modified_prices = buy_prices.copy()
        
        for hour in range(24):
            if 8 <= hour <= 19:  # F1 Peak
                modified_prices[hour] *= multipliers['F1_peak']
            elif hour in [7, 19, 20, 21, 22]:  # F2 Flat
                modified_prices[hour] *= multipliers['F2_flat']
            else:  # F3 Valley
                modified_prices[hour] *= multipliers['F3_valley']
        
        return modified_prices
    
    def _apply_p2p_pricing(self, buy_prices: np.ndarray, sell_prices: np.ndarray, strategy_params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Apply P2P trading pricing"""
        p2p_buy_price = strategy_params.get('p2p_buy_price', 0.25)
        p2p_sell_price = strategy_params.get('p2p_sell_price', 0.35)
        
        # For P2P strategies, use P2P prices when beneficial
        modified_buy = np.minimum(buy_prices, p2p_buy_price)
        modified_sell = np.maximum(sell_prices, p2p_sell_price)
        
        return modified_buy, modified_sell
    
    def _solve_cvxpy_optimization(self, net_load: np.ndarray, buy_prices: np.ndarray, 
                                 sell_prices: np.ndarray, strategy_config: StrategyConfig, strategy: OptimizationStrategy) -> OptimizationResult:
        """Solve optimization using CVXPY (most accurate)"""
        print("🔧 Using CVXPY optimization solver...")
        
        # Decision Variables (per hour t) - All nonnegative unless noted
        # General Decision Variables
        G_t_in = cp.Variable(24, nonneg=True)      # G_t^in (kW): grid import
        G_t_out = cp.Variable(24, nonneg=True)     # G_t^out (kW): grid export
        P_t_ch = cp.Variable(24, nonneg=True)      # P_t^ch (kW): battery charge power
        P_t_dis = cp.Variable(24, nonneg=True)     # P_t^dis (kW): battery discharge power
        SOC_t = cp.Variable(25, nonneg=True)       # SOC_t (kWh): battery energy state (bounded)
        S_t_curt = cp.Variable(24, nonneg=True)    # S_t^curt (kW): PV curtailment (to keep model feasible)
        
        # Variables specific to P2P strategies
        P_t_p2p_buy = None   # P_t^{p2p,buy} (kW): peer buy
        P_t_p2p_sell = None  # P_t^{p2p,sell} (kW): peer sell
        
        # Variables specific to DR-P2P
        L_t_tilde = None     # L̃_t (kW): DR-adjusted load
        
        # Initialize P2P variables if strategy requires them
        if strategy_config.include_p2p_trading:
            P_t_p2p_buy = cp.Variable(24, nonneg=True)   # P2P buy power
            P_t_p2p_sell = cp.Variable(24, nonneg=True)  # P2P sell power
        
        # Initialize DR variables if strategy requires them
        if strategy_config.include_dr_adjustment:
            L_t_tilde = cp.Variable(24, nonneg=True)     # DR-adjusted load
        
        # Constraints
        constraints = []
        
        # Battery SOC constraints
        constraints.append(SOC_t[0] == self.battery.initial_soc * self.battery.capacity_kwh)
        constraints.append(SOC_t[24] == self.battery.initial_soc * self.battery.capacity_kwh)  # End of day
        
        for t in range(24):
            # SOC evolution
            constraints.append(
                SOC_t[t+1] == SOC_t[t] + 
                P_t_ch[t] * self.battery.charge_efficiency - 
                P_t_dis[t] / self.battery.discharge_efficiency
            )
            
            # SOC bounds
            constraints.append(SOC_t[t+1] >= self.battery.soc_min * self.battery.capacity_kwh)
            constraints.append(SOC_t[t+1] <= self.battery.soc_max * self.battery.capacity_kwh)
            
            # Power limits
            constraints.append(P_t_ch[t] <= self.battery.max_charge_kw)
            constraints.append(P_t_dis[t] <= self.battery.max_discharge_kw)
            
            # PV curtailment constraint (to keep model feasible)
            constraints.append(S_t_curt[t] <= self.pv_data['pv_generation_kw'].iloc[t])
            constraints.append(S_t_curt[t] >= 0)  # Non-negativity
            
            # Battery operational constraints
            # Note: Simultaneous charge/discharge prevention is handled by numerical penalty in objective
            # Linear constraint: SOC must be sufficient for discharge
            constraints.append(P_t_dis[t] <= (SOC_t[t] - self.battery.soc_min * self.battery.capacity_kwh) * self.battery.discharge_efficiency)
            
            # Linear constraint: SOC must have space for charge
            constraints.append(P_t_ch[t] <= (self.battery.soc_max * self.battery.capacity_kwh - SOC_t[t]) / self.battery.charge_efficiency)
            
            # Energy balance - depends on strategy
            if strategy_config.include_dr_adjustment and L_t_tilde is not None:
                # DR strategies: Use DR-adjusted load
                effective_load = L_t_tilde[t]
                # DR load adjustment constraints
                constraints.append(L_t_tilde[t] >= 0)  # Non-negative DR-adjusted load
                constraints.append(L_t_tilde[t] <= net_load[t] * strategy_config.dr_max_increase)  # Max increase allowed
            else:
                # Other strategies: Use original net load
                effective_load = net_load[t]
            
            # Universal energy balance constraint
            # PV Generation - Load - Curtailment = Grid Import - Grid Export + Battery Discharge - Battery Charge + P2P Buy - P2P Sell
            pv_gen = self.pv_data['pv_generation_kw'].iloc[t]
            
            if strategy_config.include_p2p_trading and P_t_p2p_buy is not None and P_t_p2p_sell is not None:
                # P2P strategies: Include P2P trading
                constraints.append(
                    pv_gen - effective_load - S_t_curt[t] == 
                    G_t_in[t] - G_t_out[t] + P_t_dis[t] - P_t_ch[t] + P_t_p2p_buy[t] - P_t_p2p_sell[t]
                )
                
                # P2P trading constraints
                constraints.append(P_t_p2p_buy[t] >= 0)  # Non-negative P2P buy
                constraints.append(P_t_p2p_sell[t] >= 0)  # Non-negative P2P sell
                
                # P2P trading limits (reasonable bounds)
                constraints.append(P_t_p2p_buy[t] <= 100)  # Max 100 kW P2P buy
                constraints.append(P_t_p2p_sell[t] <= 100)  # Max 100 kW P2P sell
                
                # P2P market constraints (can't buy and sell simultaneously in same hour)
                # This is handled by the numerical penalty in objective function
                # Linear approximation: limit total P2P activity
                constraints.append(P_t_p2p_buy[t] + P_t_p2p_sell[t] <= 100)  # Max total P2P activity
                
            else:
                # Non-P2P strategies: Standard grid + battery
                constraints.append(
                    pv_gen - effective_load - S_t_curt[t] == 
                    G_t_in[t] - G_t_out[t] + P_t_dis[t] - P_t_ch[t]
                )
            
            # MSC-specific constraints
            if strategy_config.strategy_type == StrategyType.MSC and strategy_config.msc_forbid_export:
                # MSC with export forbidden: force grid export to zero
                constraints.append(G_t_out[t] == 0)
            
            # Grid connection constraints
            constraints.append(G_t_in[t] >= 0)  # Non-negative grid import
            constraints.append(G_t_out[t] >= 0)  # Non-negative grid export
            
            # Grid power limits (reasonable bounds)
            constraints.append(G_t_in[t] <= 200)  # Max 200 kW grid import
            constraints.append(G_t_out[t] <= 200)  # Max 200 kW grid export
            
            # Additional operational constraints
            
            # Minimum operating constraints (avoid tiny values)
            # These are handled by the numerical penalty in objective function
            # Linear approximation: ensure reasonable power levels when active
            min_power = 0.1  # 100 W minimum when active
            # Note: Actual minimum constraints would require binary variables
            
            # Ramp rate constraints (battery power change limits)
            if t > 0:
                max_ramp = self.battery.max_charge_kw * 0.5  # 50% of max power per hour
                constraints.append(P_t_ch[t] - P_t_ch[t-1] <= max_ramp)  # Ramp up limit
                constraints.append(P_t_ch[t-1] - P_t_ch[t] <= max_ramp)  # Ramp down limit
                constraints.append(P_t_dis[t] - P_t_dis[t-1] <= max_ramp)  # Ramp up limit
                constraints.append(P_t_dis[t-1] - P_t_dis[t] <= max_ramp)  # Ramp down limit
        
        # Objective: minimize daily cost
        # Base cost: min C = Σ_{t=1}^{24} (p_t^buy * G_t^in - p_t^sell * G_t^out) * Δt
        # Since Δt = 1 hour, power in kW equals energy in kWh per step
        total_cost = cp.sum(
            G_t_in * buy_prices - 
            G_t_out * sell_prices
        )
        
        # Add strategy terms when enabled
        if strategy_config.include_p2p_trading and P_t_p2p_buy is not None and P_t_p2p_sell is not None:
            if strategy_config.strategy_type == StrategyType.MMR_P2P:
                # MMR-P2P: + Σ_t p_t^p2p * P_t^p2p,buy * Δt - Σ_t p_t^p2p * P_t^p2p,sell * Δt
                p2p_price = strategy_config.p2p_single_price or 0.30  # Default price
                total_cost += cp.sum(P_t_p2p_buy * p2p_price - P_t_p2p_sell * p2p_price)
            
            elif strategy_config.strategy_type == StrategyType.DR_P2P:
                # DR-P2P (SDR-based): replace p_t^p2p with two prices p_t^p2p,buy, p_t^p2p,sell computed from SDR
                p2p_buy_price = strategy_config.p2p_buy_price or 0.25  # Default buy price
                p2p_sell_price = strategy_config.p2p_sell_price or 0.35  # Default sell price
                total_cost += cp.sum(P_t_p2p_buy * p2p_buy_price - P_t_p2p_sell * p2p_sell_price)
        
        # Add strategy-specific bonus terms
        if strategy_config.self_consumption_bonus and strategy_config.self_consumption_bonus > 0:
            self_consumption = cp.sum(cp.minimum(net_load, P_t_dis))
            total_cost -= strategy_config.self_consumption_bonus * self_consumption
        
        if strategy_config.liquidity_bonus and strategy_config.liquidity_bonus > 0:
            liquidity = cp.sum(G_t_out + G_t_in)
            if strategy_config.include_p2p_trading and P_t_p2p_buy is not None and P_t_p2p_sell is not None:
                liquidity += cp.sum(P_t_p2p_buy + P_t_p2p_sell)
            total_cost -= strategy_config.liquidity_bonus * liquidity
        
        # Add DR incentive if applicable
        if strategy_config.include_dr_adjustment and L_t_tilde is not None and strategy_config.dr_incentive:
            dr_incentive = strategy_config.dr_incentive
            # DR incentive for load reduction
            load_reduction = cp.sum(net_load - L_t_tilde)
            total_cost -= dr_incentive * load_reduction
        
        # Add export penalty for MSC if configured
        if strategy_config.export_penalty and strategy_config.export_penalty > 0:
            total_cost += strategy_config.export_penalty * cp.sum(G_t_out)
        
        # Tip (numerical): Add tiny penalty ε Σ_t (P_t^ch + P_t^dis) with ε ≈ 10^{-6} 
        # to discourage simultaneous charge/discharge while staying LP
        if not strategy_config.allow_simultaneous_charge_discharge:
            epsilon = strategy_config.epsilon_penalty
            simultaneous_penalty = epsilon * cp.sum(P_t_ch + P_t_dis)
            total_cost += simultaneous_penalty
        
        # Solve
        problem = cp.Problem(cp.Minimize(total_cost), constraints)
        problem.solve(verbose=False)
        
        if problem.status == cp.OPTIMAL:
            return self._create_result_from_cvxpy(
                strategy, G_t_in.value, G_t_out.value, 
                P_t_ch.value, P_t_dis.value, 
                SOC_t.value, total_cost.value, "OPTIMAL",
                S_t_curt.value, P_t_p2p_buy.value if P_t_p2p_buy is not None else None,
                P_t_p2p_sell.value if P_t_p2p_sell is not None else None,
                L_t_tilde.value if L_t_tilde is not None else None
            )
        else:
            raise Exception(f"Optimization failed: {problem.status}")
    
    def _solve_scipy_optimization(self, net_load: np.ndarray, buy_prices: np.ndarray, 
                                 sell_prices: np.ndarray, strategy_params: Dict, strategy: OptimizationStrategy) -> OptimizationResult:
        """Solve optimization using SciPy (fallback)"""
        print("🔧 Using SciPy optimization solver...")
        
        # Simplified optimization for SciPy
        # Variables: [grid_import_0, ..., grid_import_23, battery_charge_0, ..., battery_charge_23, ...]
        n_vars = 24 * 4  # grid_import, grid_export, battery_charge, battery_discharge
        
        # Objective: minimize cost
        c = np.zeros(n_vars)
        c[0:24] = buy_prices  # grid_import costs
        c[24:48] = -sell_prices  # grid_export rewards (negative for minimization)
        
        # Constraints: A_ub * x <= b_ub
        A_ub = []
        b_ub = []
        
        # Battery power limits
        for t in range(24):
            # battery_charge[t] <= max_charge
            constraint = np.zeros(n_vars)
            constraint[48 + t] = 1  # battery_charge[t]
            A_ub.append(constraint)
            b_ub.append(self.battery.max_charge_kw)
            
            # battery_discharge[t] <= max_discharge
            constraint = np.zeros(n_vars)
            constraint[72 + t] = 1  # battery_discharge[t]
            A_ub.append(constraint)
            b_ub.append(self.battery.max_discharge_kw)
        
        # Energy balance constraints
        for t in range(24):
            if net_load[t] >= 0:  # Need energy
                constraint = np.zeros(n_vars)
                constraint[t] = 1  # grid_import[t]
                constraint[48 + t] = -1  # battery_charge[t]
                constraint[72 + t] = 1  # battery_discharge[t]
                A_ub.append(constraint)
                b_ub.append(net_load[t])
                
                # grid_export[t] = 0
                constraint = np.zeros(n_vars)
                constraint[24 + t] = 1  # grid_export[t]
                A_ub.append(constraint)
                b_ub.append(0)
            else:  # Excess energy
                constraint = np.zeros(n_vars)
                constraint[24 + t] = 1  # grid_export[t]
                constraint[48 + t] = 1  # battery_charge[t]
                constraint[72 + t] = -1  # battery_discharge[t]
                A_ub.append(constraint)
                b_ub.append(-net_load[t])
                
                # grid_import[t] = 0
                constraint = np.zeros(n_vars)
                constraint[t] = 1  # grid_import[t]
                A_ub.append(constraint)
                b_ub.append(0)
        
        # Solve
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
        
        if result.success:
            x = result.x
            grid_import = x[0:24]
            grid_export = x[24:48]
            battery_charge = x[48:72]
            battery_discharge = x[72:96]
            
            # Calculate SOC trajectory
            battery_soc = self._calculate_soc_trajectory(battery_charge, battery_discharge)
            
            return self._create_result_from_arrays(
                strategy, grid_import, grid_export, battery_charge, 
                battery_discharge, battery_soc, result.fun, "OPTIMAL"
            )
        else:
            raise Exception(f"Optimization failed: {result.message}")
    
    def _solve_simplified_optimization(self, net_load: np.ndarray, buy_prices: np.ndarray, 
                                     sell_prices: np.ndarray, strategy_params: Dict, strategy: OptimizationStrategy) -> OptimizationResult:
        """Simplified optimization without external solvers"""
        print("🔧 Using simplified optimization...")
        
        # Initialize arrays
        grid_import = np.zeros(24)
        grid_export = np.zeros(24)
        battery_charge = np.zeros(24)
        battery_discharge = np.zeros(24)
        battery_soc = np.zeros(25)
        battery_soc[0] = self.battery.initial_soc * self.battery.capacity_kwh
        
        total_cost = 0.0
        
        # Simple greedy strategy
        for t in range(24):
            current_soc = battery_soc[t]
            available_capacity = (self.battery.soc_max - current_soc / self.battery.capacity_kwh) * self.battery.capacity_kwh
            available_energy = (current_soc / self.battery.capacity_kwh - self.battery.soc_min) * self.battery.capacity_kwh
            
            if net_load[t] >= 0:  # Need energy
                # Try to use battery first
                if available_energy > 0:
                    discharge = min(net_load[t], available_energy, self.battery.max_discharge_kw)
                    battery_discharge[t] = discharge
                    remaining = net_load[t] - discharge
                else:
                    remaining = net_load[t]
                
                # Import from grid
                grid_import[t] = remaining
                total_cost += remaining * buy_prices[t]
                
            else:  # Excess energy
                excess = -net_load[t]
                
                # Try to charge battery first
                if available_capacity > 0:
                    charge = min(excess, available_capacity, self.battery.max_charge_kw)
                    battery_charge[t] = charge
                    remaining = excess - charge
                else:
                    remaining = excess
                
                # Export to grid
                grid_export[t] = remaining
                total_cost -= remaining * sell_prices[t]
            
            # Update SOC
            battery_soc[t+1] = battery_soc[t] + battery_charge[t] * self.battery.charge_efficiency - battery_discharge[t] / self.battery.discharge_efficiency
        
        return self._create_result_from_arrays(
            strategy, 
            grid_import, grid_export, battery_charge, battery_discharge, 
            battery_soc, total_cost, "SIMPLIFIED"
        )
    
    def _calculate_soc_trajectory(self, battery_charge: np.ndarray, battery_discharge: np.ndarray) -> np.ndarray:
        """Calculate battery SOC trajectory"""
        soc = np.zeros(25)
        soc[0] = self.battery.initial_soc * self.battery.capacity_kwh
        
        for t in range(24):
            soc[t+1] = soc[t] + battery_charge[t] * self.battery.charge_efficiency - battery_discharge[t] / self.battery.discharge_efficiency
        
        return soc
    
    def _create_result_from_cvxpy(self, strategy: OptimizationStrategy, grid_import: np.ndarray, 
                                 grid_export: np.ndarray, battery_charge: np.ndarray, 
                                 battery_discharge: np.ndarray, battery_soc: np.ndarray, 
                                 total_cost: float, status: str, pv_curtailment: np.ndarray = None,
                                 p2p_buy: np.ndarray = None, p2p_sell: np.ndarray = None,
                                 dr_adjusted_load: np.ndarray = None) -> OptimizationResult:
        """Create result object from CVXPY solution"""
        hourly_results = []
        net_load = self.calculate_net_load()
        
        for t in range(24):
            result = {
                'hour': t + 1,
                'net_load_kw': net_load[t],
                'grid_import_kw': grid_import[t],
                'grid_export_kw': grid_export[t],
                'battery_charge_kw': battery_charge[t],
                'battery_discharge_kw': battery_discharge[t],
                'battery_soc_kwh': battery_soc[t+1],
                'battery_soc_percent': (battery_soc[t+1] / self.battery.capacity_kwh) * 100,
                'pv_curtailment_kw': pv_curtailment[t] if pv_curtailment is not None else 0.0
            }
            
            # Add P2P variables if applicable
            if p2p_buy is not None and p2p_sell is not None:
                result['p2p_buy_kw'] = p2p_buy[t]
                result['p2p_sell_kw'] = p2p_sell[t]
            else:
                result['p2p_buy_kw'] = 0.0
                result['p2p_sell_kw'] = 0.0
            
            # Add DR-adjusted load if applicable
            if dr_adjusted_load is not None:
                result['dr_adjusted_load_kw'] = dr_adjusted_load[t]
                result['dr_load_reduction_kw'] = net_load[t] - dr_adjusted_load[t]
            else:
                result['dr_adjusted_load_kw'] = net_load[t]
                result['dr_load_reduction_kw'] = 0.0
            
            hourly_results.append(result)
        
        return OptimizationResult(
            strategy=strategy,
            total_cost_eur=total_cost,
            hourly_results=hourly_results,
            battery_soc_trajectory=battery_soc.tolist(),
            grid_import_kwh=grid_import.tolist(),
            grid_export_kwh=grid_export.tolist(),
            battery_charge_kwh=battery_charge.tolist(),
            battery_discharge_kwh=battery_discharge.tolist(),
            optimization_status=status,
            pv_curtailment_kwh=pv_curtailment.tolist() if pv_curtailment is not None else None,
            p2p_buy_kwh=p2p_buy.tolist() if p2p_buy is not None else None,
            p2p_sell_kwh=p2p_sell.tolist() if p2p_sell is not None else None,
            dr_adjusted_load_kwh=dr_adjusted_load.tolist() if dr_adjusted_load is not None else None
        )
    
    def _create_result_from_arrays(self, strategy: OptimizationStrategy, grid_import: np.ndarray, 
                                  grid_export: np.ndarray, battery_charge: np.ndarray, 
                                  battery_discharge: np.ndarray, battery_soc: np.ndarray, 
                                  total_cost: float, status: str) -> OptimizationResult:
        """Create result object from array solution"""
        hourly_results = []
        for t in range(24):
            hourly_results.append({
                'hour': t + 1,
                'net_load_kw': self.calculate_net_load()[t],
                'grid_import_kw': grid_import[t],
                'grid_export_kw': grid_export[t],
                'battery_charge_kw': battery_charge[t],
                'battery_discharge_kw': battery_discharge[t],
                'battery_soc_kwh': battery_soc[t+1],
                'battery_soc_percent': (battery_soc[t+1] / self.battery.capacity_kwh) * 100
            })
        
        return OptimizationResult(
            strategy=strategy,
            total_cost_eur=total_cost,
            hourly_results=hourly_results,
            battery_soc_trajectory=battery_soc.tolist(),
            grid_import_kwh=grid_import.tolist(),
            grid_export_kwh=grid_export.tolist(),
            battery_charge_kwh=battery_charge.tolist(),
            battery_discharge_kwh=battery_discharge.tolist(),
            optimization_status=status
        )
    
    def run_all_strategies(self) -> Dict[OptimizationStrategy, OptimizationResult]:
        """Run optimization for all four strategies"""
        print("🚀 Running optimization for all strategies...")
        
        # Initialize strategy adapter
        from strategy_adapter import StrategyAdapter, StrategyType
        strategy_adapter = StrategyAdapter()
        
        results = {}
        for strategy in OptimizationStrategy:
            try:
                # Create strategy config
                strategy_type_map = {
                    OptimizationStrategy.MSC: StrategyType.MSC,
                    OptimizationStrategy.TOU: StrategyType.TOU,
                    OptimizationStrategy.MMR_P2P: StrategyType.MMR_P2P,
                    OptimizationStrategy.DR_P2P: StrategyType.DR_P2P
                }
                strategy_type = strategy_type_map[strategy]
                strategy_config = strategy_adapter.get_strategy_config(strategy_type)
                
                # Run optimization with config
                result = self.optimize_strategy(strategy, strategy_config)
                results[strategy] = result
                print(f"✅ {strategy.value}: €{result.total_cost_eur:.2f}")
            except Exception as e:
                print(f"❌ {strategy.value}: {e}")
        
        return results
    
    def save_results(self, results: Dict[OptimizationStrategy, OptimizationResult], output_dir: str = "results"):
        """Save optimization results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save summary
        summary = []
        for strategy, result in results.items():
            summary.append({
                'strategy': strategy.value,
                'total_cost_eur': result.total_cost_eur,
                'optimization_status': result.optimization_status,
                'total_grid_import_kwh': sum(result.grid_import_kwh),
                'total_grid_export_kwh': sum(result.grid_export_kwh),
                'total_battery_charge_kwh': sum(result.battery_charge_kwh),
                'total_battery_discharge_kwh': sum(result.battery_discharge_kwh)
            })
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(os.path.join(output_dir, "optimization_summary.csv"), index=False)
        
        # Save detailed results for each strategy
        for strategy, result in results.items():
            strategy_name = strategy.name.lower()
            
            # Hourly results
            hourly_df = pd.DataFrame(result.hourly_results)
            hourly_df.to_csv(os.path.join(output_dir, f"{strategy_name}_hourly_results.csv"), index=False)
            
            # SOC trajectory
            soc_df = pd.DataFrame({
                'hour': range(25),
                'soc_kwh': result.battery_soc_trajectory,
                'soc_percent': [(soc / self.battery.capacity_kwh) * 100 for soc in result.battery_soc_trajectory]
            })
            soc_df.to_csv(os.path.join(output_dir, f"{strategy_name}_soc_trajectory.csv"), index=False)
        
        print(f"📁 Results saved to {output_dir}/")

def main():
    """Main function to run optimization model"""
    print("=" * 60)
    print("24-HOUR ENERGY OPTIMIZATION MODEL")
    print("=" * 60)
    print("🔧 Building optimization model with real data...")
    
    try:
        # Initialize model
        model = EnergyOptimizationModel()
        
        # Run all strategies
        results = model.run_all_strategies()
        
        # Save results
        model.save_results(results)
        
        # Print summary
        print("\n" + "=" * 60)
        print("OPTIMIZATION RESULTS SUMMARY")
        print("=" * 60)
        
        for strategy, result in results.items():
            print(f"\n📊 {strategy.value}:")
            print(f"   💰 Total Cost: €{result.total_cost_eur:.2f}")
            print(f"   🔋 Battery Usage: {sum(result.battery_charge_kwh):.1f} kWh charged, {sum(result.battery_discharge_kwh):.1f} kWh discharged")
            print(f"   ⚡ Grid Import: {sum(result.grid_import_kwh):.1f} kWh")
            print(f"   📤 Grid Export: {sum(result.grid_export_kwh):.1f} kWh")
            print(f"   ✅ Status: {result.optimization_status}")
        
        print("\n🎉 Optimization completed successfully!")
        print("📁 Detailed results saved to results/ directory")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
