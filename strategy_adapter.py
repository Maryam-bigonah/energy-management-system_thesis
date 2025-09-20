#!/usr/bin/env python3
"""
Strategy Adapter for Energy Optimization Model
Config object system for switching between optimization strategies

This module provides a strategy adapter that creates configuration objects
to tell the model what to include before building/solving the LP.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import numpy as np

class StrategyType(Enum):
    """Available optimization strategies"""
    MSC = "Max Self-Consumption"
    TOU = "Time-of-Use"
    MMR_P2P = "Market-Making Retail P2P"
    DR_P2P = "Demand Response P2P"

@dataclass
class StrategyConfig:
    """Base configuration object for optimization strategies"""
    
    # Strategy identification
    strategy_type: StrategyType
    strategy_name: str
    description: str
    
    # Model components to include
    include_p2p_trading: bool = False
    include_dr_adjustment: bool = False
    include_curtailment: bool = True
    include_grid_export: bool = True
    include_grid_import: bool = True
    
    # Strategy-specific parameters
    p2p_buy_price: Optional[float] = None
    p2p_sell_price: Optional[float] = None
    p2p_single_price: Optional[float] = None  # For MMR-P2P
    
    # DR parameters
    dr_incentive: Optional[float] = None
    dr_max_increase: Optional[float] = None  # Max load increase factor
    
    # MSC-specific parameters
    msc_forbid_export: bool = False  # If True, set grid_out=0 and curtail surplus
    msc_priority_order: bool = True  # Enforce exact priority order
    
    # TOU-specific parameters
    tou_use_fig_b2_logic: bool = True  # Use Fig. B2 dispatch logic
    tou_valley_charge_priority: bool = True  # Valley → charge battery
    tou_peak_discharge_priority: bool = True  # Peak → discharge battery
    tou_flat_neutral: bool = True  # Flat → neutral (like MSC)
    tou_equations: Optional[Dict[str, str]] = None  # TOU dispatch equations
    
    # MMR-P2P specific parameters
    mmr_use_equations_b1_b3: bool = True  # Use equations B1-B3
    mmr_approximation_mode: str = 'first_pass'  # 'first_pass', 'iteration', 'approximation'
    mmr_max_iterations: int = 3  # Max iterations for convergence
    mmr_convergence_tolerance: float = 1e-3  # Convergence tolerance
    mmr_variables: Optional[Dict[str, str]] = None  # MMR-P2P variables
    mmr_definitions: Optional[Dict[str, str]] = None  # MMR-P2P definitions
    mmr_price_rules: Optional[Dict[str, str]] = None  # MMR-P2P price rules
    mmr_bilinear_handling: Optional[Dict[str, str]] = None  # Bilinear term handling
    
    # DR-P2P specific parameters
    dr_p2p_use_sdr_pricing: bool = True  # Use SDR-based pricing
    dr_p2p_delta: float = 0.10  # DR flexibility parameter (δ = 0.10)
    dr_p2p_sdr_epsilon: float = 1e-6  # Epsilon for SDR calculation (ε ≈ 10^-6)
    dr_p2p_daily_load_equality: bool = True  # Enforce daily load equality
    dr_p2p_sdr_pricing_equations: Optional[Dict[str, str]] = None  # SDR pricing equations
    dr_p2p_community_calculations: Optional[Dict[str, str]] = None  # Community supply/demand calculations
    dr_max_decrease: Optional[float] = None  # DR max decrease factor
    dr_bounds_constraint: Optional[str] = None  # DR bounds constraint
    daily_equality_constraint: Optional[str] = None  # Daily equality constraint
    
    # Objective function modifiers
    self_consumption_bonus: float = 0.0
    liquidity_bonus: float = 0.0
    export_penalty: float = 0.0
    
    # Constraint modifiers
    enforce_terminal_neutrality: bool = True  # SOC[24] >= SOC[0]
    allow_simultaneous_charge_discharge: bool = False
    
    # Numerical parameters
    epsilon_penalty: float = 1e-6  # Penalty for simultaneous charge/discharge
    min_power_threshold: float = 0.001  # Minimum power threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for easy access"""
        return {
            'strategy_type': self.strategy_type,
            'strategy_name': self.strategy_name,
            'description': self.description,
            'include_p2p_trading': self.include_p2p_trading,
            'include_dr_adjustment': self.include_dr_adjustment,
            'include_curtailment': self.include_curtailment,
            'include_grid_export': self.include_grid_export,
            'include_grid_import': self.include_grid_import,
            'p2p_buy_price': self.p2p_buy_price,
            'p2p_sell_price': self.p2p_sell_price,
            'p2p_single_price': self.p2p_single_price,
            'dr_incentive': self.dr_incentive,
            'dr_max_increase': self.dr_max_increase,
            'msc_forbid_export': self.msc_forbid_export,
            'msc_priority_order': self.msc_priority_order,
            'tou_use_fig_b2_logic': self.tou_use_fig_b2_logic,
            'tou_valley_charge_priority': self.tou_valley_charge_priority,
            'tou_peak_discharge_priority': self.tou_peak_discharge_priority,
            'tou_flat_neutral': self.tou_flat_neutral,
            'tou_equations': self.tou_equations,
            'mmr_use_equations_b1_b3': self.mmr_use_equations_b1_b3,
            'mmr_approximation_mode': self.mmr_approximation_mode,
            'mmr_max_iterations': self.mmr_max_iterations,
            'mmr_convergence_tolerance': self.mmr_convergence_tolerance,
            'mmr_variables': self.mmr_variables,
            'mmr_definitions': self.mmr_definitions,
            'mmr_price_rules': self.mmr_price_rules,
            'mmr_bilinear_handling': self.mmr_bilinear_handling,
            'dr_p2p_use_sdr_pricing': self.dr_p2p_use_sdr_pricing,
            'dr_p2p_delta': self.dr_p2p_delta,
            'dr_p2p_sdr_epsilon': self.dr_p2p_sdr_epsilon,
            'dr_p2p_daily_load_equality': self.dr_p2p_daily_load_equality,
            'dr_p2p_sdr_pricing_equations': self.dr_p2p_sdr_pricing_equations,
            'dr_p2p_community_calculations': self.dr_p2p_community_calculations,
            'dr_max_decrease': self.dr_max_decrease,
            'dr_bounds_constraint': self.dr_bounds_constraint,
            'daily_equality_constraint': self.daily_equality_constraint,
            'self_consumption_bonus': self.self_consumption_bonus,
            'liquidity_bonus': self.liquidity_bonus,
            'export_penalty': self.export_penalty,
            'enforce_terminal_neutrality': self.enforce_terminal_neutrality,
            'allow_simultaneous_charge_discharge': self.allow_simultaneous_charge_discharge,
            'epsilon_penalty': self.epsilon_penalty,
            'min_power_threshold': self.min_power_threshold
        }

class StrategyAdapter:
    """Strategy adapter for creating configuration objects"""
    
    def __init__(self):
        self.strategies = {
            StrategyType.MSC: self._create_msc_config,
            StrategyType.TOU: self._create_tou_config,
            StrategyType.MMR_P2P: self._create_mmr_p2p_config,
            StrategyType.DR_P2P: self._create_dr_p2p_config
        }
    
    def get_strategy_config(self, strategy_type: StrategyType, **kwargs) -> StrategyConfig:
        """Get configuration object for a specific strategy"""
        if strategy_type not in self.strategies:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        return self.strategies[strategy_type](**kwargs)
    
    def _create_msc_config(self, **kwargs) -> StrategyConfig:
        """Create MSC (Max Self-Consumption) strategy configuration"""
        return StrategyConfig(
            strategy_type=StrategyType.MSC,
            strategy_name="Max Self-Consumption",
            description="Maximize self-consumption, minimize grid dependency",
            
            # Model components
            include_p2p_trading=False,
            include_dr_adjustment=False,
            include_curtailment=True,
            include_grid_export=not kwargs.get('forbid_export', False),
            include_grid_import=True,
            
            # MSC-specific parameters
            msc_forbid_export=kwargs.get('forbid_export', False),
            msc_priority_order=kwargs.get('priority_order', True),
            
            # Objective modifiers
            self_consumption_bonus=kwargs.get('self_consumption_bonus', 0.05),
            export_penalty=kwargs.get('export_penalty', 0.0),
            
            # Constraints
            enforce_terminal_neutrality=True,
            allow_simultaneous_charge_discharge=False,
            
            # Numerical parameters
            epsilon_penalty=1e-6,
            min_power_threshold=0.001
        )
    
    def _create_tou_config(self, **kwargs) -> StrategyConfig:
        """Create TOU (Time-of-Use) strategy configuration"""
        return StrategyConfig(
            strategy_type=StrategyType.TOU,
            strategy_name="Time-of-Use Optimization",
            description="Optimize based on TOU pricing (F1/F2/F3) using Fig. B2 dispatch logic",
            
            # Model components
            include_p2p_trading=False,
            include_dr_adjustment=False,
            include_curtailment=True,
            include_grid_export=True,
            include_grid_import=True,
            
            # TOU-specific parameters
            tou_use_fig_b2_logic=kwargs.get('use_fig_b2_logic', True),
            tou_valley_charge_priority=kwargs.get('valley_charge_priority', True),
            tou_peak_discharge_priority=kwargs.get('peak_discharge_priority', True),
            tou_flat_neutral=kwargs.get('flat_neutral', True),
            
            # TOU dispatch equations
            tou_equations={
                'battery_charge': 'P_b_ch = min(P_pv - P_de, P_b_ch_max)',
                'grid_export': 'P_g_s = max(P_pv - P_de - P_b_ch, 0)',
                'battery_discharge': 'P_b_de = min(P_de - P_pv, P_b_dis_max)',
                'grid_import': 'P_g_de = max(P_de - P_pv - P_b_dis, 0)'
            },
            
            # Objective modifiers
            self_consumption_bonus=kwargs.get('self_consumption_bonus', 0.0),
            
            # Constraints
            enforce_terminal_neutrality=True,
            allow_simultaneous_charge_discharge=False,
            
            # Numerical parameters
            epsilon_penalty=1e-6,
            min_power_threshold=0.001
        )
    
    def _create_mmr_p2p_config(self, **kwargs) -> StrategyConfig:
        """Create MMR-P2P (Mid-Market Rate Peer-to-Peer) strategy configuration"""
        return StrategyConfig(
            strategy_type=StrategyType.MMR_P2P,
            strategy_name="Mid-Market Rate Peer-to-Peer",
            description="P2P trading with mid-market rate pricing using equations B1-B3",
            
            # Model components
            include_p2p_trading=True,
            include_dr_adjustment=False,
            include_curtailment=True,
            include_grid_export=True,
            include_grid_import=True,
            
            # MMR-P2P specific parameters
            mmr_use_equations_b1_b3=kwargs.get('use_equations_b1_b3', True),
            mmr_approximation_mode=kwargs.get('approximation_mode', 'first_pass'),  # 'first_pass', 'iteration', 'approximation'
            mmr_max_iterations=kwargs.get('max_iterations', 3),
            mmr_convergence_tolerance=kwargs.get('convergence_tolerance', 1e-3),
            
            # MMR-P2P variables
            mmr_variables={
                'P_t_p2p_buy': 'Power bought from peers at hour t',
                'P_t_p2p_sell': 'Power sold to peers at hour t'
            },
            
            # MMR-P2P definitions
            mmr_definitions={
                'P_avg_t': '(p_t_buy + p_t_sell) / 2',  # Mid-market rate
                'Gen_t': 'PV_t + P_t_dis',  # Total generation
                'Dem_t': 'L_t + P_t_ch'  # Total demand
            },
            
            # MMR-P2P price rules (equations B1-B3)
            mmr_price_rules={
                'equation_b1': 'If Gen_t = Dem_t: P_t_P2P_buy = P_t_P2P_sell = P_avg_t',
                'equation_b2': 'If Gen_t < Dem_t: P_t_P2P_sell = P_avg_t, P_t_P2P_buy = (Gen_t * P_avg_t + (Dem_t - Gen_t) * p_t_buy) / Dem_t',
                'equation_b3': 'If Gen_t > Dem_t: P_t_P2P_buy = P_avg_t, P_t_P2P_sell = (Dem_t * P_avg_t + (Gen_t - Dem_t) * p_t_sell) / Gen_t'
            },
            
            # Bilinear term handling
            mmr_bilinear_handling={
                'method': kwargs.get('bilinear_method', 'approximation'),  # 'first_pass', 'iteration', 'approximation'
                'approximation': 'Gen_t ≈ PV_t, Dem_t ≈ L_t',  # Simplification for linearity
                'note': 'Prices depend on P_t_ch and P_t_dis variables, creating bilinear terms'
            },
            
            # P2P parameters (fallback for approximation mode)
            p2p_single_price=kwargs.get('p2p_price', 0.30),
            
            # Objective modifiers
            liquidity_bonus=kwargs.get('liquidity_bonus', 0.02),
            
            # Constraints
            enforce_terminal_neutrality=True,
            allow_simultaneous_charge_discharge=False,
            
            # Numerical parameters
            epsilon_penalty=1e-6,
            min_power_threshold=0.001
        )
    
    def _create_dr_p2p_config(self, **kwargs) -> StrategyConfig:
        """Create DR-P2P (Demand Response + P2P with SDR pricing) strategy configuration"""
        return StrategyConfig(
            strategy_type=StrategyType.DR_P2P,
            strategy_name="Demand Response + P2P with SDR pricing",
            description="DR with P2P trading using System Demand Ratio (SDR) pricing",
            
            # Model components
            include_p2p_trading=True,
            include_dr_adjustment=True,
            include_curtailment=True,
            include_grid_export=True,
            include_grid_import=True,
            
            # DR-P2P specific parameters
            dr_p2p_use_sdr_pricing=kwargs.get('use_sdr_pricing', True),
            dr_p2p_delta=kwargs.get('delta', 0.10),  # δ = 0.10 (10% flexibility)
            dr_p2p_sdr_epsilon=kwargs.get('sdr_epsilon', 1e-6),  # ε ≈ 10^-6
            dr_p2p_daily_load_equality=kwargs.get('daily_load_equality', True),
            
            # DR load adjustment bounds
            dr_max_increase=kwargs.get('dr_max_increase', 1.10),  # (1 + δ) = 1.10
            dr_max_decrease=kwargs.get('dr_max_decrease', 0.90),  # (1 - δ) = 0.90
            
            # DR-P2P community calculations
            dr_p2p_community_calculations={
                'S_t': 'max(0, PV_t - L̃_t)',  # Community supply
                'D_t': 'max(0, L̃_t - PV_t)',  # Community demand
                'SDR_t': 'S_t / max(D_t, ε)'  # System Demand Ratio
            },
            
            # DR-P2P SDR pricing equations
            dr_p2p_sdr_pricing_equations={
                'p2p_sell_sdr_le_1': 'P2P_t^sell = (p_t^buy - p_t^sell)SDR_t + p_t^sell',
                'p2p_sell_sdr_gt_1': 'P2P_t^sell = p_t^sell',
                'p2p_buy_sdr_le_1': 'P2P_t^buy = P2P_t^sell ⋅ SDR_t + p_t^buy (1 - SDR_t)',
                'p2p_buy_sdr_gt_1': 'P2P_t^buy = p_t^sell'
            },
            
            # DR bounds constraint
            dr_bounds_constraint='(1 - δ)L_t ≤ L̃_t ≤ (1 + δ)L_t',
            
            # Daily equality constraint
            daily_equality_constraint='Σ_t L̃_t Δt = Σ_t L_t Δt',
            
            # P2P parameters (fallback for non-SDR mode)
            p2p_buy_price=kwargs.get('p2p_buy_price', 0.25),
            p2p_sell_price=kwargs.get('p2p_sell_price', 0.35),
            
            # DR parameters
            dr_incentive=kwargs.get('dr_incentive', 0.05),
            
            # Objective modifiers
            liquidity_bonus=kwargs.get('liquidity_bonus', 0.02),
            
            # Constraints
            enforce_terminal_neutrality=True,
            allow_simultaneous_charge_discharge=False,
            
            # Numerical parameters
            epsilon_penalty=1e-6,
            min_power_threshold=0.001
        )
    
    def list_available_strategies(self) -> List[StrategyType]:
        """List all available strategies"""
        return list(self.strategies.keys())
    
    def get_strategy_info(self, strategy_type: StrategyType) -> Dict[str, str]:
        """Get information about a strategy"""
        config = self.get_strategy_config(strategy_type)
        return {
            'name': config.strategy_name,
            'description': config.description,
            'components': {
                'P2P Trading': config.include_p2p_trading,
                'DR Adjustment': config.include_dr_adjustment,
                'Curtailment': config.include_curtailment,
                'Grid Export': config.include_grid_export,
                'Grid Import': config.include_grid_import
            }
        }

def create_msc_config(forbid_export: bool = False, priority_order: bool = True, 
                     self_consumption_bonus: float = 0.05) -> StrategyConfig:
    """Convenience function to create MSC configuration"""
    adapter = StrategyAdapter()
    return adapter.get_strategy_config(
        StrategyType.MSC,
        forbid_export=forbid_export,
        priority_order=priority_order,
        self_consumption_bonus=self_consumption_bonus
    )

def create_tou_config(self_consumption_bonus: float = 0.0) -> StrategyConfig:
    """Convenience function to create TOU configuration"""
    adapter = StrategyAdapter()
    return adapter.get_strategy_config(
        StrategyType.TOU,
        self_consumption_bonus=self_consumption_bonus
    )

def create_mmr_p2p_config(p2p_price: float = 0.30, liquidity_bonus: float = 0.02) -> StrategyConfig:
    """Convenience function to create MMR-P2P configuration"""
    adapter = StrategyAdapter()
    return adapter.get_strategy_config(
        StrategyType.MMR_P2P,
        p2p_price=p2p_price,
        liquidity_bonus=liquidity_bonus
    )

def create_dr_p2p_config(p2p_buy_price: float = 0.25, p2p_sell_price: float = 0.35,
                        dr_incentive: float = 0.05, dr_max_increase: float = 1.2) -> StrategyConfig:
    """Convenience function to create DR-P2P configuration"""
    adapter = StrategyAdapter()
    return adapter.get_strategy_config(
        StrategyType.DR_P2P,
        p2p_buy_price=p2p_buy_price,
        p2p_sell_price=p2p_sell_price,
        dr_incentive=dr_incentive,
        dr_max_increase=dr_max_increase
    )

def main():
    """Demonstrate strategy adapter usage"""
    print("=" * 60)
    print("STRATEGY ADAPTER DEMONSTRATION")
    print("=" * 60)
    
    adapter = StrategyAdapter()
    
    # List available strategies
    print("\n📋 Available Strategies:")
    for strategy in adapter.list_available_strategies():
        info = adapter.get_strategy_info(strategy)
        print(f"  • {strategy.value}: {info['description']}")
    
    # Demonstrate MSC configuration
    print("\n🔧 MSC (Max Self-Consumption) Configuration:")
    msc_config = create_msc_config(forbid_export=True, priority_order=True)
    print(f"  Strategy: {msc_config.strategy_name}")
    print(f"  Description: {msc_config.description}")
    print(f"  Components:")
    print(f"    - P2P Trading: {msc_config.include_p2p_trading}")
    print(f"    - DR Adjustment: {msc_config.include_dr_adjustment}")
    print(f"    - Curtailment: {msc_config.include_curtailment}")
    print(f"    - Grid Export: {msc_config.include_grid_export}")
    print(f"    - Grid Import: {msc_config.include_grid_import}")
    print(f"  MSC Parameters:")
    print(f"    - Forbid Export: {msc_config.msc_forbid_export}")
    print(f"    - Priority Order: {msc_config.msc_priority_order}")
    print(f"    - Self-Consumption Bonus: {msc_config.self_consumption_bonus}")
    
    # Demonstrate TOU configuration
    print("\n🔧 TOU (Time-of-Use) Configuration:")
    tou_config = create_tou_config()
    print(f"  Strategy: {tou_config.strategy_name}")
    print(f"  Description: {tou_config.description}")
    print(f"  Components:")
    print(f"    - P2P Trading: {tou_config.include_p2p_trading}")
    print(f"    - DR Adjustment: {tou_config.include_dr_adjustment}")
    print(f"    - Curtailment: {tou_config.include_curtailment}")
    print(f"    - Grid Export: {tou_config.include_grid_export}")
    print(f"    - Grid Import: {tou_config.include_grid_import}")
    print(f"  TOU Parameters:")
    print(f"    - Use Fig. B2 Logic: {tou_config.tou_use_fig_b2_logic}")
    print(f"    - Valley Charge Priority: {tou_config.tou_valley_charge_priority}")
    print(f"    - Peak Discharge Priority: {tou_config.tou_peak_discharge_priority}")
    print(f"    - Flat Neutral: {tou_config.tou_flat_neutral}")
    print(f"  TOU Dispatch Equations:")
    if tou_config.tou_equations:
        for eq_name, equation in tou_config.tou_equations.items():
            print(f"    - {eq_name}: {equation}")
    
    # Demonstrate MMR-P2P configuration
    print("\n🔧 MMR-P2P (Mid-Market Rate Peer-to-Peer) Configuration:")
    mmr_config = create_mmr_p2p_config()
    print(f"  Strategy: {mmr_config.strategy_name}")
    print(f"  Description: {mmr_config.description}")
    print(f"  Components:")
    print(f"    - P2P Trading: {mmr_config.include_p2p_trading}")
    print(f"    - DR Adjustment: {mmr_config.include_dr_adjustment}")
    print(f"    - Curtailment: {mmr_config.include_curtailment}")
    print(f"    - Grid Export: {mmr_config.include_grid_export}")
    print(f"    - Grid Import: {mmr_config.include_grid_import}")
    print(f"  MMR-P2P Parameters:")
    print(f"    - Use Equations B1-B3: {mmr_config.mmr_use_equations_b1_b3}")
    print(f"    - Approximation Mode: {mmr_config.mmr_approximation_mode}")
    print(f"    - Max Iterations: {mmr_config.mmr_max_iterations}")
    print(f"    - Convergence Tolerance: {mmr_config.mmr_convergence_tolerance}")
    print(f"  MMR-P2P Variables:")
    if mmr_config.mmr_variables:
        for var_name, var_desc in mmr_config.mmr_variables.items():
            print(f"    - {var_name}: {var_desc}")
    print(f"  MMR-P2P Definitions:")
    if mmr_config.mmr_definitions:
        for def_name, def_formula in mmr_config.mmr_definitions.items():
            print(f"    - {def_name}: {def_formula}")
    print(f"  MMR-P2P Price Rules (Equations B1-B3):")
    if mmr_config.mmr_price_rules:
        for eq_name, eq_rule in mmr_config.mmr_price_rules.items():
            print(f"    - {eq_name}: {eq_rule}")
    print(f"  Bilinear Term Handling:")
    if mmr_config.mmr_bilinear_handling:
        for key, value in mmr_config.mmr_bilinear_handling.items():
            print(f"    - {key}: {value}")
    print(f"  P2P Parameters (Fallback):")
    print(f"    - P2P Single Price: {mmr_config.p2p_single_price}")
    print(f"    - Liquidity Bonus: {mmr_config.liquidity_bonus}")
    
    # Demonstrate DR-P2P configuration
    print("\n🔧 DR-P2P (Demand Response + P2P with SDR pricing) Configuration:")
    dr_config = create_dr_p2p_config()
    print(f"  Strategy: {dr_config.strategy_name}")
    print(f"  Description: {dr_config.description}")
    print(f"  Components:")
    print(f"    - P2P Trading: {dr_config.include_p2p_trading}")
    print(f"    - DR Adjustment: {dr_config.include_dr_adjustment}")
    print(f"    - Curtailment: {dr_config.include_curtailment}")
    print(f"    - Grid Export: {dr_config.include_grid_export}")
    print(f"    - Grid Import: {dr_config.include_grid_import}")
    print(f"  DR-P2P Parameters:")
    print(f"    - Use SDR Pricing: {dr_config.dr_p2p_use_sdr_pricing}")
    print(f"    - DR Delta (δ): {dr_config.dr_p2p_delta}")
    print(f"    - SDR Epsilon (ε): {dr_config.dr_p2p_sdr_epsilon}")
    print(f"    - Daily Load Equality: {dr_config.dr_p2p_daily_load_equality}")
    print(f"  DR Load Adjustment Bounds:")
    print(f"    - DR Max Increase: {dr_config.dr_max_increase}")
    print(f"    - DR Max Decrease: {dr_config.dr_max_decrease}")
    print(f"    - DR Bounds Constraint: {dr_config.dr_bounds_constraint}")
    print(f"  Community Calculations:")
    if dr_config.dr_p2p_community_calculations:
        for calc_name, calc_formula in dr_config.dr_p2p_community_calculations.items():
            print(f"    - {calc_name}: {calc_formula}")
    print(f"  SDR Pricing Equations:")
    if dr_config.dr_p2p_sdr_pricing_equations:
        for eq_name, eq_formula in dr_config.dr_p2p_sdr_pricing_equations.items():
            print(f"    - {eq_name}: {eq_formula}")
    print(f"  Constraints:")
    print(f"    - Daily Equality: {dr_config.daily_equality_constraint}")
    print(f"  P2P Parameters (Fallback):")
    print(f"    - P2P Buy Price: {dr_config.p2p_buy_price}")
    print(f"    - P2P Sell Price: {dr_config.p2p_sell_price}")
    print(f"  DR Parameters:")
    print(f"    - DR Incentive: {dr_config.dr_incentive}")
    
    print("\n" + "=" * 60)
    print("✅ STRATEGY ADAPTER READY FOR USE")
    print("=" * 60)

if __name__ == "__main__":
    main()
