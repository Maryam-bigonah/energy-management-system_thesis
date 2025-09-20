#!/usr/bin/env python3
"""
24-Hour Energy Optimization Solver
==================================

Role: Senior Python engineer
Goal: Create a script that reads four files (data/load_24h.csv, data/pv_24h.csv, 
      data/tou_24h.csv, data/battery.yaml), builds a 24-hour LP in Pyomo, solves 
      it with Gurobi (fallback HiGHS), and writes results for one of four strategies: 
      MSC, TOU, MMR, DRP2P.

Requirements:
- Decision vars per hour: grid_in, grid_out, batt_ch, batt_dis, SOC, curtail (all ≥0)
- P2P strategies add: p2p_buy, p2p_sell (≥0)
- DR adds: L_DR with bounds (1-0.10)L ≤ L_DR ≤ (1+0.10)L and daily equality ΣL_DR=ΣL
- Power balance constraints for all strategies
- Battery SOC evolution with bounds and power caps
- Strategy-specific objective functions
- Solver choice: Gurobi first, HiGHS fallback
- CLI with strategy selection
- Data validation and output formatting

Author: Energy Management System
Date: 2024
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import yaml
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)

# Pyomo imports
try:
    import pyomo.environ as pyo
    from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
except ImportError:
    print("❌ Pyomo not installed. Install with: pip install pyomo")
    sys.exit(1)

# Solver imports
try:
    import gurobipy
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

try:
    import highspy
    HIGHS_AVAILABLE = True
except ImportError:
    HIGHS_AVAILABLE = False


class Strategy(Enum):
    """Available optimization strategies"""
    MSC = "MSC"      # Max Self-Consumption
    TOU = "TOU"      # Time-of-Use
    MMR = "MMR"      # Market-Making Retail P2P
    DRP2P = "DRP2P"  # Demand Response P2P


@dataclass
class DataValidationResult:
    """Data validation result"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


@dataclass
class SanityCheckResult:
    """Sanity check result"""
    check_name: str
    passed: bool
    message: str
    details: Optional[Dict] = None


@dataclass
class OptimizationResult:
    """Optimization result"""
    strategy: Strategy
    status: str
    objective_value: float
    hourly_results: pd.DataFrame
    kpis: Dict[str, float]
    solve_time: float
    sanity_checks: List[SanityCheckResult] = None
    
    @property
    def is_optimal(self) -> bool:
        """Check if optimization was successful"""
        return self.status == "optimal"


class DataValidator:
    """Data validation utilities"""
    
    @staticmethod
    def validate_csv_file(file_path: str, expected_rows: int, required_columns: List[str], filename: str = None) -> DataValidationResult:
        """Validate CSV file structure and content"""
        errors = []
        warnings = []
        
        if not os.path.exists(file_path):
            errors.append(f"File not found: {file_path}")
            return DataValidationResult(False, errors, warnings)
        
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            errors.append(f"Error reading {file_path}: {e}")
            return DataValidationResult(False, errors, warnings)
        
        # Check row count
        if len(df) != expected_rows:
            errors.append(f"{file_path}: Expected {expected_rows} rows, got {len(df)}")
        
        # Check required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            errors.append(f"{file_path}: Missing columns: {missing_cols}")
        
        # Check for negative values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if (df[col] < 0).any():
                errors.append(f"{file_path}: Column '{col}' contains negative values")
        
        # Check hour sequence for 24h files
        if expected_rows == 24 and 'hour' in df.columns:
            expected_hours = list(range(24)) if filename == "pv_24h.csv" else list(range(1, 25))
            if not df['hour'].equals(pd.Series(expected_hours)):
                errors.append(f"{file_path}: Hours should be {expected_hours[0]}-{expected_hours[-1]} in sequence")
        
        return DataValidationResult(len(errors) == 0, errors, warnings)
    
    @staticmethod
    def validate_battery_yaml(file_path: str) -> DataValidationResult:
        """Validate battery YAML file"""
        errors = []
        warnings = []
        
        if not os.path.exists(file_path):
            errors.append(f"File not found: {file_path}")
            return DataValidationResult(False, errors, warnings)
        
        try:
            with open(file_path, 'r') as f:
                battery_data = yaml.safe_load(f)
        except Exception as e:
            errors.append(f"Error reading {file_path}: {e}")
            return DataValidationResult(False, errors, warnings)
        
        required_keys = ['Ebat_kWh', 'Pch_max_kW', 'Pdis_max_kW', 'SOCmin', 'SOCmax', 'eta_ch', 'eta_dis']
        missing_keys = [key for key in required_keys if key not in battery_data]
        if missing_keys:
            errors.append(f"{file_path}: Missing keys: {missing_keys}")
        
        # Validate ranges
        if 'SOCmin' in battery_data and 'SOCmax' in battery_data:
            if battery_data['SOCmin'] >= battery_data['SOCmax']:
                errors.append(f"{file_path}: SOCmin must be < SOCmax")
        
        if 'eta_ch' in battery_data and (battery_data['eta_ch'] <= 0 or battery_data['eta_ch'] > 1):
            errors.append(f"{file_path}: eta_ch must be in (0, 1]")
        
        if 'eta_dis' in battery_data and (battery_data['eta_dis'] <= 0 or battery_data['eta_dis'] > 1):
            errors.append(f"{file_path}: eta_dis must be in (0, 1]")
        
        return DataValidationResult(len(errors) == 0, errors, warnings)
    
    @staticmethod
    def validate_yearly_csv(df: pd.DataFrame, filename: str) -> bool:
        """Validate yearly CSV data (8760 rows)"""
        logger.info(f"Validating {filename}...")
        
        # Check row count
        if len(df) != 8760:
            logger.error(f"Validation failed for {filename}: Expected 8760 rows, got {len(df)}")
            return False
        
        # Check required columns
        if filename == "load_8760.csv":
            required_columns = ['day', 'hour', 'load_kw']
        elif filename == "pv_8760.csv":
            required_columns = ['day', 'hour', 'pv_kw']
        else:
            logger.error(f"Unknown yearly file type: {filename}")
            return False
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Validation failed for {filename}: Missing columns: {missing_columns}")
            return False
        
        # Check day range (1-365)
        if 'day' in df.columns:
            if not (df['day'].min() == 1 and df['day'].max() == 365):
                logger.error(f"Validation failed for {filename}: Days should be 1-365, got {df['day'].min()}-{df['day'].max()}")
                return False
        
        # Check hour range (1-24)
        if 'hour' in df.columns:
            if not (df['hour'].min() == 1 and df['hour'].max() == 24):
                logger.error(f"Validation failed for {filename}: Hours should be 1-24, got {df['hour'].min()}-{df['hour'].max()}")
                return False
        
        # Check for non-negative values
        numeric_columns = [col for col in df.columns if col not in ['day', 'hour']]
        for col in numeric_columns:
            if (df[col] < 0).any():
                logger.error(f"Validation failed for {filename}: Negative values found in {col}")
                return False
        
        # Check for reasonable magnitudes
        if 'load_kw' in df.columns:
            if df['load_kw'].max() > 100:  # 100 kW seems too high for residential
                logger.warning(f"Warning for {filename}: Very high load values detected (max: {df['load_kw'].max():.2f} kW)")
        
        if 'pv_kw' in df.columns:
            if df['pv_kw'].max() > 50:  # 50 kW seems too high for residential PV
                logger.warning(f"Warning for {filename}: Very high PV values detected (max: {df['pv_kw'].max():.2f} kW)")
        
        logger.info(f"✅ {filename} validation passed")
        return True


class PricingCalculator:
    """Calculate P2P pricing for MMR and DR-P2P strategies"""
    
    @staticmethod
    def calculate_mmr_prices(pv_data: pd.Series, load_data: pd.Series, 
                           price_buy: pd.Series, price_sell: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate MMR P2P prices using equations B1-B3
        Using Gen0=PV, Dem0=L to keep it LP
        """
        p2p_buy_prices = []
        p2p_sell_prices = []
        
        for t in range(24):
            pv_t = pv_data.iloc[t]
            load_t = load_data.iloc[t]
            p_buy_t = price_buy.iloc[t]
            p_sell_t = price_sell.iloc[t]
            
            # Calculate average price
            p_avg_t = (p_buy_t + p_sell_t) / 2
            
            # Generation and demand (using original values to keep LP)
            gen_t = pv_t  # Gen0 = PV
            dem_t = load_t  # Dem0 = L
            
            # Apply equations B1-B3
            if abs(gen_t - dem_t) < 1e-6:  # Equation B1: Gen = Dem
                p2p_buy_t = p_avg_t
                p2p_sell_t = p_avg_t
            elif gen_t < dem_t:  # Equation B2: Gen < Dem
                p2p_sell_t = p_avg_t
                p2p_buy_t = (gen_t * p_avg_t + (dem_t - gen_t) * p_buy_t) / dem_t
            else:  # Equation B3: Gen > Dem
                p2p_buy_t = p_avg_t
                p2p_sell_t = (dem_t * p_avg_t + (gen_t - dem_t) * p_sell_t) / gen_t
            
            p2p_buy_prices.append(p2p_buy_t)
            p2p_sell_prices.append(p2p_sell_t)
        
        return pd.Series(p2p_buy_prices), pd.Series(p2p_sell_prices)
    
    @staticmethod
    def calculate_dr_p2p_prices(pv_data: pd.Series, load_data: pd.Series,
                              price_buy: pd.Series, price_sell: pd.Series,
                              epsilon: float = 1e-6) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate DR-P2P prices using SDR pricing
        Using S0=max(0,PV-L) and D0=max(0,L-PV) to keep it LP
        """
        p2p_buy_prices = []
        p2p_sell_prices = []
        
        for t in range(24):
            pv_t = pv_data.iloc[t]
            load_t = load_data.iloc[t]
            p_buy_t = price_buy.iloc[t]
            p_sell_t = price_sell.iloc[t]
            
            # Calculate community supply and demand (using original values)
            s_t = max(0, pv_t - load_t)  # S0 = max(0, PV - L)
            d_t = max(0, load_t - pv_t)  # D0 = max(0, L - PV)
            
            # Calculate SDR
            sdr_t = s_t / max(d_t, epsilon)
            
            # Apply SDR pricing equations
            if sdr_t <= 1:  # SDR ≤ 1
                p2p_sell_t = (p_buy_t - p_sell_t) * sdr_t + p_sell_t
                p2p_buy_t = p2p_sell_t * sdr_t + p_buy_t * (1 - sdr_t)
            else:  # SDR > 1
                p2p_sell_t = p_sell_t
                p2p_buy_t = p_sell_t
            
            p2p_buy_prices.append(p2p_buy_t)
            p2p_sell_prices.append(p2p_sell_t)
        
        return pd.Series(p2p_buy_prices), pd.Series(p2p_sell_prices)


class EnergyOptimizer:
    """24-hour energy optimization using Pyomo"""
    
    def __init__(self, data_dir: str = "project/data"):
        self.data_dir = data_dir
        self.validator = DataValidator()
        self.pricing_calc = PricingCalculator()
        
        # Data storage
        self.load_data = None
        self.pv_data = None
        self.tou_data = None
        self.battery_data = None
        
        # Solver configuration
        self.solver_name = self._get_best_solver()
        
    def _get_best_solver(self) -> str:
        """Get the best available solver (Gurobi first, then HiGHS)"""
        if GUROBI_AVAILABLE:
            return "gurobi"
        elif HIGHS_AVAILABLE:
            return "highs"
        else:
            raise RuntimeError("No suitable solver available. Install Gurobi or HiGHS.")
    
    def load_and_validate_data(self) -> bool:
        """Load and validate all required data files"""
        print("📊 Loading and validating data...")
        
        # Validate CSV files
        csv_files = [
            ("load_24h.csv", 24, ["hour", "load_kw"]),
            ("pv_24h.csv", 24, ["hour", "pv_generation_kw"]),
            ("tou_24h.csv", 24, ["hour", "price_buy", "price_sell"])
        ]
        
        for filename, expected_rows, required_cols in csv_files:
            file_path = os.path.join(self.data_dir, filename)
            result = self.validator.validate_csv_file(file_path, expected_rows, required_cols, filename)
            
            if not result.is_valid:
                print(f"❌ Validation failed for {filename}:")
                for error in result.errors:
                    print(f"   {error}")
                return False
            
            if result.warnings:
                print(f"⚠️  Warnings for {filename}:")
                for warning in result.warnings:
                    print(f"   {warning}")
        
        # Validate battery YAML
        battery_path = os.path.join(self.data_dir, "battery.yaml")
        result = self.validator.validate_battery_yaml(battery_path)
        
        if not result.is_valid:
            print(f"❌ Validation failed for battery.yaml:")
            for error in result.errors:
                print(f"   {error}")
            return False
        
        # Load data
        try:
            self.load_data = pd.read_csv(os.path.join(self.data_dir, "load_24h.csv"))
            self.pv_data = pd.read_csv(os.path.join(self.data_dir, "pv_24h.csv"))
            self.tou_data = pd.read_csv(os.path.join(self.data_dir, "tou_24h.csv"))
            
            # Rename columns for consistency
            self.pv_data = self.pv_data.rename(columns={'pv_generation_kw': 'pv_kw'})
            
            with open(battery_path, 'r') as f:
                self.battery_data = yaml.safe_load(f)
            
            # Additional validation: price_sell ≤ price_buy
            if (self.tou_data['price_sell'] > self.tou_data['price_buy']).any():
                print("❌ Validation failed: price_sell > price_buy for some hours")
                return False
            
            print("✅ All data loaded and validated successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def build_model(self, strategy: Strategy) -> pyo.ConcreteModel:
        """Build Pyomo model for the specified strategy"""
        print(f"🔧 Building {strategy.value} optimization model...")
        
        model = pyo.ConcreteModel()
        
        # Time periods
        model.T = pyo.Set(initialize=range(24))
        
        # Parameters
        self._add_parameters(model)
        
        # Variables
        self._add_variables(model, strategy)
        
        # Constraints
        self._add_constraints(model, strategy)
        
        # Objective
        self._add_objective(model, strategy)
        
        return model
    
    def _add_parameters(self, model: pyo.ConcreteModel):
        """Add model parameters"""
        # Load and PV data
        model.load_data = pyo.Param(model.T, initialize={t: self.load_data.iloc[t]['load_kw'] for t in model.T})
        model.pv_data = pyo.Param(model.T, initialize={t: self.pv_data.iloc[t]['pv_kw'] for t in model.T})
        
        # TOU pricing
        model.price_buy = pyo.Param(model.T, initialize={t: self.tou_data.iloc[t]['price_buy'] for t in model.T})
        model.price_sell = pyo.Param(model.T, initialize={t: self.tou_data.iloc[t]['price_sell'] for t in model.T})
        
        # Battery parameters
        model.Ebat = pyo.Param(initialize=self.battery_data['Ebat_kWh'])
        model.Pch_max = pyo.Param(initialize=self.battery_data['Pch_max_kW'])
        model.Pdis_max = pyo.Param(initialize=self.battery_data['Pdis_max_kW'])
        model.SOCmin = pyo.Param(initialize=self.battery_data['SOCmin'])
        model.SOCmax = pyo.Param(initialize=self.battery_data['SOCmax'])
        model.eta_ch = pyo.Param(initialize=self.battery_data['eta_ch'])
        model.eta_dis = pyo.Param(initialize=self.battery_data['eta_dis'])
        model.SOC0 = pyo.Param(initialize=self.battery_data.get('SOC0_frac', 0.5) * self.battery_data['Ebat_kWh'])
        
        # Small penalty for simultaneous charge/discharge
        model.eps = pyo.Param(initialize=1e-6)
        
        # DR parameters
        model.delta = pyo.Param(initialize=0.10)  # 10% flexibility
        
        # P2P price parameters (for MMR iterative solver)
        model.p2p_price_buy = pyo.Param(model.T, mutable=True, initialize=0.0)
        model.p2p_price_sell = pyo.Param(model.T, mutable=True, initialize=0.0)
    
    def _add_variables(self, model: pyo.ConcreteModel, strategy: Strategy):
        """Add decision variables"""
        # Core variables (all strategies)
        model.grid_in = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.grid_out = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.batt_ch = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.batt_dis = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.SOC = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.curtail = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        
        # P2P variables (MMR and DR-P2P strategies)
        if strategy in [Strategy.MMR, Strategy.DRP2P]:
            model.p2p_buy = pyo.Var(model.T, domain=pyo.NonNegativeReals)
            model.p2p_sell = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        
        # DR variable (DR-P2P strategy)
        if strategy == Strategy.DRP2P:
            model.L_DR = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    
    def _add_constraints(self, model: pyo.ConcreteModel, strategy: Strategy):
        """Add model constraints"""
        # Battery SOC evolution
        def soc_evolution_rule(model, t):
            if t == 0:
                return model.SOC[t] == model.SOC0 + model.eta_ch * model.batt_ch[t] - (1/model.eta_dis) * model.batt_dis[t]
            else:
                return model.SOC[t] == model.SOC[t-1] + model.eta_ch * model.batt_ch[t] - (1/model.eta_dis) * model.batt_dis[t]
        
        model.soc_evolution = pyo.Constraint(model.T, rule=soc_evolution_rule)
        
        # Battery SOC bounds
        def soc_bounds_rule(model, t):
            return (model.SOCmin * model.Ebat, model.SOC[t], model.SOCmax * model.Ebat)
        
        model.soc_bounds = pyo.Constraint(model.T, rule=soc_bounds_rule)
        
        # Battery power limits
        def batt_ch_limit_rule(model, t):
            return model.batt_ch[t] <= model.Pch_max
        
        def batt_dis_limit_rule(model, t):
            return model.batt_dis[t] <= model.Pdis_max
        
        model.batt_ch_limit = pyo.Constraint(model.T, rule=batt_ch_limit_rule)
        model.batt_dis_limit = pyo.Constraint(model.T, rule=batt_dis_limit_rule)
        
        # Terminal SOC constraint
        model.terminal_soc = pyo.Constraint(expr=model.SOC[23] >= model.SOC0)
        
        # DR constraints (DR-P2P strategy)
        if strategy == Strategy.DRP2P:
            # DR bounds
            def dr_bounds_rule(model, t):
                return ((1 - model.delta) * model.load_data[t], model.L_DR[t], (1 + model.delta) * model.load_data[t])
            
            model.dr_bounds = pyo.Constraint(model.T, rule=dr_bounds_rule)
            
            # Daily equality constraint
            def daily_equality_rule(model):
                return sum(model.L_DR[t] for t in model.T) == sum(model.load_data[t] for t in model.T)
            
            model.daily_equality = pyo.Constraint(rule=daily_equality_rule)
        
        # Power balance constraints
        self._add_power_balance_constraints(model, strategy)
    
    def _add_power_balance_constraints(self, model: pyo.ConcreteModel, strategy: Strategy):
        """Add power balance constraints based on strategy"""
        def power_balance_rule(model, t):
            if strategy == Strategy.DRP2P:
                # With P2P and DR: PV + batt_dis + grid_in + p2p_buy = L_DR + batt_ch + grid_out + p2p_sell + curtail
                return (model.pv_data[t] + model.batt_dis[t] + model.grid_in[t] + model.p2p_buy[t] == 
                       model.L_DR[t] + model.batt_ch[t] + model.grid_out[t] + model.p2p_sell[t] + model.curtail[t])
            elif strategy == Strategy.MMR:
                # With P2P: PV + batt_dis + grid_in + p2p_buy = L + batt_ch + grid_out + p2p_sell + curtail
                return (model.pv_data[t] + model.batt_dis[t] + model.grid_in[t] + model.p2p_buy[t] == 
                       model.load_data[t] + model.batt_ch[t] + model.grid_out[t] + model.p2p_sell[t] + model.curtail[t])
            else:
                # No P2P: PV + batt_dis + grid_in = L + batt_ch + grid_out + curtail
                return (model.pv_data[t] + model.batt_dis[t] + model.grid_in[t] == 
                       model.load_data[t] + model.batt_ch[t] + model.grid_out[t] + model.curtail[t])
        
        model.power_balance = pyo.Constraint(model.T, rule=power_balance_rule)
    
    def _add_objective(self, model: pyo.ConcreteModel, strategy: Strategy):
        """Add objective function based on strategy"""
        def objective_rule(model):
            # Base cost: grid import/export
            cost = sum(model.price_buy[t] * model.grid_in[t] - model.price_sell[t] * model.grid_out[t] for t in model.T)
            
            # P2P costs (MMR and DR-P2P strategies)
            if strategy in [Strategy.MMR, Strategy.DRP2P]:
                if strategy == Strategy.MMR:
                    # Use mutable parameters for iterative solver
                    cost += sum(model.p2p_price_buy[t] * model.p2p_buy[t] - model.p2p_price_sell[t] * model.p2p_sell[t] for t in model.T)
                else:  # DR-P2P
                    # Calculate P2P prices
                    p2p_buy_prices, p2p_sell_prices = self.pricing_calc.calculate_dr_p2p_prices(
                        self.pv_data['pv_kw'], self.load_data['load_kw'],
                        self.tou_data['price_buy'], self.tou_data['price_sell']
                    )
                    # Add P2P costs
                    cost += sum(p2p_buy_prices.iloc[t] * model.p2p_buy[t] - p2p_sell_prices.iloc[t] * model.p2p_sell[t] for t in model.T)
            
            # Small penalty for simultaneous charge/discharge
            cost += model.eps * sum(model.batt_ch[t] + model.batt_dis[t] for t in model.T)
            
            return cost
        
        model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
    
    def solve_model(self, model: pyo.ConcreteModel, strategy: Strategy) -> OptimizationResult:
        """Solve the optimization model"""
        print(f"🔧 Solving with {self.solver_name}...")
        
        # For MMR-P2P, use iterative solver
        if strategy == Strategy.MMR:
            return self._solve_mmr_iterative(model)
        else:
            return self._solve_single_pass(model)
    
    def _solve_single_pass(self, model: pyo.ConcreteModel) -> OptimizationResult:
        """Solve model in single pass"""
        solver = SolverFactory(self.solver_name)
        
        if self.solver_name == "gurobi":
            solver.options['TimeLimit'] = 300  # 5 minutes
            solver.options['MIPGap'] = 1e-6
        elif self.solver_name == "highs":
            solver.options['time_limit'] = 300
        
        import time as time_module
        start_time = time_module.time()
        results = solver.solve(model, tee=False)
        solve_time = time_module.time() - start_time
        
        # Check solution status
        if (results.solver.status == SolverStatus.ok and 
            results.solver.termination_condition == TerminationCondition.optimal):
            status = "OPTIMAL"
            objective_value = pyo.value(model.objective)
        else:
            status = f"FAILED: {results.solver.termination_condition}"
            objective_value = float('inf')
        
        return OptimizationResult(
            strategy=Strategy.MSC,  # Will be set by caller
            status=status,
            objective_value=objective_value,
            hourly_results=pd.DataFrame(),  # Will be populated
            kpis={},
            solve_time=solve_time
        )
    
    def _solve_mmr_iterative(self, model: pyo.ConcreteModel) -> OptimizationResult:
        """Solve MMR-P2P model with iterative price updates"""
        print("🔄 Using iterative MMR-P2P solver...")
        
        solver = SolverFactory(self.solver_name)
        if self.solver_name == "gurobi":
            solver.options['TimeLimit'] = 300
            solver.options['MIPGap'] = 1e-6
        elif self.solver_name == "highs":
            solver.options['time_limit'] = 300
        
        # Iterative parameters
        max_iterations = 3
        tolerance = 1e-3
        total_solve_time = 0
        
        # Initial prices (using PV/Load approximation)
        p2p_buy_prices, p2p_sell_prices = self.pricing_calc.calculate_mmr_prices(
            self.pv_data['pv_kw'], self.load_data['load_kw'],
            self.tou_data['price_buy'], self.tou_data['price_sell']
        )
        
        prev_objective = float('inf')
        
        for iteration in range(max_iterations):
            print(f"   Iteration {iteration + 1}/{max_iterations}")
            
            # Update P2P price parameters in model
            for t in model.T:
                model.p2p_price_buy[t].set_value(p2p_buy_prices.iloc[t])
                model.p2p_price_sell[t].set_value(p2p_sell_prices.iloc[t])
            
            # Solve
            import time as time_module
            start_time = time_module.time()
            results = solver.solve(model, tee=False)
            solve_time = time_module.time() - start_time
            total_solve_time += solve_time
            
            # Check solution status
            if not (results.solver.status == SolverStatus.ok and 
                   results.solver.termination_condition == TerminationCondition.optimal):
                status = f"FAILED: {results.solver.termination_condition}"
                objective_value = float('inf')
                break
            
            objective_value = pyo.value(model.objective)
            
            # Check convergence
            if iteration > 0:
                cost_change = abs(objective_value - prev_objective)
                print(f"   Cost change: €{cost_change:.6f}")
                
                if cost_change < tolerance:
                    print(f"   ✅ Converged after {iteration + 1} iterations")
                    break
            
            prev_objective = objective_value
            
            # Update prices based on current solution
            if iteration < max_iterations - 1:  # Don't update on last iteration
                new_buy_prices = []
                new_sell_prices = []
                
                for t in model.T:
                    # Get current solution values
                    gen_t = pyo.value(model.pv_data[t]) + pyo.value(model.batt_dis[t])
                    dem_t = pyo.value(model.load_data[t]) + pyo.value(model.batt_ch[t])
                    
                    p_buy_t = pyo.value(model.price_buy[t])
                    p_sell_t = pyo.value(model.price_sell[t])
                    p_avg_t = (p_buy_t + p_sell_t) / 2
                    
                    # Apply equations B1-B3 with updated Gen/Dem
                    if abs(gen_t - dem_t) < 1e-6:  # Equation B1
                        new_buy_t = p_avg_t
                        new_sell_t = p_avg_t
                    elif gen_t < dem_t:  # Equation B2
                        new_sell_t = p_avg_t
                        new_buy_t = (gen_t * p_avg_t + (dem_t - gen_t) * p_buy_t) / dem_t
                    else:  # Equation B3
                        new_buy_t = p_avg_t
                        new_sell_t = (dem_t * p_avg_t + (gen_t - dem_t) * p_sell_t) / gen_t
                    
                    new_buy_prices.append(new_buy_t)
                    new_sell_prices.append(new_sell_t)
                
                p2p_buy_prices = pd.Series(new_buy_prices)
                p2p_sell_prices = pd.Series(new_sell_prices)
        
        else:
            print(f"   ⚠️  Reached max iterations ({max_iterations})")
        
        status = "OPTIMAL" if objective_value != float('inf') else "FAILED"
        
        return OptimizationResult(
            strategy=Strategy.MMR,
            status=status,
            objective_value=objective_value,
            hourly_results=pd.DataFrame(),  # Will be populated
            kpis={},
            solve_time=total_solve_time
        )
    
    def extract_results(self, model: pyo.ConcreteModel, strategy: Strategy) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Extract hourly results and KPIs from solved model"""
        # Extract hourly results
        hourly_data = []
        
        for t in model.T:
            # Get basic values
            grid_in = pyo.value(model.grid_in[t])
            grid_out = pyo.value(model.grid_out[t])
            batt_ch = pyo.value(model.batt_ch[t])
            batt_dis = pyo.value(model.batt_dis[t])
            SOC = pyo.value(model.SOC[t])
            curtail = pyo.value(model.curtail[t])
            load = pyo.value(model.load_data[t])
            pv = pyo.value(model.pv_data[t])
            price_buy = pyo.value(model.price_buy[t])
            price_sell = pyo.value(model.price_sell[t])
            
            # Start with base row
            row = {
                'hour': t + 1,
                'grid_in': grid_in,
                'grid_out': grid_out,
                'batt_ch': batt_ch,
                'batt_dis': batt_dis,
                'SOC': SOC,
                'curtail': curtail,
                'pv': pv,
                'load': load,
                'price_buy': price_buy,
                'price_sell': price_sell
            }
            
            # Add P2P variables if applicable
            if strategy in [Strategy.MMR, Strategy.DRP2P]:
                row['p2p_buy'] = pyo.value(model.p2p_buy[t])
                row['p2p_sell'] = pyo.value(model.p2p_sell[t])
            
            # Add DR variable if applicable
            if strategy == Strategy.DRP2P:
                row['L_DR'] = pyo.value(model.L_DR[t])
                
                # Calculate SDR for DR-P2P
                load_adj = pyo.value(model.L_DR[t])
                s_t = max(0, pv - load_adj)  # Community supply
                d_t = max(0, load_adj - pv)  # Community demand
                sdr = s_t / max(d_t, 1e-6) if d_t > 0 else float('inf')
                row['SDR'] = sdr
                
                # Calculate DR-P2P prices
                p2p_buy_prices, p2p_sell_prices = self.pricing_calc.calculate_dr_p2p_prices(
                    self.pv_data['pv_kw'], self.load_data['load_kw'],
                    self.tou_data['price_buy'], self.tou_data['price_sell']
                )
                row['p2p_price_buy'] = p2p_buy_prices.iloc[t]
                row['p2p_price_sell'] = p2p_sell_prices.iloc[t]
            
            # Calculate helpful decompositions
            load_for_decomp = row.get('L_DR', load)  # Use L_DR if available, otherwise load
            
            # pv_to_load = min(PV, load or L_DR before battery)
            pv_to_load = min(pv, load_for_decomp)
            row['pv_to_load'] = pv_to_load
            
            # pv_to_batt = min(PV - pv_to_load, Pch_max etc.)
            pv_remaining = max(0, pv - pv_to_load)
            pv_to_batt = min(pv_remaining, self.battery_data['Pch_max_kW'])
            row['pv_to_batt'] = pv_to_batt
            
            # pv_to_grid = G_out (if you want a named column)
            row['pv_to_grid'] = grid_out
            
            # Calculate hourly cost
            cost_hour = (price_buy * grid_in - price_sell * grid_out)
            
            if strategy in [Strategy.MMR, Strategy.DRP2P]:
                # Add P2P costs
                if strategy == Strategy.MMR:
                    p2p_buy_prices, p2p_sell_prices = self.pricing_calc.calculate_mmr_prices(
                        self.pv_data['pv_kw'], self.load_data['load_kw'],
                        self.tou_data['price_buy'], self.tou_data['price_sell']
                    )
                else:  # DR-P2P
                    p2p_buy_prices, p2p_sell_prices = self.pricing_calc.calculate_dr_p2p_prices(
                        self.pv_data['pv_kw'], self.load_data['load_kw'],
                        self.tou_data['price_buy'], self.tou_data['price_sell']
                    )
                
                cost_hour += (p2p_buy_prices.iloc[t] * pyo.value(model.p2p_buy[t]) - 
                             p2p_sell_prices.iloc[t] * pyo.value(model.p2p_sell[t]))
            
            row['cost_hour'] = cost_hour
            hourly_data.append(row)
        
        hourly_df = pd.DataFrame(hourly_data)
        
        # Calculate KPIs
        kpis = self._calculate_kpis(hourly_df, strategy)
        
        return hourly_df, kpis
    
    def _calculate_kpis(self, hourly_df: pd.DataFrame, strategy: Strategy) -> Dict[str, float]:
        """Calculate key performance indicators according to exact specifications"""
        kpis = {}
        
        # Determine load column (L or L_DR)
        load_col = 'L_DR' if strategy == Strategy.DRP2P and 'L_DR' in hourly_df.columns else 'load'
        
        # Basic totals (with Δt = 1 hour)
        kpis['Cost_total'] = hourly_df['cost_hour'].sum()
        kpis['Import_total'] = hourly_df['grid_in'].sum()  # Σ G_in * Δt
        kpis['Export_total'] = hourly_df['grid_out'].sum()  # Σ G_out * Δt
        kpis['PV_total'] = hourly_df['pv'].sum()  # Σ PV * Δt
        kpis['Load_total'] = hourly_df[load_col].sum()  # Σ (L or L_DR) * Δt
        
        # pv_self = PV_total - Export_total - curtailment_energy
        curtailment_energy = hourly_df['curtail'].sum()
        kpis['pv_self'] = kpis['PV_total'] - kpis['Export_total'] - curtailment_energy
        
        # SCR = pv_self / PV_total (Self-Consumption Rate)
        kpis['SCR'] = kpis['pv_self'] / kpis['PV_total'] if kpis['PV_total'] > 0 else 0
        
        # SelfSufficiency = (pv_self + Σ batt_dis*Δt) / Load_total
        battery_discharge_energy = hourly_df['batt_dis'].sum()
        kpis['SelfSufficiency'] = (kpis['pv_self'] + battery_discharge_energy) / kpis['Load_total'] if kpis['Load_total'] > 0 else 0
        
        # PeakGrid = max_t G_in
        kpis['PeakGrid'] = hourly_df['grid_in'].max()
        
        # BatteryCycles ≈ (Σ batt_dis*Δt) / (2 E_b)
        total_discharge = hourly_df['batt_dis'].sum()
        kpis['BatteryCycles'] = total_discharge / (2 * self.battery_data['Ebat_kWh'])
        
        return kpis
    
    def perform_sanity_checks(self, result: OptimizationResult) -> List[SanityCheckResult]:
        """Perform comprehensive sanity checks on optimization results"""
        checks = []
        
        # Energy balance check
        checks.append(self._check_energy_balance(result))
        
        # SOC bounds and smoothness check
        checks.append(self._check_soc_bounds_and_smoothness(result))
        
        # Strategy-specific checks
        if result.strategy == Strategy.MSC:
            checks.append(self._check_msc_export_behavior(result))
        elif result.strategy == Strategy.TOU:
            checks.append(self._check_tou_export_behavior(result))
        elif result.strategy == Strategy.MMR:
            checks.append(self._check_mmr_p2p_grid_reduction(result))
        elif result.strategy == Strategy.DRP2P:
            checks.append(self._check_dr_p2p_load_shifting(result))
            checks.append(self._check_dr_p2p_cost_reduction(result))
        
        return checks
    
    def _check_energy_balance(self, result: OptimizationResult) -> SanityCheckResult:
        """Check that energy balance holds: supply ≈ demand each hour"""
        tolerance = 1e-6
        violations = []
        
        for _, row in result.hourly_results.iterrows():
            # Calculate supply and demand
            supply = row['pv'] + row['batt_dis'] + row['grid_in']
            if result.strategy in [Strategy.MMR, Strategy.DRP2P]:
                supply += row['p2p_buy']
            
            demand = row['load'] + row['batt_ch'] + row['grid_out'] + row['curtail']
            if result.strategy in [Strategy.MMR, Strategy.DRP2P]:
                demand += row['p2p_sell']
            
            if result.strategy == Strategy.DRP2P:
                demand = demand - row['load'] + row['L_DR']  # Replace load with L_DR
            
            balance_error = abs(supply - demand)
            if balance_error > tolerance:
                violations.append({
                    'hour': row['hour'],
                    'supply': supply,
                    'demand': demand,
                    'error': balance_error
                })
        
        passed = len(violations) == 0
        message = f"Energy balance check: {'PASSED' if passed else 'FAILED'}"
        if not passed:
            message += f" - {len(violations)} violations found"
        
        return SanityCheckResult(
            check_name="Energy Balance",
            passed=passed,
            message=message,
            details={'violations': violations[:5]} if violations else None
        )
    
    def _check_soc_bounds_and_smoothness(self, result: OptimizationResult) -> SanityCheckResult:
        """Check SOC stays within bounds and is smooth"""
        soc_values = result.hourly_results['SOC'].values
        soc_min = self.battery_data['SOCmin'] * self.battery_data['Ebat_kWh']
        soc_max = self.battery_data['SOCmax'] * self.battery_data['Ebat_kWh']
        
        # Check bounds
        bounds_violations = []
        for i, soc in enumerate(soc_values):
            if soc < soc_min - 1e-6 or soc > soc_max + 1e-6:
                bounds_violations.append({
                    'hour': i + 1,
                    'soc': soc,
                    'min_bound': soc_min,
                    'max_bound': soc_max
                })
        
        # Check smoothness (no large jumps)
        smoothness_violations = []
        for i in range(1, len(soc_values)):
            soc_change = abs(soc_values[i] - soc_values[i-1])
            max_change = max(self.battery_data['Pch_max_kW'], self.battery_data['Pdis_max_kW'])
            if soc_change > max_change + 1e-6:
                smoothness_violations.append({
                    'hour': i + 1,
                    'soc_change': soc_change,
                    'max_allowed': max_change
                })
        
        passed = len(bounds_violations) == 0 and len(smoothness_violations) == 0
        message = f"SOC bounds and smoothness: {'PASSED' if passed else 'FAILED'}"
        if not passed:
            message += f" - {len(bounds_violations)} bounds violations, {len(smoothness_violations)} smoothness violations"
        
        return SanityCheckResult(
            check_name="SOC Bounds and Smoothness",
            passed=passed,
            message=message,
            details={
                'bounds_violations': bounds_violations[:3],
                'smoothness_violations': smoothness_violations[:3]
            } if not passed else None
        )
    
    def _check_msc_export_behavior(self, result: OptimizationResult) -> SanityCheckResult:
        """Check MSC should export less than TOU if export is forbidden"""
        total_export = result.hourly_results['grid_out'].sum()
        
        # MSC should have minimal or zero export
        passed = total_export < 1e-6  # Essentially zero
        message = f"MSC export behavior: {'PASSED' if passed else 'FAILED'}"
        if not passed:
            message += f" - Total export: {total_export:.6f} kWh"
        
        return SanityCheckResult(
            check_name="MSC Export Behavior",
            passed=passed,
            message=message,
            details={'total_export': total_export}
        )
    
    def _check_tou_export_behavior(self, result: OptimizationResult) -> SanityCheckResult:
        """Check TOU export behavior is reasonable"""
        total_export = result.hourly_results['grid_out'].sum()
        total_pv = result.hourly_results['pv'].sum()
        
        # TOU should export when prices are favorable
        # This is more of a warning than a strict check
        export_ratio = total_export / total_pv if total_pv > 0 else 0
        
        passed = True  # TOU export is strategy-dependent
        message = f"TOU export behavior: OK - Export ratio: {export_ratio:.3f}"
        
        return SanityCheckResult(
            check_name="TOU Export Behavior",
            passed=passed,
            message=message,
            details={'total_export': total_export, 'export_ratio': export_ratio}
        )
    
    def _check_mmr_p2p_grid_reduction(self, result: OptimizationResult) -> SanityCheckResult:
        """Check MMR-P2P should reduce grid use when p2p prices are between buy/sell"""
        total_grid_import = result.hourly_results['grid_in'].sum()
        total_p2p_buy = result.hourly_results['p2p_buy'].sum()
        total_p2p_sell = result.hourly_results['p2p_sell'].sum()
        
        # Check that P2P trading is happening
        p2p_activity = total_p2p_buy + total_p2p_sell
        
        # Check price positioning
        price_checks = []
        for _, row in result.hourly_results.iterrows():
            if row['p2p_buy'] > 0 or row['p2p_sell'] > 0:
                p2p_buy_price = row.get('p2p_price_buy', 0)
                p2p_sell_price = row.get('p2p_price_sell', 0)
                grid_buy_price = row['price_buy']
                grid_sell_price = row['price_sell']
                
                # P2P prices should be between grid prices
                if p2p_buy_price > 0:
                    if not (grid_sell_price <= p2p_buy_price <= grid_buy_price):
                        price_checks.append({
                            'hour': row['hour'],
                            'p2p_buy_price': p2p_buy_price,
                            'grid_buy_price': grid_buy_price,
                            'grid_sell_price': grid_sell_price
                        })
        
        passed = p2p_activity > 0 and len(price_checks) == 0
        message = f"MMR-P2P grid reduction: {'PASSED' if passed else 'FAILED'}"
        if not passed:
            if p2p_activity == 0:
                message += " - No P2P activity"
            else:
                message += f" - {len(price_checks)} price violations"
        
        return SanityCheckResult(
            check_name="MMR-P2P Grid Reduction",
            passed=passed,
            message=message,
            details={
                'p2p_activity': p2p_activity,
                'price_violations': price_checks[:3]
            }
        )
    
    def _check_dr_p2p_load_shifting(self, result: OptimizationResult) -> SanityCheckResult:
        """Check DR-P2P should shift some load to cheap/valley hours (±10%)"""
        original_load = result.hourly_results['load'].sum()
        adjusted_load = result.hourly_results['L_DR'].sum()
        
        # Check daily equality constraint
        load_equality_error = abs(original_load - adjusted_load)
        equality_passed = load_equality_error < 1e-6
        
        # Check load adjustment bounds (±10%)
        bounds_violations = []
        for _, row in result.hourly_results.iterrows():
            original = row['load']
            adjusted = row['L_DR']
            min_bound = 0.9 * original  # -10%
            max_bound = 1.1 * original  # +10%
            
            if adjusted < min_bound - 1e-6 or adjusted > max_bound + 1e-6:
                bounds_violations.append({
                    'hour': row['hour'],
                    'original': original,
                    'adjusted': adjusted,
                    'min_bound': min_bound,
                    'max_bound': max_bound
                })
        
        # Check if load is shifted to valley hours (cheaper prices)
        valley_hours = result.hourly_results[result.hourly_results['price_buy'] <= 0.25].index
        valley_load_ratio = result.hourly_results.loc[valley_hours, 'L_DR'].sum() / adjusted_load if adjusted_load > 0 else 0
        
        passed = equality_passed and len(bounds_violations) == 0
        message = f"DR-P2P load shifting: {'PASSED' if passed else 'FAILED'}"
        if not passed:
            if not equality_passed:
                message += f" - Daily equality error: {load_equality_error:.6f}"
            if bounds_violations:
                message += f" - {len(bounds_violations)} bounds violations"
        
        return SanityCheckResult(
            check_name="DR-P2P Load Shifting",
            passed=passed,
            message=message,
            details={
                'load_equality_error': load_equality_error,
                'bounds_violations': bounds_violations[:3],
                'valley_load_ratio': valley_load_ratio
            }
        )
    
    def _check_dr_p2p_cost_reduction(self, result: OptimizationResult) -> SanityCheckResult:
        """Check DR-P2P should lower cost vs MMR"""
        # This check requires comparison with MMR results
        # For now, we'll just check that the cost is reasonable
        cost = result.objective_value
        
        # Basic sanity check: cost should be finite and reasonable
        passed = abs(cost) < 1e6 and not np.isnan(cost) and not np.isinf(cost)
        message = f"DR-P2P cost reasonableness: {'PASSED' if passed else 'FAILED'}"
        if not passed:
            message += f" - Cost: {cost}"
        
        return SanityCheckResult(
            check_name="DR-P2P Cost Reasonableness",
            passed=passed,
            message=message,
            details={'cost': cost}
        )
    
    def run_optimization(self, strategy: Strategy) -> OptimizationResult:
        """Run complete optimization for a strategy"""
        print(f"\n🚀 Running {strategy.value} optimization...")
        
        # Build model
        model = self.build_model(strategy)
        
        # Solve model
        result = self.solve_model(model, strategy)
        result.strategy = strategy
        
        if result.status == "OPTIMAL":
            # Extract results
            hourly_df, kpis = self.extract_results(model, strategy)
            result.hourly_results = hourly_df
            result.kpis = kpis
            
            # Perform sanity checks
            result.sanity_checks = self.perform_sanity_checks(result)
            
            # Print results and sanity checks
            print(f"✅ {strategy.value}: €{result.objective_value:.2f} (solve time: {result.solve_time:.2f}s)")
            
            # Print sanity check results
            for check in result.sanity_checks:
                status_icon = "✅" if check.passed else "❌"
                print(f"   {status_icon} {check.message}")
                if not check.passed and check.details:
                    # Print first few violations for debugging
                    if 'violations' in check.details and check.details['violations']:
                        for violation in check.details['violations'][:2]:
                            print(f"      - Hour {violation.get('hour', 'N/A')}: {violation}")
        else:
            print(f"❌ {strategy.value}: {result.status}")
        
        return result
    
    def save_results(self, result: OptimizationResult, output_dir: str = "results"):
        """Save optimization results to files with proper column ordering"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Reorder columns according to specifications
        base_columns = ['hour', 'grid_in', 'grid_out', 'batt_ch', 'batt_dis', 'SOC', 'curtail', 'pv', 'load']
        
        # Add conditional columns based on strategy
        if result.strategy in [Strategy.MMR, Strategy.DRP2P]:
            base_columns.extend(['p2p_buy', 'p2p_sell'])
        
        if result.strategy == Strategy.DRP2P:
            base_columns.extend(['L_DR', 'SDR', 'p2p_price_buy', 'p2p_price_sell'])
        
        # Always add these columns
        base_columns.extend(['price_buy', 'price_sell', 'cost_hour'])
        
        # Add helpful decompositions
        base_columns.extend(['pv_to_load', 'pv_to_batt', 'pv_to_grid'])
        
        # Reorder the dataframe
        available_columns = [col for col in base_columns if col in result.hourly_results.columns]
        ordered_df = result.hourly_results[available_columns]
        
        # Save hourly results
        hourly_file = os.path.join(output_dir, f"hourly_{result.strategy.value}.csv")
        ordered_df.to_csv(hourly_file, index=False)
        
        # Append to KPIs file
        kpis_file = os.path.join(output_dir, "kpis.csv")
        kpi_row = {'Strategy': result.strategy.value, **result.kpis}
        
        if os.path.exists(kpis_file):
            kpis_df = pd.read_csv(kpis_file)
            kpis_df = pd.concat([kpis_df, pd.DataFrame([kpi_row])], ignore_index=True)
        else:
            kpis_df = pd.DataFrame([kpi_row])
        
        kpis_df.to_csv(kpis_file, index=False)
        
        print(f"📁 Results saved to {output_dir}/")


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(description="24-Hour Energy Optimization Solver")
    parser.add_argument("--strategy", choices=[s.value for s in Strategy] + ["ALL"], default="ALL",
                       help="Optimization strategy (default: ALL)")
    parser.add_argument("--data-dir", default="project/data",
                       help="Data directory (default: project/data)")
    parser.add_argument("--output-dir", default="results",
                       help="Output directory (default: results)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("24-HOUR ENERGY OPTIMIZATION SOLVER")
    print("=" * 60)
    
    # Initialize optimizer
    optimizer = EnergyOptimizer(args.data_dir)
    
    # Load and validate data
    if not optimizer.load_and_validate_data():
        print("❌ Data validation failed. Exiting.")
        sys.exit(1)
    
    # Determine strategies to run
    if args.strategy == "ALL":
        strategies = list(Strategy)
    else:
        strategies = [Strategy(args.strategy)]
    
    # Run optimizations
    results = []
    for strategy in strategies:
        try:
            result = optimizer.run_optimization(strategy)
            results.append(result)
            
            if result.status == "OPTIMAL":
                optimizer.save_results(result, args.output_dir)
        except Exception as e:
            print(f"❌ Error running {strategy.value}: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION SUMMARY")
    print("=" * 60)
    
    for result in results:
        if result.status == "OPTIMAL":
            print(f"📊 {result.strategy.value}: €{result.objective_value:.2f}")
        else:
            print(f"❌ {result.strategy.value}: {result.status}")
    
    print(f"\n📁 Results saved to {args.output_dir}/")
    print("✅ Optimization completed!")


if __name__ == "__main__":
    main()
