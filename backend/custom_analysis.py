"""
Custom Solar Analysis API Endpoints

This module provides endpoints for custom solar analysis with user-defined parameters
for PV capacity estimation and PVGIS simulation.
"""

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import pandas as pd
import json
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torino_energy.data_sources.osm_roofs import OSMBuildingProcessor
from torino_energy.data_sources.pvgis_integration import BuildingSolarAnalyzer

router = APIRouter()

# Pydantic models for request validation
class PVParameters(BaseModel):
    """PV capacity estimation parameters."""
    coverage_factor: float = Field(0.65, ge=0.1, le=1.0, description="Rooftop coverage factor (0.1-1.0)")
    module_area: float = Field(2.0, gt=0, description="PV module area in m²")
    module_power: float = Field(420, gt=0, description="Module rated power in W")
    cells_per_module: int = Field(144, gt=0, description="Number of cells per module")

class PVGISParameters(BaseModel):
    """PVGIS simulation parameters."""
    location: str = Field("Torino, Italy", description="Location for simulation")
    tilt_angle: float = Field(35, ge=0, le=90, description="Tilt angle in degrees")
    azimuth: float = Field(0, ge=-180, le=180, description="Azimuth angle in degrees (0=south)")
    system_losses: float = Field(14, ge=0, le=50, description="System losses in percentage")
    mounting_position: str = Field("building_integrated", description="Mounting position")

class BuildingData(BaseModel):
    """Building data input."""
    osm_id: str = Field(..., description="OSM ID of the building")
    name: Optional[str] = Field(None, description="Building name")
    building_type: str = Field("apartments", description="Type of building")
    category: str = Field("Residential", description="Building category")
    height: float = Field(18, gt=0, description="Building height in meters")
    floors: int = Field(6, gt=0, description="Number of floors")
    footprint_area: float = Field(968.77, gt=0, description="Footprint area in m²")
    roof_area: float = Field(1000.99, gt=0, description="Roof area in m²")
    estimated_population: int = Field(142, gt=0, description="Estimated population")
    latitude: float = Field(45.0447177, description="Latitude coordinate")
    longitude: float = Field(7.6367993, description="Longitude coordinate")
    address: Optional[str] = Field(None, description="Building address")
    construction_year: Optional[str] = Field(None, description="Construction year")
    material: Optional[str] = Field(None, description="Building material")
    energy_class: Optional[str] = Field(None, description="Energy efficiency class")

class CustomAnalysisRequest(BaseModel):
    """Complete custom analysis request."""
    building: BuildingData
    pv_parameters: Optional[PVParameters] = Field(default_factory=PVParameters)
    pvgis_parameters: Optional[PVGISParameters] = Field(default_factory=PVGISParameters)
    custom_roof_area: Optional[float] = Field(None, gt=0, description="Custom roof area override")

class PVCapacityResult(BaseModel):
    """PV capacity calculation results."""
    roof_area_m2: float
    usable_pv_area_m2: float
    num_modules: int
    installed_capacity_kw: float
    total_cells: int
    coverage_factor: float

class CustomAnalysisResponse(BaseModel):
    """Complete custom analysis response."""
    building_id: str
    pv_capacity: PVCapacityResult
    pv_parameters: PVParameters
    pvgis_parameters: PVGISParameters
    system_config: Optional[Dict[str, Any]] = None
    estimated_analysis: Optional[Dict[str, Any]] = None
    success: bool
    message: str

# Initialize analyzer
analyzer = BuildingSolarAnalyzer()

@router.post("/custom-analysis", response_model=CustomAnalysisResponse)
async def run_custom_analysis(request: CustomAnalysisRequest):
    """
    Run custom solar analysis with user-defined parameters.
    
    This endpoint allows users to:
    1. Define custom PV capacity estimation parameters
    2. Configure PVGIS simulation parameters
    3. Input building-specific data
    4. Get detailed solar analysis results
    """
    try:
        # Extract data from request
        building_data = request.building.dict()
        pv_params = request.pv_parameters.dict()
        pvgis_params = request.pvgis_parameters.dict()
        
        # Use custom roof area if provided
        roof_area = request.custom_roof_area or building_data['roof_area']
        
        # Step 1: Calculate PV capacity
        pv_capacity = calculate_pv_capacity(roof_area, pv_params)
        
        # Step 2: Try PVGIS simulation
        system_config = None
        estimated_analysis = None
        
        try:
            # Create DataFrame for analyzer
            df = pd.DataFrame([building_data])
            analyzer.select_building(df, index=0)
            
            # Update analyzer parameters
            analyzer.pvgis_client.tilt = pvgis_params['tilt_angle']
            analyzer.pvgis_client.azimuth = pvgis_params['azimuth']
            analyzer.pvgis_client.system_loss = pvgis_params['system_losses'] / 100
            
            # Run analysis
            system_config = analyzer.calculate_optimal_system(roof_area=roof_area)
            
        except Exception as e:
            # Fallback to estimated analysis
            estimated_analysis = calculate_estimated_analysis(
                building_data, pv_capacity, pvgis_params
            )
        
        return CustomAnalysisResponse(
            building_id=building_data['osm_id'],
            pv_capacity=PVCapacityResult(**pv_capacity),
            pv_parameters=PVParameters(**pv_params),
            pvgis_parameters=PVGISParameters(**pvgis_params),
            system_config=system_config,
            estimated_analysis=estimated_analysis,
            success=True,
            message="Analysis completed successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/pv-capacity", response_model=PVCapacityResult)
async def calculate_pv_capacity_only(
    roof_area: float = Body(..., gt=0, description="Roof area in m²"),
    pv_params: PVParameters = Body(default_factory=PVParameters)
):
    """
    Calculate PV capacity only with custom parameters.
    
    This endpoint calculates:
    - Usable PV area based on coverage factor
    - Number of modules that can be installed
    - Total installed capacity in kW
    - Total number of cells
    """
    try:
        result = calculate_pv_capacity(roof_area, pv_params.dict())
        return PVCapacityResult(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Calculation failed: {str(e)}")

@router.get("/parameter-presets")
async def get_parameter_presets():
    """
    Get predefined parameter presets for different scenarios.
    """
    presets = {
        "residential_torino": {
            "name": "Residential Buildings in Torino",
            "description": "Optimized for residential buildings in Torino, Italy",
            "pv_parameters": {
                "coverage_factor": 0.65,
                "module_area": 2.0,
                "module_power": 420,
                "cells_per_module": 144
            },
            "pvgis_parameters": {
                "location": "Torino, Italy",
                "tilt_angle": 35,
                "azimuth": 0,
                "system_losses": 14,
                "mounting_position": "building_integrated"
            }
        },
        "commercial_torino": {
            "name": "Commercial Buildings in Torino",
            "description": "Optimized for commercial buildings in Torino, Italy",
            "pv_parameters": {
                "coverage_factor": 0.75,
                "module_area": 2.0,
                "module_power": 450,
                "cells_per_module": 144
            },
            "pvgis_parameters": {
                "location": "Torino, Italy",
                "tilt_angle": 30,
                "azimuth": 0,
                "system_losses": 12,
                "mounting_position": "building_integrated"
            }
        },
        "high_efficiency": {
            "name": "High Efficiency Configuration",
            "description": "Maximum efficiency setup with premium components",
            "pv_parameters": {
                "coverage_factor": 0.8,
                "module_area": 2.2,
                "module_power": 500,
                "cells_per_module": 144
            },
            "pvgis_parameters": {
                "location": "Torino, Italy",
                "tilt_angle": 35,
                "azimuth": 0,
                "system_losses": 10,
                "mounting_position": "building_integrated"
            }
        },
        "cost_optimized": {
            "name": "Cost Optimized Configuration",
            "description": "Balanced cost and performance setup",
            "pv_parameters": {
                "coverage_factor": 0.6,
                "module_area": 1.8,
                "module_power": 380,
                "cells_per_module": 120
            },
            "pvgis_parameters": {
                "location": "Torino, Italy",
                "tilt_angle": 35,
                "azimuth": 0,
                "system_losses": 16,
                "mounting_position": "building_integrated"
            }
        }
    }
    
    return presets

def calculate_pv_capacity(roof_area: float, pv_params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate PV capacity based on parameters."""
    # A_PV = coverage_factor × A_roof
    usable_pv_area = pv_params['coverage_factor'] * roof_area
    
    # N_modules = A_PV / A_module
    num_modules = usable_pv_area / pv_params['module_area']
    
    # P_DC = N_modules × (P_Wp / 1000) [kW]
    installed_capacity = num_modules * (pv_params['module_power'] / 1000)
    
    # N_cells = N_modules × cells_per_module
    total_cells = num_modules * pv_params['cells_per_module']
    
    return {
        'roof_area_m2': roof_area,
        'usable_pv_area_m2': usable_pv_area,
        'num_modules': int(num_modules),
        'installed_capacity_kw': installed_capacity,
        'total_cells': int(total_cells),
        'coverage_factor': pv_params['coverage_factor']
    }

def calculate_estimated_analysis(
    building_data: Dict[str, Any], 
    pv_capacity: Dict[str, Any], 
    pvgis_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculate estimated analysis when PVGIS is not available."""
    roof_area = building_data['roof_area']
    lat = building_data['latitude']
    
    # Estimated calculations
    optimal_tilt = max(0, min(90, lat + 10))
    solar_irradiance = 1400  # kWh/m²/year for Torino
    efficiency = 0.22  # 22% for crystalline silicon
    system_loss = pvgis_params['system_losses'] / 100
    
    annual_energy = roof_area * solar_irradiance * efficiency * (1 - system_loss)
    peak_power = pv_capacity['installed_capacity_kw']
    
    return {
        'annual_energy_kwh': annual_energy,
        'energy_per_m2_kwh': annual_energy / roof_area,
        'capacity_factor': 0.18,  # Estimated
        'peak_power_kw': peak_power,
        'optimal_tilt_degrees': optimal_tilt,
        'azimuth_degrees': pvgis_params['azimuth'],
        'solar_irradiance_kwh_m2': solar_irradiance,
        'efficiency': efficiency,
        'system_losses_percent': pvgis_params['system_losses']
    }
