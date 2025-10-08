"""
FastAPI Backend for Torino Solar Building Analysis

This backend provides REST API endpoints for:
- Building data retrieval and analysis
- Solar energy calculations using PVGIS
- Technology comparison
- Data visualization endpoints
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import sys
import os
import pandas as pd
import json
from typing import List, Optional, Dict, Any
import logging

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torino_energy.data_sources.osm_roofs import OSMBuildingProcessor
from torino_energy.data_sources.pvgis_integration import BuildingSolarAnalyzer
from custom_analysis import router as custom_router

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Torino Solar Building Analysis API",
    description="API for analyzing solar energy potential of buildings in Torino",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include custom analysis router
app.include_router(custom_router, prefix="/api/v1", tags=["Custom Analysis"])

# Global variables for data processors
building_processor = None
solar_analyzer = None

@app.on_event("startup")
async def startup_event():
    """Initialize data processors on startup."""
    global building_processor, solar_analyzer
    
    try:
        # Initialize building processor
        building_processor = OSMBuildingProcessor()
        
        # Load processed data if available
        processed_file = "/Users/mariabigonah/Desktop/thesis/code/processed_building_data.csv"
        if os.path.exists(processed_file):
            building_processor.load_from_csv(processed_file)
            logger.info(f"Loaded {len(building_processor.df)} buildings from CSV")
        else:
            # Create sample data
            sample_data = create_sample_data()
            building_processor.load_from_dataframe(sample_data)
            building_processor.df = building_processor.clean_data()
            logger.info(f"Created {len(building_processor.df)} sample buildings")
        
        # Initialize solar analyzer
        solar_analyzer = BuildingSolarAnalyzer()
        
        logger.info("Backend initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing backend: {e}")
        raise

def create_sample_data():
    """Create sample building data."""
    sample_buildings = [
        {
            'OSM ID': 'way/49062146',
            'Name': '',
            'Building Type': 'apartments',
            'Category': 'Residential',
            'Height (m)': 18,
            'Floors': 6,
            'Footprint Area (m²)': 968.77,
            'Roof Area (m²)': 1000.99,
            'Estimated Population': 142,
            'Flats': '',
            'Units': '',
            'Apartments': '',
            'Rooms': '',
            'Address': '',
            'Construction Year': '',
            'Material': '',
            'Energy Class': '',
            'Data Sources': 'osm',
            'Latitude': 45.0447177,
            'Longitude': 7.6367993
        },
        {
            'OSM ID': 'way/182810424',
            'Name': '',
            'Building Type': 'apartments',
            'Category': 'Residential',
            'Height (m)': 24,
            'Floors': 8,
            'Footprint Area (m²)': 675.73,
            'Roof Area (m²)': 675.73,
            'Estimated Population': 132,
            'Flats': '',
            'Units': '',
            'Apartments': '',
            'Rooms': '',
            'Address': '181, Via Filadelfia, Torino',
            'Construction Year': '',
            'Material': '',
            'Energy Class': '',
            'Data Sources': 'osm',
            'Latitude': 45.0450191,
            'Longitude': 7.6359563
        },
        {
            'OSM ID': 'way/123456789',
            'Name': '',
            'Building Type': 'apartments',
            'Category': 'Residential',
            'Height (m)': 17,
            'Floors': 5,
            'Footprint Area (m²)': 749.27,
            'Roof Area (m²)': 774.18,
            'Estimated Population': 94,
            'Flats': '',
            'Units': '',
            'Apartments': '',
            'Rooms': '',
            'Address': '',
            'Construction Year': '',
            'Material': '',
            'Energy Class': '',
            'Data Sources': 'osm',
            'Latitude': 45.0439202,
            'Longitude': 7.639273
        },
        {
            'OSM ID': 'way/987654321',
            'Name': '',
            'Building Type': 'apartments',
            'Category': 'Residential',
            'Height (m)': 30,
            'Floors': 10,
            'Footprint Area (m²)': 1180.39,
            'Roof Area (m²)': 1219.64,
            'Estimated Population': 110,
            'Flats': '',
            'Units': '',
            'Apartments': '',
            'Rooms': '',
            'Address': '',
            'Construction Year': '',
            'Material': '',
            'Energy Class': '',
            'Data Sources': 'osm',
            'Latitude': 45.0445000,
            'Longitude': 7.6370000
        },
        {
            'OSM ID': 'way/555666777',
            'Name': '',
            'Building Type': 'apartments',
            'Category': 'Residential',
            'Height (m)': 15,
            'Floors': 5,
            'Footprint Area (m²)': 850.00,
            'Roof Area (m²)': 850.00,
            'Estimated Population': 180,
            'Flats': '',
            'Units': '',
            'Apartments': '',
            'Rooms': '',
            'Address': '',
            'Construction Year': '',
            'Material': '',
            'Energy Class': '',
            'Data Sources': 'osm',
            'Latitude': 45.0455000,
            'Longitude': 7.6380000
        }
    ]
    
    return pd.DataFrame(sample_buildings)

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Torino Solar Building Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "buildings": "/buildings",
            "building_details": "/buildings/{building_id}",
            "solar_analysis": "/buildings/{building_id}/solar-analysis",
            "technology_comparison": "/buildings/{building_id}/technology-comparison",
            "statistics": "/statistics"
        }
    }

@app.get("/buildings")
async def get_buildings(
    limit: Optional[int] = Query(None, description="Limit number of results"),
    building_type: Optional[str] = Query(None, description="Filter by building type"),
    min_roof_area: Optional[float] = Query(None, description="Minimum roof area in m²"),
    max_roof_area: Optional[float] = Query(None, description="Maximum roof area in m²")
):
    """Get list of buildings with optional filtering."""
    try:
        df = building_processor.df.copy()
        
        # Apply filters
        if building_type:
            df = df[df['Building Type'] == building_type]
        
        if min_roof_area is not None:
            df = df[df['Roof Area (m²)'] >= min_roof_area]
            
        if max_roof_area is not None:
            df = df[df['Roof Area (m²)'] <= max_roof_area]
        
        # Apply limit
        if limit:
            df = df.head(limit)
        
        # Convert to dict format for JSON response
        buildings = []
        for _, row in df.iterrows():
            building = {
                "osm_id": row['OSM ID'],
                "name": row.get('Name', ''),
                "building_type": row['Building Type'],
                "category": row['Category'],
                "height": row['Height (m)'],
                "floors": row['Floors'],
                "footprint_area": row['Footprint Area (m²)'],
                "roof_area": row['Roof Area (m²)'],
                "estimated_population": row['Estimated Population'],
                "address": row.get('Address', ''),
                "latitude": row['Latitude'],
                "longitude": row['Longitude'],
                "solar_potential": row.get('Solar_Potential_kWh_year', 0),
                "energy_per_person": row.get('Energy_per_Person_kWh_year', 0),
                "energy_per_m2": row.get('Energy_per_m2_kWh_year', 0)
            }
            buildings.append(building)
        
        return {
            "buildings": buildings,
            "total": len(buildings),
            "filters_applied": {
                "building_type": building_type,
                "min_roof_area": min_roof_area,
                "max_roof_area": max_roof_area,
                "limit": limit
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting buildings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/buildings/{building_id}")
async def get_building_details(building_id: str):
    """Get detailed information for a specific building."""
    try:
        df = building_processor.df
        building_row = df[df['OSM ID'] == building_id]
        
        if building_row.empty:
            raise HTTPException(status_code=404, detail="Building not found")
        
        row = building_row.iloc[0]
        
        building = {
            "osm_id": row['OSM ID'],
            "name": row.get('Name', ''),
            "building_type": row['Building Type'],
            "category": row['Category'],
            "height": row['Height (m)'],
            "floors": row['Floors'],
            "footprint_area": row['Footprint Area (m²)'],
            "roof_area": row['Roof Area (m²)'],
            "estimated_population": row['Estimated Population'],
            "flats": row.get('Flats', ''),
            "units": row.get('Units', ''),
            "apartments": row.get('Apartments', ''),
            "rooms": row.get('Rooms', ''),
            "address": row.get('Address', ''),
            "construction_year": row.get('Construction Year', ''),
            "material": row.get('Material', ''),
            "energy_class": row.get('Energy Class', ''),
            "data_sources": row['Data Sources'],
            "latitude": row['Latitude'],
            "longitude": row['Longitude'],
            "solar_potential": row.get('Solar_Potential_kWh_year', 0),
            "energy_per_person": row.get('Energy_per_Person_kWh_year', 0),
            "energy_per_m2": row.get('Energy_per_m2_kWh_year', 0)
        }
        
        return building
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting building details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/buildings/{building_id}/solar-analysis")
async def get_solar_analysis(
    building_id: str,
    technology: str = Query("mono_crystalline", description="Solar technology type"),
    roof_area: Optional[float] = Query(None, description="Custom roof area in m²")
):
    """Get solar analysis for a specific building."""
    try:
        df = building_processor.df
        building_row = df[df['OSM ID'] == building_id]
        
        if building_row.empty:
            raise HTTPException(status_code=404, detail="Building not found")
        
        # Select building for analysis
        building_idx = building_row.index[0]
        solar_analyzer.select_building(df, index=building_idx)
        
        # Calculate optimal system
        system_config = solar_analyzer.calculate_optimal_system(
            technology=technology,
            roof_area=roof_area
        )
        
        return system_config
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting solar analysis: {e}")
        # Return estimated analysis if PVGIS fails
        try:
            df = building_processor.df
            building_row = df[df['OSM ID'] == building_id]
            if not building_row.empty:
                row = building_row.iloc[0]
                return get_estimated_analysis(row, technology, roof_area)
        except:
            pass
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/buildings/{building_id}/technology-comparison")
async def get_technology_comparison(building_id: str):
    """Compare different solar technologies for a building."""
    try:
        df = building_processor.df
        building_row = df[df['OSM ID'] == building_id]
        
        if building_row.empty:
            raise HTTPException(status_code=404, detail="Building not found")
        
        # Select building for analysis
        building_idx = building_row.index[0]
        solar_analyzer.select_building(df, index=building_idx)
        
        # Compare technologies
        comparison_df = solar_analyzer.compare_technologies()
        
        # Convert to list of dictionaries
        comparison = []
        for _, row in comparison_df.iterrows():
            comparison.append({
                "technology": row['Technology'],
                "efficiency_percent": row['Efficiency (%)'],
                "annual_energy_kwh": row['Annual Energy (kWh)'],
                "total_cost_eur": row['Total Cost (EUR)'],
                "payback_period_years": row['Payback Period (years)'],
                "cost_per_kwh_eur": row['Cost per kWh (EUR)']
            })
        
        return {
            "building_id": building_id,
            "comparison": comparison
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting technology comparison: {e}")
        # Return estimated comparison if PVGIS fails
        try:
            df = building_processor.df
            building_row = df[df['OSM ID'] == building_id]
            if not building_row.empty:
                row = building_row.iloc[0]
                return get_estimated_comparison(row)
        except:
            pass
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics")
async def get_statistics():
    """Get overall statistics for the building dataset."""
    try:
        stats = building_processor.get_building_statistics()
        
        # Add some additional statistics
        df = building_processor.df
        total_solar_potential = df['Solar_Potential_kWh_year'].sum() if 'Solar_Potential_kWh_year' in df.columns else 0
        total_population = df['Estimated Population'].sum()
        avg_energy_per_person = df['Energy_per_Person_kWh_year'].mean() if 'Energy_per_Person_kWh_year' in df.columns else 0
        
        stats.update({
            "total_solar_potential_kwh": total_solar_potential,
            "total_population": total_population,
            "average_energy_per_person_kwh": avg_energy_per_person,
            "building_types": df['Building Type'].value_counts().to_dict(),
            "roof_area_distribution": {
                "min": df['Roof Area (m²)'].min(),
                "max": df['Roof Area (m²)'].max(),
                "mean": df['Roof Area (m²)'].mean(),
                "median": df['Roof Area (m²)'].median()
            }
        })
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/plot-data/roof-area-distribution")
async def get_roof_area_distribution():
    """Get data for roof area distribution plot."""
    try:
        df = building_processor.df
        return {
            "roof_areas": df['Roof Area (m²)'].tolist(),
            "building_types": df['Building Type'].tolist(),
            "osm_ids": df['OSM ID'].tolist()
        }
    except Exception as e:
        logger.error(f"Error getting roof area distribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/plot-data/solar-potential")
async def get_solar_potential_data():
    """Get data for solar potential visualization."""
    try:
        df = building_processor.df
        if 'Solar_Potential_kWh_year' not in df.columns:
            raise HTTPException(status_code=404, detail="Solar potential data not available")
        
        return {
            "osm_ids": df['OSM ID'].tolist(),
            "solar_potentials": df['Solar_Potential_kWh_year'].tolist(),
            "roof_areas": df['Roof Area (m²)'].tolist(),
            "populations": df['Estimated Population'].tolist(),
            "addresses": df.get('Address', [''] * len(df)).tolist(),
            "latitudes": df['Latitude'].tolist(),
            "longitudes": df['Longitude'].tolist()
        }
    except Exception as e:
        logger.error(f"Error getting solar potential data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def get_estimated_analysis(building_row, technology, roof_area=None):
    """Get estimated solar analysis when PVGIS is not available."""
    roof_area = roof_area or building_row['Roof Area (m²)']
    lat = building_row['Latitude']
    
    # Technology specifications
    tech_specs = {
        "mono_crystalline": {"efficiency": 0.22, "power_per_m2": 220, "cost_per_m2": 180},
        "poly_crystalline": {"efficiency": 0.18, "power_per_m2": 180, "cost_per_m2": 140},
        "thin_film": {"efficiency": 0.12, "power_per_m2": 120, "cost_per_m2": 100},
        "perovskite": {"efficiency": 0.25, "power_per_m2": 250, "cost_per_m2": 200}
    }
    
    specs = tech_specs.get(technology, tech_specs["mono_crystalline"])
    
    # Estimated calculations
    optimal_tilt = max(0, min(90, lat + 10))
    solar_irradiance = 1400  # kWh/m²/year for Torino
    system_loss = 0.14  # 14% system losses
    
    annual_energy = roof_area * solar_irradiance * specs["efficiency"] * (1 - system_loss)
    peak_power = (roof_area * specs["power_per_m2"]) / 1000  # kW
    total_cost = roof_area * specs["cost_per_m2"]
    
    return {
        "building_id": building_row['OSM ID'],
        "building_address": building_row.get('Address', ''),
        "roof_area_m2": roof_area,
        "technology": technology,
        "cell_specifications": {
            "efficiency_percent": specs["efficiency"] * 100,
            "power_per_m2_w": specs["power_per_m2"],
            "cost_per_m2_eur": specs["cost_per_m2"],
            "lifespan_years": 25
        },
        "system_configuration": {
            "peak_power_kw": peak_power,
            "optimal_tilt_degrees": optimal_tilt,
            "optimal_azimuth_degrees": 180,  # South-facing
            "number_of_panels": int(roof_area / 2),  # Assuming 2 m² per panel
            "panel_area_m2": roof_area
        },
        "energy_production": {
            "annual_energy_kwh": annual_energy,
            "energy_per_m2_kwh": annual_energy / roof_area,
            "capacity_factor": 0.18  # Estimated
        },
        "economic_analysis": {
            "total_system_cost_eur": total_cost,
            "cost_per_kwh_eur": total_cost / annual_energy,
            "payback_period_years": total_cost / (annual_energy * 0.15)  # Assuming 15 cent/kWh
        }
    }

def get_estimated_comparison(building_row):
    """Get estimated technology comparison when PVGIS is not available."""
    roof_area = building_row['Roof Area (m²)']
    
    technologies = ["mono_crystalline", "poly_crystalline", "thin_film", "perovskite"]
    comparison = []
    
    for tech in technologies:
        analysis = get_estimated_analysis(building_row, tech)
        comparison.append({
            "technology": tech,
            "efficiency_percent": analysis["cell_specifications"]["efficiency_percent"],
            "annual_energy_kwh": analysis["energy_production"]["annual_energy_kwh"],
            "total_cost_eur": analysis["economic_analysis"]["total_system_cost_eur"],
            "payback_period_years": analysis["economic_analysis"]["payback_period_years"],
            "cost_per_kwh_eur": analysis["economic_analysis"]["cost_per_kwh_eur"]
        })
    
    return {
        "building_id": building_row['OSM ID'],
        "comparison": comparison
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
