import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  CircularProgress,
  Alert,
  Chip,
  Tabs,
  Tab,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
} from '@mui/material';
import {
  Business,
  SolarPower,
  TrendingUp,
  People,
  LocationOn,
  Height,
  Layers,
} from '@mui/icons-material';
import { useParams, useNavigate } from 'react-router-dom';
import { apiService } from '../services/api';

// Chart components will be imported here
const SolarAnalysisChart = ({ data }) => {
  // Placeholder for chart component
  return (
    <Box sx={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <Typography>Chart component will be rendered here</Typography>
    </Box>
  );
};

const TechnologyComparisonChart = ({ data }) => {
  // Placeholder for chart component
  return (
    <Box sx={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <Typography>Technology comparison chart will be rendered here</Typography>
    </Box>
  );
};

const BuildingDetails = () => {
  const { buildingId } = useParams();
  const navigate = useNavigate();
  const [tabValue, setTabValue] = useState(0);
  
  const [building, setBuilding] = useState(null);
  const [solarAnalysis, setSolarAnalysis] = useState(null);
  const [technologyComparison, setTechnologyComparison] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  const [selectedTechnology, setSelectedTechnology] = useState('mono_crystalline');
  const [customRoofArea, setCustomRoofArea] = useState('');

  useEffect(() => {
    if (buildingId) {
      loadBuildingData();
    }
  }, [buildingId]);

  useEffect(() => {
    if (building && selectedTechnology) {
      loadSolarAnalysis();
    }
  }, [building, selectedTechnology, customRoofArea]);

  useEffect(() => {
    if (building) {
      loadTechnologyComparison();
    }
  }, [building]);

  const loadBuildingData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await apiService.getBuilding(buildingId);
      setBuilding(response.data);
      
    } catch (err) {
      console.error('Error loading building:', err);
      setError(err.response?.data?.detail || 'Failed to load building details');
    } finally {
      setLoading(false);
    }
  };

  const loadSolarAnalysis = async () => {
    try {
      const params = {
        technology: selectedTechnology,
        roof_area: customRoofArea ? parseFloat(customRoofArea) : null,
      };
      
      const response = await apiService.getSolarAnalysis(buildingId, params.technology, params.roof_area);
      setSolarAnalysis(response.data);
      
    } catch (err) {
      console.error('Error loading solar analysis:', err);
    }
  };

  const loadTechnologyComparison = async () => {
    try {
      const response = await apiService.getTechnologyComparison(buildingId);
      setTechnologyComparison(response.data);
      
    } catch (err) {
      console.error('Error loading technology comparison:', err);
    }
  };

  const handleTechnologyChange = (event) => {
    setSelectedTechnology(event.target.value);
  };

  const handleCustomRoofArea = () => {
    if (customRoofArea) {
      loadSolarAnalysis();
    }
  };

  const formatNumber = (num) => {
    return num?.toLocaleString() || '0';
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress size={60} />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        {error}
      </Alert>
    );
  }

  if (!building) {
    return (
      <Alert severity="warning" sx={{ mb: 2 }}>
        Building not found
      </Alert>
    );
  }

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom sx={{ fontWeight: 'bold' }}>
          🏢 {building.osm_id}
        </Typography>
        <Typography variant="h6" color="text.secondary" gutterBottom>
          {building.address || 'No address available'}
        </Typography>
        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mt: 2 }}>
          <Chip label={building.building_type} color="primary" />
          <Chip label={building.category} variant="outlined" />
        </Box>
      </Box>

      {/* Building Overview Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Height sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
              <Typography variant="h4" sx={{ fontWeight: 'bold' }}>
                {building.height}m
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Height
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Layers sx={{ fontSize: 40, color: 'secondary.main', mb: 1 }} />
              <Typography variant="h4" sx={{ fontWeight: 'bold' }}>
                {building.floors}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Floors
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Business sx={{ fontSize: 40, color: 'success.main', mb: 1 }} />
              <Typography variant="h4" sx={{ fontWeight: 'bold' }}>
                {Math.round(building.roof_area)} m²
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Roof Area
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <People sx={{ fontSize: 40, color: 'warning.main', mb: 1 }} />
              <Typography variant="h4" sx={{ fontWeight: 'bold' }}>
                {building.estimated_population}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Population
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Tabs */}
      <Card>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={tabValue} onChange={(e, newValue) => setTabValue(newValue)}>
            <Tab label="Solar Analysis" />
            <Tab label="Technology Comparison" />
            <Tab label="Building Details" />
          </Tabs>
        </Box>

        <CardContent>
          {/* Solar Analysis Tab */}
          {tabValue === 0 && (
            <Box>
              <Box sx={{ mb: 3, display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap' }}>
                <FormControl sx={{ minWidth: 200 }}>
                  <InputLabel>Solar Technology</InputLabel>
                  <Select
                    value={selectedTechnology}
                    label="Solar Technology"
                    onChange={handleTechnologyChange}
                  >
                    <MenuItem value="mono_crystalline">Mono-crystalline</MenuItem>
                    <MenuItem value="poly_crystalline">Poly-crystalline</MenuItem>
                    <MenuItem value="thin_film">Thin Film</MenuItem>
                    <MenuItem value="perovskite">Perovskite</MenuItem>
                  </Select>
                </FormControl>
                
                <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                  <Typography variant="body2">Custom roof area:</Typography>
                  <input
                    type="number"
                    value={customRoofArea}
                    onChange={(e) => setCustomRoofArea(e.target.value)}
                    placeholder={building.roof_area}
                    style={{ width: '100px', padding: '8px', border: '1px solid #ccc', borderRadius: '4px' }}
                  />
                  <Button variant="outlined" size="small" onClick={handleCustomRoofArea}>
                    Apply
                  </Button>
                </Box>
              </Box>

              {solarAnalysis && (
                <Grid container spacing={3}>
                  <Grid item xs={12} md={6}>
                    <Typography variant="h6" gutterBottom>
                      System Configuration
                    </Typography>
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="body2" color="text.secondary">
                        Peak Power: <strong>{solarAnalysis.system_configuration?.peak_power_kw?.toFixed(2)} kW</strong>
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Optimal Tilt: <strong>{solarAnalysis.system_configuration?.optimal_tilt_degrees?.toFixed(1)}°</strong>
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Optimal Azimuth: <strong>{solarAnalysis.system_configuration?.optimal_azimuth_degrees?.toFixed(1)}°</strong>
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Number of Panels: <strong>{solarAnalysis.system_configuration?.number_of_panels}</strong>
                      </Typography>
                    </Box>

                    <Typography variant="h6" gutterBottom>
                      Energy Production
                    </Typography>
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="body2" color="text.secondary">
                        Annual Energy: <strong>{formatNumber(solarAnalysis.energy_production?.annual_energy_kwh)} kWh/year</strong>
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Energy per m²: <strong>{solarAnalysis.energy_production?.energy_per_m2_kwh?.toFixed(1)} kWh/m²/year</strong>
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Capacity Factor: <strong>{(solarAnalysis.energy_production?.capacity_factor * 100)?.toFixed(1)}%</strong>
                      </Typography>
                    </Box>

                    <Typography variant="h6" gutterBottom>
                      Economic Analysis
                    </Typography>
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Total Cost: <strong>€{formatNumber(solarAnalysis.economic_analysis?.total_system_cost_eur)}</strong>
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Cost per kWh: <strong>€{solarAnalysis.economic_analysis?.cost_per_kwh_eur?.toFixed(3)}</strong>
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Payback Period: <strong>{solarAnalysis.economic_analysis?.payback_period_years?.toFixed(1)} years</strong>
                      </Typography>
                    </Box>
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <SolarAnalysisChart data={solarAnalysis} />
                  </Grid>
                </Grid>
              )}
            </Box>
          )}

          {/* Technology Comparison Tab */}
          {tabValue === 1 && (
            <Box>
              {technologyComparison && (
                <>
                  <Typography variant="h6" gutterBottom sx={{ mb: 3 }}>
                    Solar Technology Comparison
                  </Typography>
                  
                  <Grid container spacing={2}>
                    {technologyComparison.comparison.map((tech, index) => (
                      <Grid item xs={12} sm={6} md={4} key={tech.technology}>
                        <Card variant="outlined">
                          <CardContent>
                            <Typography variant="h6" gutterBottom>
                              {tech.technology.replace('_', ' ').toUpperCase()}
                            </Typography>
                            <Typography variant="body2" color="text.secondary" gutterBottom>
                              Efficiency: {tech.efficiency_percent.toFixed(1)}%
                            </Typography>
                            <Typography variant="body2" color="text.secondary" gutterBottom>
                              Annual Energy: {formatNumber(tech.annual_energy_kwh)} kWh
                            </Typography>
                            <Typography variant="body2" color="text.secondary" gutterBottom>
                              Total Cost: €{formatNumber(tech.total_cost_eur)}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              Payback: {tech.payback_period_years.toFixed(1)} years
                            </Typography>
                          </CardContent>
                        </Card>
                      </Grid>
                    ))}
                  </Grid>
                  
                  <Box sx={{ mt: 3 }}>
                    <TechnologyComparisonChart data={technologyComparison.comparison} />
                  </Box>
                </>
              )}
            </Box>
          )}

          {/* Building Details Tab */}
          {tabValue === 2 && (
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom>
                  Basic Information
                </Typography>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    OSM ID: <strong>{building.osm_id}</strong>
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Building Type: <strong>{building.building_type}</strong>
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Category: <strong>{building.category}</strong>
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Address: <strong>{building.address || 'N/A'}</strong>
                  </Typography>
                </Box>

                <Typography variant="h6" gutterBottom>
                  Dimensions
                </Typography>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    Height: <strong>{building.height} m</strong>
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Floors: <strong>{building.floors}</strong>
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Footprint Area: <strong>{Math.round(building.footprint_area)} m²</strong>
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Roof Area: <strong>{Math.round(building.roof_area)} m²</strong>
                  </Typography>
                </Box>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom>
                  Population & Energy
                </Typography>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    Estimated Population: <strong>{building.estimated_population}</strong>
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Energy per Person: <strong>{Math.round(building.energy_per_person)} kWh/year</strong>
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Energy per m²: <strong>{Math.round(building.energy_per_m2)} kWh/m²/year</strong>
                  </Typography>
                </Box>

                <Typography variant="h6" gutterBottom>
                  Location
                </Typography>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    Latitude: <strong>{building.latitude.toFixed(6)}</strong>
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Longitude: <strong>{building.longitude.toFixed(6)}</strong>
                  </Typography>
                </Box>
              </Grid>
            </Grid>
          )}
        </CardContent>
      </Card>
    </Box>
  );
};

export default BuildingDetails;
