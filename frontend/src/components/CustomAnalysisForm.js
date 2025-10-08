import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Stepper,
  Step,
  StepLabel,
  Alert,
  CircularProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  Divider,
} from '@mui/material';
import {
  ExpandMore,
  Settings,
  Business,
  SolarPower,
  Calculate,
} from '@mui/icons-material';
import { apiService } from '../services/api';

const CustomAnalysisForm = () => {
  const [activeStep, setActiveStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [presets, setPresets] = useState({});

  // Form data
  const [buildingData, setBuildingData] = useState({
    osm_id: '',
    name: '',
    building_type: 'apartments',
    category: 'Residential',
    height: 18,
    floors: 6,
    footprint_area: 968.77,
    roof_area: 1000.99,
    estimated_population: 142,
    latitude: 45.0447177,
    longitude: 7.6367993,
    address: '',
    construction_year: '',
    material: '',
    energy_class: '',
  });

  const [pvParameters, setPvParameters] = useState({
    coverage_factor: 0.65,
    module_area: 2.0,
    module_power: 420,
    cells_per_module: 144,
  });

  const [pvgisParameters, setPvgisParameters] = useState({
    location: 'Torino, Italy',
    tilt_angle: 35,
    azimuth: 0,
    system_losses: 14,
    mounting_position: 'building_integrated',
  });

  const steps = [
    'Building Data',
    'PV Parameters',
    'PVGIS Parameters',
    'Analysis Results'
  ];

  const handleNext = () => {
    setActiveStep((prevActiveStep) => prevActiveStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevActiveStep) => prevActiveStep - 1);
  };

  const handleReset = () => {
    setActiveStep(0);
    setResult(null);
    setError(null);
  };

  const loadPresets = async () => {
    try {
      const response = await apiService.get('/api/v1/parameter-presets');
      setPresets(response.data);
    } catch (err) {
      console.error('Error loading presets:', err);
    }
  };

  React.useEffect(() => {
    loadPresets();
  }, []);

  const applyPreset = (presetName) => {
    if (presets[presetName]) {
      const preset = presets[presetName];
      setPvParameters(preset.pv_parameters);
      setPvgisParameters(preset.pvgis_parameters);
    }
  };

  const runAnalysis = async () => {
    try {
      setLoading(true);
      setError(null);

      const requestData = {
        building: buildingData,
        pv_parameters: pvParameters,
        pvgis_parameters: pvgisParameters,
      };

      const response = await apiService.post('/api/v1/custom-analysis', requestData);
      setResult(response.data);
      setActiveStep(3);

    } catch (err) {
      console.error('Error running analysis:', err);
      setError(err.response?.data?.detail || 'Analysis failed');
    } finally {
      setLoading(false);
    }
  };

  const renderBuildingDataStep = () => (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Business />
          Building Information
        </Typography>
        
        <Grid container spacing={3}>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="OSM ID *"
              value={buildingData.osm_id}
              onChange={(e) => setBuildingData({...buildingData, osm_id: e.target.value})}
              placeholder="e.g., way/123456"
              required
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Building Name"
              value={buildingData.name}
              onChange={(e) => setBuildingData({...buildingData, name: e.target.value})}
            />
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth>
              <InputLabel>Building Type</InputLabel>
              <Select
                value={buildingData.building_type}
                onChange={(e) => setBuildingData({...buildingData, building_type: e.target.value})}
              >
                <MenuItem value="apartments">Apartments</MenuItem>
                <MenuItem value="house">House</MenuItem>
                <MenuItem value="commercial">Commercial</MenuItem>
                <MenuItem value="industrial">Industrial</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth>
              <InputLabel>Category</InputLabel>
              <Select
                value={buildingData.category}
                onChange={(e) => setBuildingData({...buildingData, category: e.target.value})}
              >
                <MenuItem value="Residential">Residential</MenuItem>
                <MenuItem value="Commercial">Commercial</MenuItem>
                <MenuItem value="Industrial">Industrial</MenuItem>
                <MenuItem value="Public">Public</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <TextField
              fullWidth
              label="Height (m)"
              type="number"
              value={buildingData.height}
              onChange={(e) => setBuildingData({...buildingData, height: parseFloat(e.target.value) || 0})}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <TextField
              fullWidth
              label="Floors"
              type="number"
              value={buildingData.floors}
              onChange={(e) => setBuildingData({...buildingData, floors: parseInt(e.target.value) || 0})}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <TextField
              fullWidth
              label="Footprint Area (m²)"
              type="number"
              value={buildingData.footprint_area}
              onChange={(e) => setBuildingData({...buildingData, footprint_area: parseFloat(e.target.value) || 0})}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <TextField
              fullWidth
              label="Roof Area (m²) *"
              type="number"
              value={buildingData.roof_area}
              onChange={(e) => setBuildingData({...buildingData, roof_area: parseFloat(e.target.value) || 0})}
              required
            />
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Estimated Population"
              type="number"
              value={buildingData.estimated_population}
              onChange={(e) => setBuildingData({...buildingData, estimated_population: parseInt(e.target.value) || 0})}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Address"
              value={buildingData.address}
              onChange={(e) => setBuildingData({...buildingData, address: e.target.value})}
            />
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Latitude"
              type="number"
              value={buildingData.latitude}
              onChange={(e) => setBuildingData({...buildingData, latitude: parseFloat(e.target.value) || 0})}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Longitude"
              type="number"
              value={buildingData.longitude}
              onChange={(e) => setBuildingData({...buildingData, longitude: parseFloat(e.target.value) || 0})}
            />
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );

  const renderPVParametersStep = () => (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <SolarPower />
          PV Capacity Estimation Parameters
        </Typography>
        
        {/* Presets */}
        {Object.keys(presets).length > 0 && (
          <Box sx={{ mb: 3 }}>
            <Typography variant="subtitle1" gutterBottom>
              Quick Presets:
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              {Object.entries(presets).map(([key, preset]) => (
                <Chip
                  key={key}
                  label={preset.name}
                  onClick={() => applyPreset(key)}
                  variant="outlined"
                  size="small"
                />
              ))}
            </Box>
          </Box>
        )}
        
        <Grid container spacing={3}>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Coverage Factor"
              type="number"
              value={pvParameters.coverage_factor}
              onChange={(e) => setPvParameters({...pvParameters, coverage_factor: parseFloat(e.target.value) || 0})}
              helperText="Fraction of roof area usable for PV (0.1-1.0)"
              inputProps={{ min: 0.1, max: 1.0, step: 0.01 }}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Module Area (m²)"
              type="number"
              value={pvParameters.module_area}
              onChange={(e) => setPvParameters({...pvParameters, module_area: parseFloat(e.target.value) || 0})}
              helperText="Area of a single PV module"
            />
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Module Power (W)"
              type="number"
              value={pvParameters.module_power}
              onChange={(e) => setPvParameters({...pvParameters, module_power: parseFloat(e.target.value) || 0})}
              helperText="Rated power of a single module"
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Cells per Module"
              type="number"
              value={pvParameters.cells_per_module}
              onChange={(e) => setPvParameters({...pvParameters, cells_per_module: parseInt(e.target.value) || 0})}
              helperText="Number of solar cells per module"
            />
          </Grid>
        </Grid>
        
        {/* Calculation Preview */}
        <Box sx={{ mt: 3, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
          <Typography variant="subtitle2" gutterBottom>
            PV Capacity Preview:
          </Typography>
          <Typography variant="body2">
            Usable PV Area: {(buildingData.roof_area * pvParameters.coverage_factor).toFixed(2)} m²
          </Typography>
          <Typography variant="body2">
            Number of Modules: {Math.round((buildingData.roof_area * pvParameters.coverage_factor) / pvParameters.module_area)}
          </Typography>
          <Typography variant="body2">
            Installed Capacity: {((buildingData.roof_area * pvParameters.coverage_factor / pvParameters.module_area) * (pvParameters.module_power / 1000)).toFixed(2)} kW
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );

  const renderPVGISParametersStep = () => (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Settings />
          PVGIS Simulation Parameters
        </Typography>
        
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Location"
              value={pvgisParameters.location}
              onChange={(e) => setPvgisParameters({...pvgisParameters, location: e.target.value})}
              helperText="Location for solar simulation"
            />
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Tilt Angle (°)"
              type="number"
              value={pvgisParameters.tilt_angle}
              onChange={(e) => setPvgisParameters({...pvgisParameters, tilt_angle: parseFloat(e.target.value) || 0})}
              helperText="Panel tilt angle (0-90°)"
              inputProps={{ min: 0, max: 90 }}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Azimuth (°)"
              type="number"
              value={pvgisParameters.azimuth}
              onChange={(e) => setPvgisParameters({...pvgisParameters, azimuth: parseFloat(e.target.value) || 0})}
              helperText="Panel azimuth (0° = South, -180° to 180°)"
              inputProps={{ min: -180, max: 180 }}
            />
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="System Losses (%)"
              type="number"
              value={pvgisParameters.system_losses}
              onChange={(e) => setPvgisParameters({...pvgisParameters, system_losses: parseFloat(e.target.value) || 0})}
              helperText="Total system losses (0-50%)"
              inputProps={{ min: 0, max: 50 }}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth>
              <InputLabel>Mounting Position</InputLabel>
              <Select
                value={pvgisParameters.mounting_position}
                onChange={(e) => setPvgisParameters({...pvgisParameters, mounting_position: e.target.value})}
              >
                <MenuItem value="building_integrated">Building Integrated</MenuItem>
                <MenuItem value="free_standing">Free Standing</MenuItem>
                <MenuItem value="pitched_roof">Pitched Roof</MenuItem>
              </Select>
            </FormControl>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );

  const renderResultsStep = () => (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Calculate />
          Analysis Results
        </Typography>
        
        {result && (
          <Box>
            {/* PV Capacity Results */}
            <Accordion defaultExpanded>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Typography variant="h6">PV Capacity Analysis</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6} md={3}>
                    <Typography variant="body2" color="text.secondary">Roof Area</Typography>
                    <Typography variant="h6">{result.pv_capacity.roof_area_m2} m²</Typography>
                  </Grid>
                  <Grid item xs={12} sm={6} md={3}>
                    <Typography variant="body2" color="text.secondary">Usable PV Area</Typography>
                    <Typography variant="h6">{result.pv_capacity.usable_pv_area_m2.toFixed(2)} m²</Typography>
                  </Grid>
                  <Grid item xs={12} sm={6} md={3}>
                    <Typography variant="body2" color="text.secondary">Number of Modules</Typography>
                    <Typography variant="h6">{result.pv_capacity.num_modules}</Typography>
                  </Grid>
                  <Grid item xs={12} sm={6} md={3}>
                    <Typography variant="body2" color="text.secondary">Installed Capacity</Typography>
                    <Typography variant="h6">{result.pv_capacity.installed_capacity_kw.toFixed(2)} kW</Typography>
                  </Grid>
                </Grid>
              </AccordionDetails>
            </Accordion>

            {/* System Analysis Results */}
            {(result.system_config || result.estimated_analysis) && (
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Typography variant="h6">Energy Production Analysis</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  {result.system_config ? (
                    <Grid container spacing={2}>
                      <Grid item xs={12} sm={6} md={4}>
                        <Typography variant="body2" color="text.secondary">Annual Energy</Typography>
                        <Typography variant="h6">{result.system_config.energy_production?.annual_energy_kwh?.toFixed(0)} kWh/year</Typography>
                      </Grid>
                      <Grid item xs={12} sm={6} md={4}>
                        <Typography variant="body2" color="text.secondary">Energy per m²</Typography>
                        <Typography variant="h6">{result.system_config.energy_production?.energy_per_m2_kwh?.toFixed(1)} kWh/m²/year</Typography>
                      </Grid>
                      <Grid item xs={12} sm={6} md={4}>
                        <Typography variant="body2" color="text.secondary">Capacity Factor</Typography>
                        <Typography variant="h6">{(result.system_config.energy_production?.capacity_factor * 100)?.toFixed(1)}%</Typography>
                      </Grid>
                    </Grid>
                  ) : (
                    <Grid container spacing={2}>
                      <Grid item xs={12} sm={6} md={4}>
                        <Typography variant="body2" color="text.secondary">Annual Energy (Estimated)</Typography>
                        <Typography variant="h6">{result.estimated_analysis.annual_energy_kwh?.toFixed(0)} kWh/year</Typography>
                      </Grid>
                      <Grid item xs={12} sm={6} md={4}>
                        <Typography variant="body2" color="text.secondary">Energy per m² (Estimated)</Typography>
                        <Typography variant="h6">{result.estimated_analysis.energy_per_m2_kwh?.toFixed(1)} kWh/m²/year</Typography>
                      </Grid>
                      <Grid item xs={12} sm={6} md={4}>
                        <Typography variant="body2" color="text.secondary">Capacity Factor (Estimated)</Typography>
                        <Typography variant="h6">{(result.estimated_analysis.capacity_factor * 100)?.toFixed(1)}%</Typography>
                      </Grid>
                    </Grid>
                  )}
                </AccordionDetails>
              </Accordion>
            )}
          </Box>
        )}
      </CardContent>
    </Card>
  );

  const renderStepContent = (step) => {
    switch (step) {
      case 0:
        return renderBuildingDataStep();
      case 1:
        return renderPVParametersStep();
      case 2:
        return renderPVGISParametersStep();
      case 3:
        return renderResultsStep();
      default:
        return null;
    }
  };

  return (
    <Box sx={{ maxWidth: 800, mx: 'auto' }}>
      <Typography variant="h4" component="h1" gutterBottom sx={{ fontWeight: 'bold', mb: 4 }}>
        🔧 Custom Solar Analysis
      </Typography>
      
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
        {steps.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>

      {renderStepContent(activeStep)}

      <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 4 }}>
        <Button
          disabled={activeStep === 0}
          onClick={handleBack}
          sx={{ mr: 1 }}
        >
          Back
        </Button>
        
        <Box>
          {activeStep === steps.length - 1 ? (
            <Button onClick={handleReset}>
              New Analysis
            </Button>
          ) : activeStep === steps.length - 2 ? (
            <Button
              variant="contained"
              onClick={runAnalysis}
              disabled={loading || !buildingData.osm_id || !buildingData.roof_area}
              startIcon={loading ? <CircularProgress size={20} /> : <Calculate />}
            >
              {loading ? 'Running Analysis...' : 'Run Analysis'}
            </Button>
          ) : (
            <Button
              variant="contained"
              onClick={handleNext}
              disabled={activeStep === 0 && (!buildingData.osm_id || !buildingData.roof_area)}
            >
              Next
            </Button>
          )}
        </Box>
      </Box>
    </Box>
  );
};

export default CustomAnalysisForm;
