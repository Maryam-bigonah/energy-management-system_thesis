import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  CircularProgress,
  Alert,
} from '@mui/material';
import { apiService } from '../services/api';

// Placeholder chart components
const RoofAreaChart = ({ data }) => (
  <Box sx={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
    <Typography>Roof Area Distribution Chart</Typography>
  </Box>
);

const BuildingTypeChart = ({ data }) => (
  <Box sx={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
    <Typography>Building Type Distribution Chart</Typography>
  </Box>
);

const SolarPotentialChart = ({ data }) => (
  <Box sx={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
    <Typography>Solar Potential Chart</Typography>
  </Box>
);

const Statistics = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [statistics, setStatistics] = useState(null);
  const [roofAreaData, setRoofAreaData] = useState(null);
  const [solarPotentialData, setSolarPotentialData] = useState(null);

  useEffect(() => {
    loadStatistics();
  }, []);

  const loadStatistics = async () => {
    try {
      setLoading(true);
      setError(null);

      const [statsResponse, roofAreaResponse, solarPotentialResponse] = await Promise.all([
        apiService.getStatistics(),
        apiService.getRoofAreaDistribution(),
        apiService.getSolarPotentialData(),
      ]);

      setStatistics(statsResponse.data);
      setRoofAreaData(roofAreaResponse.data);
      setSolarPotentialData(solarPotentialResponse.data);

    } catch (err) {
      console.error('Error loading statistics:', err);
      setError(err.response?.data?.detail || 'Failed to load statistics');
    } finally {
      setLoading(false);
    }
  };

  const formatNumber = (num) => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
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

  return (
    <Box>
      <Typography variant="h3" component="h1" gutterBottom sx={{ fontWeight: 'bold', mb: 4 }}>
        📊 Statistics & Analytics
      </Typography>

      {/* Summary Statistics */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" sx={{ fontWeight: 'bold', color: 'primary.main' }}>
                {statistics?.total_buildings || 0}
              </Typography>
              <Typography variant="h6" color="text.secondary">
                Total Buildings
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" sx={{ fontWeight: 'bold', color: 'secondary.main' }}>
                {formatNumber(statistics?.total_solar_potential_kwh)}
              </Typography>
              <Typography variant="h6" color="text.secondary">
                Total Solar Potential (kWh)
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" sx={{ fontWeight: 'bold', color: 'success.main' }}>
                {formatNumber(statistics?.total_population)}
              </Typography>
              <Typography variant="h6" color="text.secondary">
                Total Population
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" sx={{ fontWeight: 'bold', color: 'warning.main' }}>
                {Math.round(statistics?.average_energy_per_person_kwh || 0)}
              </Typography>
              <Typography variant="h6" color="text.secondary">
                Avg Energy/Person (kWh)
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Charts */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Roof Area Distribution
              </Typography>
              <RoofAreaChart data={roofAreaData} />
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Building Type Distribution
              </Typography>
              <BuildingTypeChart data={statistics?.building_types} />
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Solar Potential Analysis */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Solar Potential Analysis
          </Typography>
          <SolarPotentialChart data={solarPotentialData} />
        </CardContent>
      </Card>

      {/* Detailed Statistics */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Roof Area Statistics
              </Typography>
              <Box sx={{ mt: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Minimum: {Math.round(statistics?.roof_area_distribution?.min || 0)} m²
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Maximum: {Math.round(statistics?.roof_area_distribution?.max || 0)} m²
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Average: {Math.round(statistics?.roof_area_distribution?.mean || 0)} m²
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Median: {Math.round(statistics?.roof_area_distribution?.median || 0)} m²
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Building Type Breakdown
              </Typography>
              <Box sx={{ mt: 2 }}>
                {Object.entries(statistics?.building_types || {}).map(([type, count]) => (
                  <Box key={type} sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2" color="text.secondary">
                      {type}:
                    </Typography>
                    <Typography variant="body2" fontWeight="bold">
                      {count} buildings
                    </Typography>
                  </Box>
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Statistics;
