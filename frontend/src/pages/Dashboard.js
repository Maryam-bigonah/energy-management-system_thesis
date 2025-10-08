import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  CircularProgress,
  Alert,
  Chip,
  Button,
} from '@mui/material';
import {
  Business,
  SolarPower,
  TrendingUp,
  Map,
  People,
  EnergySavingsLeaf,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { apiService } from '../services/api';

const StatCard = ({ title, value, icon, color = 'primary', subtitle, onClick }) => (
  <Card 
    sx={{ 
      height: '100%', 
      cursor: onClick ? 'pointer' : 'default',
      transition: 'transform 0.2s, box-shadow 0.2s',
      '&:hover': onClick ? { 
        transform: 'translateY(-4px)', 
        boxShadow: 4 
      } : {}
    }}
    onClick={onClick}
  >
    <CardContent sx={{ textAlign: 'center', p: 3 }}>
      <Box sx={{ color: `${color}.main`, mb: 2 }}>
        {icon}
      </Box>
      <Typography variant="h4" component="div" sx={{ fontWeight: 'bold', mb: 1 }}>
        {value}
      </Typography>
      <Typography variant="h6" color="text.secondary" sx={{ mb: 1 }}>
        {title}
      </Typography>
      {subtitle && (
        <Typography variant="body2" color="text.secondary">
          {subtitle}
        </Typography>
      )}
    </CardContent>
  </Card>
);

const Dashboard = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState(null);
  const [buildings, setBuildings] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      setError(null);

      const [statsResponse, buildingsResponse] = await Promise.all([
        apiService.getStatistics(),
        apiService.getBuildings({ limit: 5 })
      ]);

      setStats(statsResponse.data);
      setBuildings(buildingsResponse.data.buildings);

    } catch (err) {
      console.error('Error loading dashboard data:', err);
      setError(err.response?.data?.detail || 'Failed to load dashboard data');
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
        🌞 Torino Solar Analysis Dashboard
      </Typography>

      {/* Statistics Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Total Buildings"
            value={stats?.total_buildings || 0}
            icon={<Business sx={{ fontSize: 40 }} />}
            color="primary"
            subtitle="Residential buildings analyzed"
            onClick={() => navigate('/buildings')}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Solar Potential"
            value={`${formatNumber(stats?.total_solar_potential_kwh)} kWh`}
            icon={<SolarPower sx={{ fontSize: 40 }} />}
            color="secondary"
            subtitle="Annual energy generation"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Total Population"
            value={formatNumber(stats?.total_population)}
            icon={<People sx={{ fontSize: 40 }} />}
            color="success"
            subtitle="Residents covered"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Avg Energy/Person"
            value={`${Math.round(stats?.average_energy_per_person_kwh || 0)} kWh`}
            icon={<EnergySavingsLeaf sx={{ fontSize: 40 }} />}
            color="warning"
            subtitle="Per year"
          />
        </Grid>
      </Grid>

      {/* Building Types Distribution */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h5" gutterBottom sx={{ fontWeight: 'bold' }}>
                Building Types
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 2 }}>
                {Object.entries(stats?.building_types || {}).map(([type, count]) => (
                  <Chip
                    key={type}
                    label={`${type}: ${count}`}
                    variant="outlined"
                    color="primary"
                  />
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h5" gutterBottom sx={{ fontWeight: 'bold' }}>
                Roof Area Distribution
              </Typography>
              <Box sx={{ mt: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Min: {Math.round(stats?.roof_area_distribution?.min || 0)} m²
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Max: {Math.round(stats?.roof_area_distribution?.max || 0)} m²
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Average: {Math.round(stats?.roof_area_distribution?.mean || 0)} m²
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Median: {Math.round(stats?.roof_area_distribution?.median || 0)} m²
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Top Buildings */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Typography variant="h5" sx={{ fontWeight: 'bold' }}>
              Top Buildings by Solar Potential
            </Typography>
            <Button 
              variant="outlined" 
              onClick={() => navigate('/buildings')}
              startIcon={<Business />}
            >
              View All Buildings
            </Button>
          </Box>
          
          <Grid container spacing={2}>
            {buildings.map((building, index) => (
              <Grid item xs={12} sm={6} md={4} key={building.osm_id}>
                <Card 
                  variant="outlined"
                  sx={{ 
                    cursor: 'pointer',
                    transition: 'all 0.2s',
                    '&:hover': { 
                      backgroundColor: 'action.hover',
                      transform: 'translateY(-2px)'
                    }
                  }}
                  onClick={() => navigate(`/buildings/${building.osm_id}`)}
                >
                  <CardContent sx={{ p: 2 }}>
                    <Typography variant="h6" noWrap>
                      {building.osm_id}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      {building.address || 'No address'}
                    </Typography>
                    <Box display="flex" justifyContent="space-between" alignItems="center">
                      <Typography variant="body2">
                        Roof: {Math.round(building.roof_area)} m²
                      </Typography>
                      <Chip 
                        label={`#${index + 1}`} 
                        size="small" 
                        color="primary" 
                      />
                    </Box>
                    <Typography variant="body2" color="primary" sx={{ mt: 1, fontWeight: 'bold' }}>
                      {Math.round(building.solar_potential)} kWh/year
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </CardContent>
      </Card>

      {/* Quick Actions */}
      <Grid container spacing={3}>
        <Grid item xs={12} sm={6} md={3}>
          <Card 
            sx={{ 
              cursor: 'pointer',
              transition: 'all 0.2s',
              '&:hover': { 
                backgroundColor: 'primary.light',
                color: 'white',
                transform: 'translateY(-4px)'
              }
            }}
            onClick={() => navigate('/map')}
          >
            <CardContent sx={{ textAlign: 'center', p: 3 }}>
              <Map sx={{ fontSize: 48, mb: 2 }} />
              <Typography variant="h6" gutterBottom>
                Interactive Map
              </Typography>
              <Typography variant="body2">
                Explore buildings on map
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card 
            sx={{ 
              cursor: 'pointer',
              transition: 'all 0.2s',
              '&:hover': { 
                backgroundColor: 'secondary.light',
                color: 'white',
                transform: 'translateY(-4px)'
              }
            }}
            onClick={() => navigate('/statistics')}
          >
            <CardContent sx={{ textAlign: 'center', p: 3 }}>
              <TrendingUp sx={{ fontSize: 48, mb: 2 }} />
              <Typography variant="h6" gutterBottom>
                Detailed Analytics
              </Typography>
              <Typography variant="body2">
                View comprehensive statistics
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;
