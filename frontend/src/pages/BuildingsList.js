import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  CircularProgress,
  Alert,
  Chip,
  Button,
  Pagination,
} from '@mui/material';
import { Business, Search, FilterList } from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { apiService } from '../services/api';

const BuildingsList = () => {
  const [buildings, setBuildings] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState({
    building_type: '',
    min_roof_area: '',
    max_roof_area: '',
  });
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);

  useEffect(() => {
    loadBuildings();
  }, [page, filters]);

  const loadBuildings = async () => {
    try {
      setLoading(true);
      setError(null);

      const params = {
        ...filters,
        limit: 12,
        offset: (page - 1) * 12,
      };

      // Remove empty filters
      Object.keys(params).forEach(key => {
        if (params[key] === '' || params[key] === null) {
          delete params[key];
        }
      });

      const response = await apiService.getBuildings(params);
      setBuildings(response.data.buildings);
      setTotalPages(Math.ceil(response.data.total / 12));

    } catch (err) {
      console.error('Error loading buildings:', err);
      setError(err.response?.data?.detail || 'Failed to load buildings');
    } finally {
      setLoading(false);
    }
  };

  const handleFilterChange = (field, value) => {
    setFilters(prev => ({
      ...prev,
      [field]: value,
    }));
    setPage(1); // Reset to first page when filters change
  };

  const clearFilters = () => {
    setFilters({
      building_type: '',
      min_roof_area: '',
      max_roof_area: '',
    });
    setPage(1);
  };

  const formatNumber = (num) => {
    return num?.toLocaleString() || '0';
  };

  if (loading && page === 1) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress size={60} />
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h3" component="h1" gutterBottom sx={{ fontWeight: 'bold', mb: 4 }}>
        🏢 Buildings Directory
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Filters */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <FilterList />
            Filters
          </Typography>
          
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} sm={6} md={3}>
              <FormControl fullWidth>
                <InputLabel>Building Type</InputLabel>
                <Select
                  value={filters.building_type}
                  label="Building Type"
                  onChange={(e) => handleFilterChange('building_type', e.target.value)}
                >
                  <MenuItem value="">All Types</MenuItem>
                  <MenuItem value="apartments">Apartments</MenuItem>
                  <MenuItem value="house">House</MenuItem>
                  <MenuItem value="commercial">Commercial</MenuItem>
                  <MenuItem value="industrial">Industrial</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <TextField
                fullWidth
                label="Min Roof Area (m²)"
                type="number"
                value={filters.min_roof_area}
                onChange={(e) => handleFilterChange('min_roof_area', e.target.value)}
              />
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <TextField
                fullWidth
                label="Max Roof Area (m²)"
                type="number"
                value={filters.max_roof_area}
                onChange={(e) => handleFilterChange('max_roof_area', e.target.value)}
              />
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <Button
                variant="outlined"
                onClick={clearFilters}
                sx={{ height: '56px' }}
              >
                Clear Filters
              </Button>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Buildings Grid */}
      <Grid container spacing={3}>
        {buildings.map((building) => (
          <Grid item xs={12} sm={6} md={4} key={building.osm_id}>
            <Card
              sx={{
                height: '100%',
                cursor: 'pointer',
                transition: 'all 0.2s',
                '&:hover': {
                  transform: 'translateY(-4px)',
                  boxShadow: 4,
                },
              }}
              onClick={() => navigate(`/buildings/${building.osm_id}`)}
            >
              <CardContent sx={{ p: 3 }}>
                <Box display="flex" alignItems="center" mb={2}>
                  <Business sx={{ mr: 1, color: 'primary.main' }} />
                  <Typography variant="h6" noWrap sx={{ fontWeight: 'bold' }}>
                    {building.osm_id}
                  </Typography>
                </Box>

                <Typography variant="body2" color="text.secondary" gutterBottom>
                  {building.address || 'No address available'}
                </Typography>

                <Box sx={{ mb: 2 }}>
                  <Chip
                    label={building.building_type || 'Unknown'}
                    size="small"
                    color="primary"
                    variant="outlined"
                  />
                </Box>

                <Grid container spacing={1} sx={{ mb: 2 }}>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      Height: {building.height}m
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      Floors: {building.floors}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      Roof: {Math.round(building.roof_area)} m²
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      Population: {building.estimated_population}
                    </Typography>
                  </Grid>
                </Grid>

                <Box sx={{ p: 2, bgcolor: 'primary.light', borderRadius: 1, textAlign: 'center' }}>
                  <Typography variant="body2" color="primary.contrastText" gutterBottom>
                    Solar Potential
                  </Typography>
                  <Typography variant="h6" color="primary.contrastText" sx={{ fontWeight: 'bold' }}>
                    {formatNumber(building.solar_potential)} kWh/year
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Pagination */}
      {totalPages > 1 && (
        <Box display="flex" justifyContent="center" mt={4}>
          <Pagination
            count={totalPages}
            page={page}
            onChange={(event, value) => setPage(value)}
            color="primary"
            size="large"
          />
        </Box>
      )}

      {buildings.length === 0 && !loading && (
        <Box textAlign="center" py={8}>
          <Business sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
          <Typography variant="h6" color="text.secondary">
            No buildings found
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Try adjusting your filters or check back later
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default BuildingsList;
