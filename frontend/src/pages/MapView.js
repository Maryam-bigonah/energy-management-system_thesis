import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  CircularProgress,
  Alert,
  Chip,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import { MapContainer, TileLayer, Marker, Popup, Circle } from 'react-leaflet';
import { Icon } from 'leaflet';
import { apiService } from '../services/api';

// Fix for default markers
delete Icon.Default.prototype._getIconUrl;
Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

const MapView = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [buildings, setBuildings] = useState([]);
  const [filteredBuildings, setFilteredBuildings] = useState([]);
  const [solarPotentialRange, setSolarPotentialRange] = useState([0, 1000000]);
  const [buildingTypeFilter, setBuildingTypeFilter] = useState('all');
  const [maxSolarPotential, setMaxSolarPotential] = useState(0);

  useEffect(() => {
    loadBuildingsData();
  }, []);

  useEffect(() => {
    applyFilters();
  }, [buildings, solarPotentialRange, buildingTypeFilter]);

  const loadBuildingsData = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await apiService.getBuildings();
      const buildingsData = response.data.buildings;
      
      setBuildings(buildingsData);
      setFilteredBuildings(buildingsData);
      
      // Calculate max solar potential for slider
      const maxPotential = Math.max(...buildingsData.map(b => b.solar_potential || 0));
      setMaxSolarPotential(maxPotential);
      setSolarPotentialRange([0, maxPotential]);

    } catch (err) {
      console.error('Error loading buildings data:', err);
      setError(err.response?.data?.detail || 'Failed to load buildings data');
    } finally {
      setLoading(false);
    }
  };

  const applyFilters = () => {
    let filtered = buildings.filter(building => {
      const solarPotential = building.solar_potential || 0;
      const solarInRange = solarPotential >= solarPotentialRange[0] && solarPotential <= solarPotentialRange[1];
      const typeMatch = buildingTypeFilter === 'all' || building.building_type === buildingTypeFilter;
      
      return solarInRange && typeMatch;
    });
    
    setFilteredBuildings(filtered);
  };

  const getMarkerColor = (solarPotential) => {
    const maxPotential = Math.max(...buildings.map(b => b.solar_potential || 0));
    const ratio = solarPotential / maxPotential;
    
    if (ratio > 0.8) return '#e53935'; // Red for highest
    if (ratio > 0.6) return '#ff9800'; // Orange for high
    if (ratio > 0.4) return '#ffeb3b'; // Yellow for medium
    if (ratio > 0.2) return '#4caf50'; // Green for low-medium
    return '#2196f3'; // Blue for lowest
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

  const center = [45.0447, 7.637]; // Torino coordinates

  return (
    <Box>
      <Typography variant="h3" component="h1" gutterBottom sx={{ fontWeight: 'bold', mb: 4 }}>
        🗺️ Interactive Map View
      </Typography>

      {/* Filters */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Filters
          </Typography>
          
          <Box sx={{ display: 'flex', gap: 3, alignItems: 'center', flexWrap: 'wrap' }}>
            <Box sx={{ minWidth: 300 }}>
              <Typography variant="body2" gutterBottom>
                Solar Potential Range: {formatNumber(solarPotentialRange[0])} - {formatNumber(solarPotentialRange[1])} kWh/year
              </Typography>
              <Slider
                value={solarPotentialRange}
                onChange={(event, newValue) => setSolarPotentialRange(newValue)}
                valueLabelDisplay="auto"
                min={0}
                max={maxSolarPotential}
                valueLabelFormat={(value) => formatNumber(value)}
              />
            </Box>
            
            <FormControl sx={{ minWidth: 200 }}>
              <InputLabel>Building Type</InputLabel>
              <Select
                value={buildingTypeFilter}
                label="Building Type"
                onChange={(e) => setBuildingTypeFilter(e.target.value)}
              >
                <MenuItem value="all">All Types</MenuItem>
                <MenuItem value="apartments">Apartments</MenuItem>
                <MenuItem value="house">House</MenuItem>
                <MenuItem value="commercial">Commercial</MenuItem>
                <MenuItem value="industrial">Industrial</MenuItem>
              </Select>
            </FormControl>
            
            <Chip 
              label={`${filteredBuildings.length} buildings shown`} 
              color="primary" 
              variant="outlined" 
            />
          </Box>
        </CardContent>
      </Card>

      {/* Map */}
      <Card sx={{ height: '600px' }}>
        <CardContent sx={{ height: '100%', p: 0 }}>
          <Box sx={{ height: '100%', borderRadius: 1, overflow: 'hidden' }}>
            <MapContainer
              center={center}
              zoom={13}
              style={{ height: '100%', width: '100%' }}
            >
              <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              />
              
              {filteredBuildings.map((building) => (
                <Marker
                  key={building.osm_id}
                  position={[building.latitude, building.longitude]}
                >
                  <Popup>
                    <Box sx={{ minWidth: 200 }}>
                      <Typography variant="h6" gutterBottom>
                        {building.osm_id}
                      </Typography>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        {building.address || 'No address'}
                      </Typography>
                      
                      <Box sx={{ mb: 2 }}>
                        <Chip label={building.building_type} size="small" color="primary" />
                      </Box>
                      
                      <Typography variant="body2" gutterBottom>
                        <strong>Height:</strong> {building.height}m
                      </Typography>
                      <Typography variant="body2" gutterBottom>
                        <strong>Floors:</strong> {building.floors}
                      </Typography>
                      <Typography variant="body2" gutterBottom>
                        <strong>Roof Area:</strong> {Math.round(building.roof_area)} m²
                      </Typography>
                      <Typography variant="body2" gutterBottom>
                        <strong>Population:</strong> {building.estimated_population}
                      </Typography>
                      
                      <Box sx={{ mt: 2, p: 2, bgcolor: 'primary.light', borderRadius: 1 }}>
                        <Typography variant="body2" color="primary.contrastText" gutterBottom>
                          <strong>Solar Potential:</strong>
                        </Typography>
                        <Typography variant="h6" color="primary.contrastText">
                          {formatNumber(building.solar_potential)} kWh/year
                        </Typography>
                      </Box>
                    </Box>
                  </Popup>
                  
                  <Circle
                    center={[building.latitude, building.longitude]}
                    radius={50}
                    pathOptions={{
                      fillColor: getMarkerColor(building.solar_potential),
                      color: getMarkerColor(building.solar_potential),
                      fillOpacity: 0.3,
                      weight: 2,
                    }}
                  />
                </Marker>
              ))}
            </MapContainer>
          </Box>
        </CardContent>
      </Card>

      {/* Legend */}
      <Card sx={{ mt: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Solar Potential Legend
          </Typography>
          <Box sx={{ display: 'flex', gap: 3, alignItems: 'center', flexWrap: 'wrap' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Box sx={{ width: 20, height: 20, bgcolor: '#e53935', borderRadius: '50%' }} />
              <Typography variant="body2">Highest (>80%)</Typography>
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Box sx={{ width: 20, height: 20, bgcolor: '#ff9800', borderRadius: '50%' }} />
              <Typography variant="body2">High (60-80%)</Typography>
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Box sx={{ width: 20, height: 20, bgcolor: '#ffeb3b', borderRadius: '50%' }} />
              <Typography variant="body2">Medium (40-60%)</Typography>
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Box sx={{ width: 20, height: 20, bgcolor: '#4caf50', borderRadius: '50%' }} />
              <Typography variant="body2">Low-Medium (20-40%)</Typography>
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Box sx={{ width: 20, height: 20, bgcolor: '#2196f3', borderRadius: '50%' }} />
              <Typography variant="body2">Lowest (<20%)</Typography>
            </Box>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

export default MapView;
