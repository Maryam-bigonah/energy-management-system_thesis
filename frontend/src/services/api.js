import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    console.log(`Making API request to: ${config.url}`);
    return config;
  },
  (error) => {
    console.error('Request error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    console.log(`API response from: ${response.config.url}`, response.status);
    return response;
  },
  (error) => {
    console.error('Response error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// API endpoints
export const apiService = {
  // Buildings
  getBuildings: (params = {}) => api.get('/buildings', { params }),
  getBuilding: (buildingId) => api.get(`/buildings/${buildingId}`),
  getSolarAnalysis: (buildingId, technology = 'mono_crystalline', roofArea = null) => {
    const params = { technology };
    if (roofArea) params.roof_area = roofArea;
    return api.get(`/buildings/${buildingId}/solar-analysis`, { params });
  },
  getTechnologyComparison: (buildingId) => 
    api.get(`/buildings/${buildingId}/technology-comparison`),

  // Statistics
  getStatistics: () => api.get('/statistics'),

  // Plot data
  getRoofAreaDistribution: () => api.get('/plot-data/roof-area-distribution'),
  getSolarPotentialData: () => api.get('/plot-data/solar-potential'),

  // Health check
  healthCheck: () => api.get('/'),
};

export default api;
