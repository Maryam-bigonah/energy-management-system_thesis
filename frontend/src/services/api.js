import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const healthCheck = () => api.get('/health');

export const loadSampleData = (startDate = '2023-01-01', endDate = '2023-12-31') =>
  api.post('/data/load-sample', { start_date: startDate, end_date: endDate });

export const uploadData = (pvFile, loadFile) => {
  const formData = new FormData();
  formData.append('pv_file', pvFile);
  formData.append('load_file', loadFile);
  return api.post('/data/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};

export const getHistoricalData = (startDate, endDate, limit = 1000) =>
  api.get('/data/historical', {
    params: { start_date: startDate, end_date: endDate, limit },
  });

export const trainModel = (epochs = 50, batchSize = 32, validationSplit = 0.2) =>
  api.post('/model/train', {
    epochs,
    batch_size: batchSize,
    validation_split: validationSplit,
  });

export const getPredictions = (hours = 168) =>
  api.post('/model/predict', { hours });

export const getMetrics = () => api.get('/model/metrics');

export const forecastNextHour = () => api.get('/model/forecast-next');

export default api;


