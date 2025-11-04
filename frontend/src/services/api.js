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

// Master dataset endpoints
export const getMasterDataset = (startDate, endDate, limit = 1000) =>
  api.get('/data/master-dataset', {
    params: { start_date: startDate, end_date: endDate, limit },
  });

// Battery endpoints
export const simulateBattery = (capacityKwh = 20.0, allocationMethod = 'energy_share', initialSoc = 0.5, limit = 1000) =>
  api.post('/battery/simulate', {
    capacity_kwh: capacityKwh,
    allocation_method: allocationMethod,
    initial_soc: initialSoc,
    limit,
  });

export const getBatteryData = (limit = 1000) =>
  api.get('/battery/data', { params: { limit } });

// Economic endpoints
export const analyzeEconomic = (tariffsCsv, fitCsv, limit = 1000) =>
  api.post('/economic/analyze', {
    tariffs_csv: tariffsCsv,
    fit_csv: fitCsv,
    limit,
  });

// Data summary
export const getDataSummary = () => api.get('/data/summary');

// Load master dataset from CSV
export const loadMasterCSV = (csvPath) =>
  api.post('/data/load-master-csv', { csv_path: csvPath });

// Family consumption
export const getFamilyConsumption = (startDate, endDate, limit = 1000) =>
  api.get('/data/family-consumption', {
    params: { start_date: startDate, end_date: endDate, limit },
  });

// Storage energy (PV and battery)
export const getStorageEnergy = (limit = 1000) =>
  api.get('/data/storage-energy', { params: { limit } });

// All data summary
export const getAllDataSummary = () => api.get('/data/all-data-summary');

// Database info - shows all available data
export const getDatabaseInfo = () => api.get('/data/database-info');

// New LSTM Pipeline endpoints
export const trainLSTMPipeline = (epochs = 30, batchSize = 32, lstmUnits = 64, learningRate = 1e-3, scaleTargets = true) =>
  api.post('/lstm/train', {
    epochs,
    batch_size: batchSize,
    lstm_units: lstmUnits,
    learning_rate: learningRate,
    scale_targets: scaleTargets,
    window: 24
  });

export const getLSTMEvaluation = () => api.get('/lstm/evaluation');

export const getLSTMTrainingHistory = () => api.get('/lstm/training-history');

export const getLSTMStatus = () => api.get('/lstm/status');

export default api;


