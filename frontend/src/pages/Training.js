import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { trainModel, getMetrics, loadSampleData, healthCheck } from '../services/api';
import './Training.css';

function Training() {
  const [training, setTraining] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [trainingHistory, setTrainingHistory] = useState(null);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [epochs, setEpochs] = useState(50);
  const [batchSize, setBatchSize] = useState(32);
  const [validationSplit, setValidationSplit] = useState(0.2);
  const [dataLoaded, setDataLoaded] = useState(false);

  useEffect(() => {
    checkDataStatus();
  }, []);

  const checkDataStatus = async () => {
    try {
      const response = await healthCheck();
      setDataLoaded(true);
    } catch (err) {
      setDataLoaded(false);
    }
  };

  const handleLoadSampleData = async () => {
    try {
      setError(null);
      setSuccess(null);
      const response = await loadSampleData();
      if (response.data.success) {
        setSuccess('Sample data loaded successfully!');
        setDataLoaded(true);
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to load sample data');
    }
  };

  const handleTrain = async () => {
    try {
      setTraining(true);
      setError(null);
      setSuccess(null);
      setMetrics(null);
      setTrainingHistory(null);

      const response = await trainModel(epochs, batchSize, validationSplit);

      if (response.data.success) {
        setSuccess('Model trained successfully!');
        setTrainingHistory(response.data.history);
        setMetrics(response.data.metrics);

        // Also fetch latest metrics
        try {
          const metricsResponse = await getMetrics();
          if (metricsResponse.data.success) {
            setMetrics(metricsResponse.data.metrics);
          }
        } catch (err) {
          console.error('Failed to fetch metrics:', err);
        }
      } else {
        setError(response.data.error || 'Training failed');
      }
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Training failed. Make sure data is loaded.');
    } finally {
      setTraining(false);
    }
  };

  const prepareHistoryData = () => {
    if (!trainingHistory) return [];

    const data = [];
    const maxLen = Math.max(
      trainingHistory.loss.length,
      trainingHistory.val_loss.length
    );

    for (let i = 0; i < maxLen; i++) {
      data.push({
        epoch: i + 1,
        loss: trainingHistory.loss[i] || null,
        val_loss: trainingHistory.val_loss[i] || null,
        mae: trainingHistory.mae[i] || null,
        val_mae: trainingHistory.val_mae[i] || null,
      });
    }
    return data;
  };

  return (
    <div>
      <div className="container">
        <h2>Model Training</h2>

        {error && <div className="error">{error}</div>}
        {success && <div className="success">{success}</div>}

        {!dataLoaded && (
          <div className="data-warning">
            <p>No data loaded. Please load sample data first.</p>
            <button onClick={handleLoadSampleData} className="btn btn-primary">
              Load Sample Data
            </button>
          </div>
        )}

        <div className="training-config">
          <h3>Training Configuration</h3>
          <div className="form-group">
            <label>Epochs</label>
            <input
              type="number"
              min="1"
              max="200"
              value={epochs}
              onChange={(e) => setEpochs(parseInt(e.target.value))}
            />
          </div>
          <div className="form-group">
            <label>Batch Size</label>
            <input
              type="number"
              min="8"
              max="128"
              value={batchSize}
              onChange={(e) => setBatchSize(parseInt(e.target.value))}
            />
          </div>
          <div className="form-group">
            <label>Validation Split</label>
            <input
              type="number"
              min="0.1"
              max="0.5"
              step="0.1"
              value={validationSplit}
              onChange={(e) => setValidationSplit(parseFloat(e.target.value))}
            />
          </div>
          <button
            onClick={handleTrain}
            disabled={training || !dataLoaded}
            className="btn btn-success"
            style={{ marginTop: '16px' }}
          >
            {training ? 'Training...' : 'Train Model'}
          </button>
        </div>

        {trainingHistory && (
          <>
            <div className="container">
              <h3>Training History</h3>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={prepareHistoryData()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
                  <YAxis label={{ value: 'Loss', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="loss"
                    stroke="#8884d8"
                    name="Training Loss"
                    strokeWidth={2}
                  />
                  <Line
                    type="monotone"
                    dataKey="val_loss"
                    stroke="#82ca9d"
                    name="Validation Loss"
                    strokeWidth={2}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="container">
              <h3>Training History - MAE</h3>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={prepareHistoryData()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
                  <YAxis label={{ value: 'MAE', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="mae"
                    stroke="#8884d8"
                    name="Training MAE"
                    strokeWidth={2}
                  />
                  <Line
                    type="monotone"
                    dataKey="val_mae"
                    stroke="#82ca9d"
                    name="Validation MAE"
                    strokeWidth={2}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </>
        )}

        {metrics && (
          <div className="container">
            <h3>Model Performance Metrics</h3>
            <div className="metrics-grid">
              <div className="metric-card">
                <h3>Load - MAE</h3>
                <div className="value">{metrics.load?.MAE?.toFixed(4) || 'N/A'}</div>
                <div className="label">Mean Absolute Error</div>
              </div>
              <div className="metric-card">
                <h3>Load - RMSE</h3>
                <div className="value">{metrics.load?.RMSE?.toFixed(4) || 'N/A'}</div>
                <div className="label">Root Mean Squared Error</div>
              </div>
              <div className="metric-card">
                <h3>Load - R²</h3>
                <div className="value">{metrics.load?.R2?.toFixed(4) || 'N/A'}</div>
                <div className="label">Coefficient of Determination</div>
              </div>
              <div className="metric-card">
                <h3>Load - MAPE</h3>
                <div className="value">{metrics.load?.MAPE?.toFixed(2) || 'N/A'}%</div>
                <div className="label">Mean Absolute Percentage Error</div>
              </div>
              <div className="metric-card">
                <h3>PV - MAE</h3>
                <div className="value">{metrics.pv?.MAE?.toFixed(4) || 'N/A'}</div>
                <div className="label">Mean Absolute Error</div>
              </div>
              <div className="metric-card">
                <h3>PV - RMSE</h3>
                <div className="value">{metrics.pv?.RMSE?.toFixed(4) || 'N/A'}</div>
                <div className="label">Root Mean Squared Error</div>
              </div>
              <div className="metric-card">
                <h3>PV - R²</h3>
                <div className="value">{metrics.pv?.R2?.toFixed(4) || 'N/A'}</div>
                <div className="label">Coefficient of Determination</div>
              </div>
              <div className="metric-card">
                <h3>PV - MAPE</h3>
                <div className="value">{metrics.pv?.MAPE?.toFixed(2) || 'N/A'}%</div>
                <div className="label">Mean Absolute Percentage Error</div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default Training;


