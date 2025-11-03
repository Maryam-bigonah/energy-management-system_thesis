import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { getPredictions, forecastNextHour, getMetrics } from '../services/api';
import './Forecasts.css';

function Forecasts() {
  const [forecastData, setForecastData] = useState(null);
  const [nextHour, setNextHour] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hours, setHours] = useState(168); // Default: 1 week
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    loadMetrics();
  }, []);

  const loadMetrics = async () => {
    try {
      const response = await getMetrics();
      if (response.data.success) {
        setMetrics(response.data.metrics);
      }
    } catch (err) {
      // Metrics not available, that's okay
    }
  };

  const loadForecasts = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await getPredictions(hours);

      if (response.data.success) {
        const chartData = response.data.data.timestamps.map((ts, idx) => ({
          timestamp: new Date(ts).toLocaleString(),
          load_pred: response.data.data.predictions.load[idx],
          load_actual: response.data.data.actual.load[idx],
          pv_pred: response.data.data.predictions.pv[idx],
          pv_actual: response.data.data.actual.pv[idx],
        }));
        setForecastData(chartData);
      } else {
        setError(response.data.error || 'Failed to load forecasts');
      }
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Failed to load forecasts. Make sure model is trained.');
    } finally {
      setLoading(false);
    }
  };

  const loadNextHour = async () => {
    try {
      setError(null);
      const response = await forecastNextHour();
      if (response.data.success) {
        setNextHour(response.data.forecast);
      }
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Failed to get next hour forecast');
    }
  };

  return (
    <div>
      <div className="container">
        <h2>Forecasts</h2>

        {error && <div className="error">{error}</div>}

        <div className="forecast-controls">
          <div className="form-group" style={{ maxWidth: '200px' }}>
            <label>Hours to Forecast</label>
            <input
              type="number"
              min="24"
              max="720"
              value={hours}
              onChange={(e) => setHours(parseInt(e.target.value))}
            />
          </div>
          <button onClick={loadForecasts} disabled={loading} className="btn btn-primary">
            {loading ? 'Loading...' : 'Load Forecasts'}
          </button>
          <button onClick={loadNextHour} className="btn btn-secondary">
            Next Hour Forecast
          </button>
        </div>

        {nextHour && (
          <div className="next-hour-card">
            <h3>Next Hour Forecast</h3>
            <div className="forecast-values">
              <div className="forecast-item">
                <span className="forecast-label">Timestamp:</span>
                <span className="forecast-value">{new Date(nextHour.timestamp).toLocaleString()}</span>
              </div>
              <div className="forecast-item">
                <span className="forecast-label">Load:</span>
                <span className="forecast-value">{nextHour.load_pred.toFixed(2)} kW</span>
              </div>
              <div className="forecast-item">
                <span className="forecast-label">PV:</span>
                <span className="forecast-value">{nextHour.pv_pred.toFixed(2)} kW</span>
              </div>
            </div>
          </div>
        )}

        {metrics && (
          <div className="metrics-summary">
            <h3>Current Model Performance</h3>
            <div className="metrics-grid">
              <div className="metric-card">
                <h3>Load - R²</h3>
                <div className="value">{metrics.load?.R2?.toFixed(4) || 'N/A'}</div>
              </div>
              <div className="metric-card">
                <h3>PV - R²</h3>
                <div className="value">{metrics.pv?.R2?.toFixed(4) || 'N/A'}</div>
              </div>
            </div>
          </div>
        )}

        {forecastData && (
          <>
            <div className="container">
              <h3>Load Forecast vs Actual</h3>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={forecastData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="timestamp"
                    angle={-45}
                    textAnchor="end"
                    height={80}
                    tick={{ fontSize: 10 }}
                    interval="preserveStartEnd"
                  />
                  <YAxis label={{ value: 'Load (kW)', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="load_actual"
                    stroke="#8884d8"
                    name="Actual Load"
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="load_pred"
                    stroke="#ff7300"
                    name="Predicted Load"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="container">
              <h3>PV Forecast vs Actual</h3>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={forecastData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="timestamp"
                    angle={-45}
                    textAnchor="end"
                    height={80}
                    tick={{ fontSize: 10 }}
                    interval="preserveStartEnd"
                  />
                  <YAxis label={{ value: 'PV (kW)', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="pv_actual"
                    stroke="#82ca9d"
                    name="Actual PV"
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="pv_pred"
                    stroke="#ff7300"
                    name="Predicted PV"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="forecast-stats">
              <h3>Forecast Statistics</h3>
              <div className="stats-grid">
                <div className="stat-item">
                  <strong>Load MAE:</strong>{' '}
                  {((forecastData.reduce((sum, d) => sum + Math.abs(d.load_actual - d.load_pred), 0)) / forecastData.length).toFixed(4)} kW
                </div>
                <div className="stat-item">
                  <strong>PV MAE:</strong>{' '}
                  {((forecastData.reduce((sum, d) => sum + Math.abs(d.pv_actual - d.pv_pred), 0)) / forecastData.length).toFixed(4)} kW
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default Forecasts;


