import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { getHistoricalData, healthCheck } from '../services/api';
import './Dashboard.css';

function Dashboard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [healthStatus, setHealthStatus] = useState(null);

  useEffect(() => {
    // Check API health
    healthCheck()
      .then(response => setHealthStatus(response.data))
      .catch(err => {
        setError('Cannot connect to backend API. Make sure the Flask server is running on port 5000.');
        setLoading(false);
      });

    // Load historical data
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await getHistoricalData();
      
      if (response.data.success) {
        const chartData = response.data.data.timestamps.map((ts, idx) => ({
          timestamp: new Date(ts).toLocaleString(),
          load: response.data.data.load[idx],
          pv: response.data.data.pv[idx],
        }));
        setData(chartData);
      } else {
        setError(response.data.error || 'Failed to load data');
      }
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Failed to load data');
    } finally {
      setLoading(false);
    }
  };

  if (loading && !data) {
    return (
      <div className="container">
        <div className="loading">Loading dashboard data...</div>
      </div>
    );
  }

  return (
    <div>
      <div className="container">
        <h2>Energy Dashboard</h2>
        
        {error && <div className="error">{error}</div>}
        
        {healthStatus && (
          <div className="health-status">
            <span className={`status-indicator ${healthStatus.model_trained ? 'trained' : 'not-trained'}`}></span>
            <span>Model Status: {healthStatus.model_trained ? 'Trained' : 'Not Trained'}</span>
          </div>
        )}

        {!data && !error && (
          <div className="info-message">
            <p>No data available. Please load sample data or upload your data files.</p>
            <button onClick={loadData} className="btn btn-primary">Load Sample Data</button>
          </div>
        )}

        {data && (
          <>
            <div className="stats-section">
              <div className="stat-card">
                <h4>Total Data Points</h4>
                <div className="stat-value">{data.length}</div>
              </div>
              <div className="stat-card">
                <h4>Avg Load (kW)</h4>
                <div className="stat-value">
                  {(data.reduce((sum, d) => sum + d.load, 0) / data.length).toFixed(2)}
                </div>
              </div>
              <div className="stat-card">
                <h4>Avg PV (kW)</h4>
                <div className="stat-value">
                  {(data.reduce((sum, d) => sum + d.pv, 0) / data.length).toFixed(2)}
                </div>
              </div>
              <div className="stat-card">
                <h4>Max Load (kW)</h4>
                <div className="stat-value">
                  {Math.max(...data.map(d => d.load)).toFixed(2)}
                </div>
              </div>
            </div>

            <div className="chart-wrapper">
              <h3>Load and PV Generation</h3>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={data.slice(-500)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="timestamp" 
                    angle={-45}
                    textAnchor="end"
                    height={80}
                    tick={{ fontSize: 10 }}
                    interval="preserveStartEnd"
                  />
                  <YAxis label={{ value: 'Power (kW)', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="load" 
                    stroke="#8884d8" 
                    strokeWidth={2}
                    name="Load (kW)"
                    dot={false}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="pv" 
                    stroke="#82ca9d" 
                    strokeWidth={2}
                    name="PV (kW)"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <button onClick={loadData} className="btn btn-secondary">Refresh Data</button>
          </>
        )}
      </div>
    </div>
  );
}

export default Dashboard;


