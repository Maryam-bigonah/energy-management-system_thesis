import React, { useState, useEffect } from 'react';
import { getDatabaseInfo } from '../services/api';
import './DatabaseView.css';

const DatabaseView = () => {
  const [info, setInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadDatabaseInfo();
  }, []);

  const loadDatabaseInfo = async () => {
    try {
      setLoading(true);
      const response = await getDatabaseInfo();
      if (response.data.success) {
        setInfo(response.data.info);
      } else {
        setError(response.data.error || 'Failed to load database info');
      }
    } catch (err) {
      setError(err.message || 'Failed to connect to backend');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="database-view">
        <h1>Database Information</h1>
        <p>Loading...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="database-view">
        <h1>Database Information</h1>
        <div className="error-message">{error}</div>
        <button onClick={loadDatabaseInfo}>Retry</button>
      </div>
    );
  }

  if (!info) {
    return (
      <div className="database-view">
        <h1>Database Information</h1>
        <p>No data available</p>
      </div>
    );
  }

  return (
    <div className="database-view">
      <div className="database-header">
        <h1>📊 Complete Database Information</h1>
        <button onClick={loadDatabaseInfo} className="refresh-btn">🔄 Refresh</button>
      </div>

      {/* Data Loaded Status */}
      <section className="data-section">
        <h2>📦 Data Loaded in Memory</h2>
        <div className="status-grid">
          <div className={`status-card ${info.data_loaded.master_dataset ? 'loaded' : 'not-loaded'}`}>
            <h3>Master Dataset</h3>
            <p>{info.data_loaded.master_dataset ? '✅ Loaded' : '❌ Not Loaded'}</p>
          </div>
          <div className={`status-card ${info.data_loaded.battery_results ? 'loaded' : 'not-loaded'}`}>
            <h3>Battery Results</h3>
            <p>{info.data_loaded.battery_results ? '✅ Loaded' : '❌ Not Loaded'}</p>
          </div>
          <div className={`status-card ${info.data_loaded.economic_results ? 'loaded' : 'not-loaded'}`}>
            <h3>Economic Results</h3>
            <p>{info.data_loaded.economic_results ? '✅ Loaded' : '❌ Not Loaded'}</p>
          </div>
          <div className={`status-card ${info.data_loaded.historical_data ? 'loaded' : 'not-loaded'}`}>
            <h3>Historical Data</h3>
            <p>{info.data_loaded.historical_data ? '✅ Loaded' : '❌ Not Loaded'}</p>
          </div>
        </div>
      </section>

      {/* Loaded Data Statistics */}
      {Object.keys(info.loaded_data_stats).length > 0 && (
        <section className="data-section">
          <h2>📈 Loaded Data Statistics</h2>
          
          {info.loaded_data_stats.master_dataset && (
            <div className="stats-card">
              <h3>Master Dataset</h3>
              <div className="stats-grid">
                <div className="stat-item">
                  <label>Rows:</label>
                  <span>{info.loaded_data_stats.master_dataset.rows.toLocaleString()}</span>
                </div>
                <div className="stat-item">
                  <label>Apartments:</label>
                  <span>{info.loaded_data_stats.master_dataset.n_apartments}</span>
                </div>
                <div className="stat-item">
                  <label>Date Range:</label>
                  <span>
                    {new Date(info.loaded_data_stats.master_dataset.date_range.start).toLocaleDateString()} - 
                    {new Date(info.loaded_data_stats.master_dataset.date_range.end).toLocaleDateString()}
                  </span>
                </div>
                <div className="stat-item">
                  <label>Total Load (kW):</label>
                  <span>
                    Min: {info.loaded_data_stats.master_dataset.total_load_range.min.toFixed(2)} | 
                    Max: {info.loaded_data_stats.master_dataset.total_load_range.max.toFixed(2)} | 
                    Avg: {info.loaded_data_stats.master_dataset.total_load_range.mean.toFixed(2)}
                  </span>
                </div>
                <div className="stat-item">
                  <label>PV Generation (kW):</label>
                  <span>
                    Min: {info.loaded_data_stats.master_dataset.pv_range.min.toFixed(2)} | 
                    Max: {info.loaded_data_stats.master_dataset.pv_range.max.toFixed(2)} | 
                    Avg: {info.loaded_data_stats.master_dataset.pv_range.mean.toFixed(2)}
                  </span>
                </div>
              </div>
              <details>
                <summary>Columns ({info.loaded_data_stats.master_dataset.columns.length})</summary>
                <div className="columns-list">
                  {info.loaded_data_stats.master_dataset.columns.map((col, idx) => (
                    <span key={idx} className="column-tag">{col}</span>
                  ))}
                </div>
              </details>
            </div>
          )}

          {info.loaded_data_stats.battery_results && (
            <div className="stats-card">
              <h3>Battery Results</h3>
              <div className="stats-grid">
                <div className="stat-item">
                  <label>Rows:</label>
                  <span>{info.loaded_data_stats.battery_results.rows.toLocaleString()}</span>
                </div>
                <div className="stat-item">
                  <label>Date Range:</label>
                  <span>
                    {new Date(info.loaded_data_stats.battery_results.date_range.start).toLocaleDateString()} - 
                    {new Date(info.loaded_data_stats.battery_results.date_range.end).toLocaleDateString()}
                  </span>
                </div>
                <div className="stat-item">
                  <label>Total Charge:</label>
                  <span>{info.loaded_data_stats.battery_results.total_charge_kwh.toFixed(2)} kWh</span>
                </div>
                <div className="stat-item">
                  <label>Total Discharge:</label>
                  <span>{info.loaded_data_stats.battery_results.total_discharge_kwh.toFixed(2)} kWh</span>
                </div>
                <div className="stat-item">
                  <label>Average SOC:</label>
                  <span>{(info.loaded_data_stats.battery_results.avg_soc * 100).toFixed(2)}%</span>
                </div>
              </div>
              <details>
                <summary>Columns ({info.loaded_data_stats.battery_results.columns.length})</summary>
                <div className="columns-list">
                  {info.loaded_data_stats.battery_results.columns.map((col, idx) => (
                    <span key={idx} className="column-tag">{col}</span>
                  ))}
                </div>
              </details>
            </div>
          )}

          {info.loaded_data_stats.economic_results && (
            <div className="stats-card">
              <h3>Economic Results</h3>
              <div className="stats-grid">
                <div className="stat-item">
                  <label>Rows:</label>
                  <span>{info.loaded_data_stats.economic_results.rows.toLocaleString()}</span>
                </div>
                <div className="stat-item">
                  <label>Date Range:</label>
                  <span>
                    {new Date(info.loaded_data_stats.economic_results.date_range.start).toLocaleDateString()} - 
                    {new Date(info.loaded_data_stats.economic_results.date_range.end).toLocaleDateString()}
                  </span>
                </div>
                <div className="stat-item">
                  <label>Total Import Cost:</label>
                  <span>€{info.loaded_data_stats.economic_results.total_import_cost_eur.toFixed(2)}</span>
                </div>
                <div className="stat-item">
                  <label>Total Export Revenue:</label>
                  <span>€{info.loaded_data_stats.economic_results.total_export_revenue_eur.toFixed(2)}</span>
                </div>
                <div className="stat-item">
                  <label>Net Cost:</label>
                  <span>
                    €{(info.loaded_data_stats.economic_results.total_import_cost_eur - 
                      info.loaded_data_stats.economic_results.total_export_revenue_eur).toFixed(2)}
                  </span>
                </div>
              </div>
              <details>
                <summary>Columns ({info.loaded_data_stats.economic_results.columns.length})</summary>
                <div className="columns-list">
                  {info.loaded_data_stats.economic_results.columns.map((col, idx) => (
                    <span key={idx} className="column-tag">{col}</span>
                  ))}
                </div>
              </details>
            </div>
          )}
        </section>
      )}

      {/* Data Files */}
      <section className="data-section">
        <h2>📁 Data Files in Directory</h2>
        {info.data_files.length === 0 ? (
          <p>No data files found in the data directory</p>
        ) : (
          <div className="files-grid">
            {info.data_files.map((file, idx) => (
              <div key={idx} className="file-card">
                <h3>{file.name}</h3>
                <div className="file-info">
                  <div className="file-detail">
                    <label>Type:</label>
                    <span>{file.type}</span>
                  </div>
                  <div className="file-detail">
                    <label>Size:</label>
                    <span>{file.size_mb} MB ({file.size_bytes.toLocaleString()} bytes)</span>
                  </div>
                  {file.columns && (
                    <details>
                      <summary>Columns ({file.columns.length})</summary>
                      <div className="columns-list">
                        {file.columns.map((col, colIdx) => (
                          <span key={colIdx} className="column-tag">{col}</span>
                        ))}
                      </div>
                    </details>
                  )}
                  <div className="file-detail">
                    <label>Path:</label>
                    <code className="file-path">{file.path}</code>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
};

export default DatabaseView;

