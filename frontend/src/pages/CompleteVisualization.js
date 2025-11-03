import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ComposedChart
} from 'recharts';
import {
  getMasterDataset, getBatteryData, getDataSummary,
  simulateBattery, analyzeEconomic,
  getFamilyConsumption, getStorageEnergy, getAllDataSummary
} from '../services/api';
import './CompleteVisualization.css';

function CompleteVisualization() {
  const [masterData, setMasterData] = useState(null);
  const [batteryData, setBatteryData] = useState(null);
  const [economicData, setEconomicData] = useState(null);
  const [summary, setSummary] = useState(null);
  const [familyData, setFamilyData] = useState(null);
  const [storageData, setStorageData] = useState(null);
  const [allDataSummary, setAllDataSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');

  // Family type mapping (assuming apartments 1-5 = family type 1, etc.)
  const familyTypes = {
    'couple_working': ['apartment_01', 'apartment_02', 'apartment_03', 'apartment_04', 'apartment_05'],
    'family_one_child': ['apartment_06', 'apartment_07', 'apartment_08', 'apartment_09', 'apartment_10'],
    'one_working': ['apartment_11', 'apartment_12', 'apartment_13', 'apartment_14', 'apartment_15'],
    'retired': ['apartment_16', 'apartment_17', 'apartment_18', 'apartment_19', 'apartment_20']
  };

  useEffect(() => {
    loadAllData();
  }, []);

  const loadAllData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Load master dataset
      const masterResponse = await getMasterDataset(null, null, 1000);
      if (masterResponse.data.success) {
        setMasterData(masterResponse.data.data);
      }

      // Load summary
      const summaryResponse = await getDataSummary();
      if (summaryResponse.data.success) {
        setSummary(summaryResponse.data.summary);
      }

      // Load family consumption
      try {
        const familyResponse = await getFamilyConsumption(null, null, 1000);
        if (familyResponse.data.success) {
          setFamilyData(familyResponse.data.data);
        }
      } catch (e) {
        console.log('Family data not available yet');
      }

      // Load storage energy data
      try {
        const storageResponse = await getStorageEnergy(1000);
        if (storageResponse.data.success) {
          setStorageData(storageResponse.data.data);
        }
      } catch (e) {
        console.log('Storage data not available yet');
      }

      // Load all data summary
      try {
        const allSummaryResponse = await getAllDataSummary();
        if (allSummaryResponse.data.success) {
          setAllDataSummary(allSummaryResponse.data.summary);
        }
      } catch (e) {
        console.log('All data summary not available yet');
      }

      // Load battery data if available
      try {
        const batteryResponse = await getBatteryData(1000);
        if (batteryResponse.data.success) {
          setBatteryData(batteryResponse.data.data);
        }
      } catch (e) {
        console.log('Battery data not available yet');
      }

    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Failed to load data');
    } finally {
      setLoading(false);
    }
  };

  const handleBatterySimulation = async () => {
    if (!masterData) return;

    try {
      const response = await simulateBattery(20.0, 'energy_share', 0.5, 1000);
      if (response.data.success) {
        setBatteryData(response.data.data);
      }
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Failed to simulate battery');
    }
  };

  const prepareFamilyLoadData = () => {
    if (!masterData || !masterData.apartments) return [];

    const familyData = {};
    
    // Group apartments by family type
    Object.keys(familyTypes).forEach(familyType => {
      familyData[familyType] = [];
    });

    // Sum loads for each family type
    for (let i = 0; i < masterData.timestamps.length; i++) {
      const timestamp = masterData.timestamps[i];
      const dataPoint = { timestamp };

      Object.keys(familyTypes).forEach(familyType => {
        const apartments = familyTypes[familyType];
        let totalLoad = 0;
        apartments.forEach(apt => {
          if (masterData.apartments[apt] && masterData.apartments[apt][i] !== undefined) {
            totalLoad += masterData.apartments[apt][i];
          }
        });
        dataPoint[familyType] = totalLoad;
      });

      // Add PV and total load
      if (masterData.pv && masterData.pv[i] !== undefined) {
        dataPoint.pv = masterData.pv[i];
      }
      if (masterData.total_load && masterData.total_load[i] !== undefined) {
        dataPoint.total_load = masterData.total_load[i];
      }

      Object.keys(familyTypes).forEach(familyType => {
        familyData[familyType].push(dataPoint);
      });
    }

    // Return data for charting (sample last 168 hours - 1 week)
    const sampleData = familyData['couple_working'].slice(-168);
    return sampleData;
  };

  const prepareBatteryChartData = () => {
    if (!batteryData) return [];

    return batteryData.timestamps.map((ts, idx) => ({
      timestamp: new Date(ts).toLocaleString(),
      soc: batteryData.battery_soc[idx] * 100, // Convert to percentage
      charge: batteryData.battery_charge_total[idx],
      discharge: batteryData.battery_discharge_total[idx],
      pv: batteryData.building_pv[idx] || 0,
      load: batteryData.building_total_load[idx] || 0,
      grid_import: batteryData.grid_import[idx] || 0,
      grid_export: batteryData.grid_export[idx] || 0,
    })).slice(-168); // Last week
  };

  const prepareEconomicChartData = () => {
    if (!economicData) return [];

    return economicData.timestamps.map((ts, idx) => ({
      timestamp: new Date(ts).toLocaleString(),
      import_cost: economicData.grid_import_cost[idx] || 0,
      export_revenue: economicData.grid_export_revenue[idx] || 0,
      net_cost: (economicData.grid_import_cost[idx] || 0) - (economicData.grid_export_revenue[idx] || 0),
    })).slice(-168);
  };

  if (loading) {
    return (
      <div className="container">
        <div className="loading">Loading all data...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container">
        <div className="error">{error}</div>
        <button onClick={loadAllData} className="btn btn-primary">Retry</button>
      </div>
    );
  }

  const familyLoadData = prepareFamilyLoadData();
  const batteryChartData = prepareBatteryChartData();
  const economicChartData = prepareEconomicChartData();

  return (
    <div className="complete-visualization">
      <div className="container">
        <h2>Complete Energy System Visualization</h2>

        {/* Tabs */}
        <div className="tabs">
          <button
            className={activeTab === 'overview' ? 'active' : ''}
            onClick={() => setActiveTab('overview')}
          >
            Overview
          </button>
          <button
            className={activeTab === 'families' ? 'active' : ''}
            onClick={() => setActiveTab('families')}
          >
            Family Load Profiles
          </button>
          <button
            className={activeTab === 'battery' ? 'active' : ''}
            onClick={() => setActiveTab('battery')}
          >
            Battery & PV
          </button>
          <button
            className={activeTab === 'economic' ? 'active' : ''}
            onClick={() => setActiveTab('economic')}
          >
            Economics
          </button>
          <button
            className={activeTab === 'summary' ? 'active' : ''}
            onClick={() => setActiveTab('summary')}
          >
            Summary
          </button>
        </div>

        {/* Overview Tab */}
        {activeTab === 'overview' && (
          <div className="tab-content">
            <h3>System Overview</h3>
            {summary && (
              <div className="stats-grid">
                {summary.master_dataset && (
                  <>
                    <div className="stat-card">
                      <h4>Total Data Points</h4>
                      <div className="stat-value">{summary.master_dataset.count.toLocaleString()}</div>
                    </div>
                    <div className="stat-card">
                      <h4>Number of Apartments</h4>
                      <div className="stat-value">{summary.master_dataset.n_apartments}</div>
                    </div>
                    <div className="stat-card">
                      <h4>Avg Total Load</h4>
                      <div className="stat-value">{summary.master_dataset.total_load_mean?.toFixed(2)} kW</div>
                    </div>
                    <div className="stat-card">
                      <h4>Avg PV Generation</h4>
                      <div className="stat-value">{summary.master_dataset.pv_mean?.toFixed(2)} kW</div>
                    </div>
                  </>
                )}
                {summary.battery && (
                  <>
                    <div className="stat-card">
                      <h4>Avg Battery SOC</h4>
                      <div className="stat-value">{(summary.battery.avg_soc * 100).toFixed(1)}%</div>
                    </div>
                    <div className="stat-card">
                      <h4>Total Charge</h4>
                      <div className="stat-value">{summary.battery.total_charge.toFixed(2)} kWh</div>
                    </div>
                    <div className="stat-card">
                      <h4>Total Discharge</h4>
                      <div className="stat-value">{summary.battery.total_discharge.toFixed(2)} kWh</div>
                    </div>
                  </>
                )}
                {summary.economic && (
                  <>
                    <div className="stat-card">
                      <h4>Total Import Cost</h4>
                      <div className="stat-value">€{summary.economic.total_import_cost.toFixed(2)}</div>
                    </div>
                    <div className="stat-card">
                      <h4>Total Export Revenue</h4>
                      <div className="stat-value">€{summary.economic.total_export_revenue.toFixed(2)}</div>
                    </div>
                    <div className="stat-card">
                      <h4>Net Cost</h4>
                      <div className="stat-value">€{summary.economic.net_cost.toFixed(2)}</div>
                    </div>
                  </>
                )}
              </div>
            )}

            {/* Quick charts */}
            {familyLoadData.length > 0 && (
              <div className="chart-section">
                <h4>Last Week: Total Load vs PV</h4>
                <ResponsiveContainer width="100%" height={300}>
                  <ComposedChart data={familyLoadData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="timestamp" angle={-45} textAnchor="end" height={80} tick={{ fontSize: 10 }} />
                    <YAxis label={{ value: 'Power (kW)', angle: -90, position: 'insideLeft' }} />
                    <Tooltip />
                    <Legend />
                    <Area type="monotone" dataKey="total_load" fill="#8884d8" fillOpacity={0.3} name="Total Load" />
                    <Area type="monotone" dataKey="pv" fill="#82ca9d" fillOpacity={0.3} name="PV Generation" />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        )}

        {/* Family Load Profiles Tab */}
        {activeTab === 'families' && (
          <div className="tab-content">
            <h3>Load Profiles by Family Type</h3>
            {familyLoadData.length > 0 && (
              <>
                <div className="chart-section">
                  <h4>All Family Types (Last Week)</h4>
                  <ResponsiveContainer width="100%" height={400}>
                    <LineChart data={familyLoadData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="timestamp" angle={-45} textAnchor="end" height={80} tick={{ fontSize: 10 }} />
                      <YAxis label={{ value: 'Load (kW)', angle: -90, position: 'insideLeft' }} />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="couple_working" stroke="#8884d8" name="Couple Working" dot={false} />
                      <Line type="monotone" dataKey="family_one_child" stroke="#82ca9d" name="Family One Child" dot={false} />
                      <Line type="monotone" dataKey="one_working" stroke="#ffc658" name="One Working" dot={false} />
                      <Line type="monotone" dataKey="retired" stroke="#ff7300" name="Retired" dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                {/* Individual family type charts */}
                {Object.keys(familyTypes).map(familyType => (
                  <div key={familyType} className="chart-section">
                    <h4>{familyType.replace(/_/g, ' ').toUpperCase()}</h4>
                    <ResponsiveContainer width="100%" height={300}>
                      <AreaChart data={familyLoadData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="timestamp" angle={-45} textAnchor="end" height={80} tick={{ fontSize: 10 }} />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Area type="monotone" dataKey={familyType} fill="#8884d8" fillOpacity={0.6} name={familyType} />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                ))}
              </>
            )}
          </div>
        )}

        {/* Battery & PV Tab */}
        {activeTab === 'battery' && (
          <div className="tab-content">
            <h3>Shared Battery & PV System</h3>
            {!batteryData && (
              <div className="info-section">
                <p>Battery simulation not run yet. Click below to simulate.</p>
                <button onClick={handleBatterySimulation} className="btn btn-primary">
                  Run Battery Simulation
                </button>
              </div>
            )}
            {batteryChartData.length > 0 && (
              <>
                <div className="chart-section">
                  <h4>Battery SOC & Power Flow (Last Week)</h4>
                  <ResponsiveContainer width="100%" height={400}>
                    <ComposedChart data={batteryChartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="timestamp" angle={-45} textAnchor="end" height={80} tick={{ fontSize: 10 }} />
                      <YAxis yAxisId="left" label={{ value: 'Power (kW)', angle: -90, position: 'insideLeft' }} />
                      <YAxis yAxisId="right" orientation="right" label={{ value: 'SOC (%)', angle: 90 }} />
                      <Tooltip />
                      <Legend />
                      <Area yAxisId="left" type="monotone" dataKey="pv" fill="#82ca9d" fillOpacity={0.3} name="PV Generation" />
                      <Area yAxisId="left" type="monotone" dataKey="load" fill="#8884d8" fillOpacity={0.3} name="Total Load" />
                      <Line yAxisId="left" type="monotone" dataKey="charge" stroke="#00ff00" name="Battery Charge" />
                      <Line yAxisId="left" type="monotone" dataKey="discharge" stroke="#ff0000" name="Battery Discharge" />
                      <Line yAxisId="right" type="monotone" dataKey="soc" stroke="#ff7300" name="Battery SOC (%)" />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>

                <div className="chart-section">
                  <h4>Grid Interaction (Last Week)</h4>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={batteryChartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="timestamp" angle={-45} textAnchor="end" height={80} tick={{ fontSize: 10 }} />
                      <YAxis label={{ value: 'Power (kW)', angle: -90, position: 'insideLeft' }} />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="grid_import" fill="#ff4444" name="Grid Import" />
                      <Bar dataKey="grid_export" fill="#44ff44" name="Grid Export" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </>
            )}
          </div>
        )}

        {/* Economic Tab */}
        {activeTab === 'economic' && (
          <div className="tab-content">
            <h3>Economic Analysis</h3>
            {economicChartData.length === 0 && (
              <div className="info-section">
                <p>Economic analysis not run yet. Run battery simulation first, then economic analysis.</p>
              </div>
            )}
            {economicChartData.length > 0 && (
              <>
                <div className="chart-section">
                  <h4>Costs and Revenues (Last Week)</h4>
                  <ResponsiveContainer width="100%" height={400}>
                    <ComposedChart data={economicChartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="timestamp" angle={-45} textAnchor="end" height={80} tick={{ fontSize: 10 }} />
                      <YAxis label={{ value: 'Euro (€)', angle: -90, position: 'insideLeft' }} />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="import_cost" fill="#ff4444" name="Import Cost (€)" />
                      <Bar dataKey="export_revenue" fill="#44ff44" name="Export Revenue (€)" />
                      <Line type="monotone" dataKey="net_cost" stroke="#000000" name="Net Cost (€)" />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              </>
            )}
          </div>
        )}

        {/* Summary Tab */}
        {activeTab === 'summary' && (
          <div className="tab-content">
            <h3>Data Summary</h3>
            {summary && (
              <div className="summary-table">
                <table>
                  <thead>
                    <tr>
                      <th>Category</th>
                      <th>Metric</th>
                      <th>Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {summary.master_dataset && (
                      <>
                        <tr>
                          <td rowSpan="4">Master Dataset</td>
                          <td>Total Records</td>
                          <td>{summary.master_dataset.count.toLocaleString()}</td>
                        </tr>
                        <tr>
                          <td>Number of Apartments</td>
                          <td>{summary.master_dataset.n_apartments}</td>
                        </tr>
                        <tr>
                          <td>Avg Total Load</td>
                          <td>{summary.master_dataset.total_load_mean?.toFixed(2)} kW</td>
                        </tr>
                        <tr>
                          <td>Avg PV Generation</td>
                          <td>{summary.master_dataset.pv_mean?.toFixed(2)} kW</td>
                        </tr>
                      </>
                    )}
                    {summary.battery && (
                      <>
                        <tr>
                          <td rowSpan="5">Battery</td>
                          <td>Avg SOC</td>
                          <td>{(summary.battery.avg_soc * 100).toFixed(1)}%</td>
                        </tr>
                        <tr>
                          <td>Total Charge</td>
                          <td>{summary.battery.total_charge.toFixed(2)} kWh</td>
                        </tr>
                        <tr>
                          <td>Total Discharge</td>
                          <td>{summary.battery.total_discharge.toFixed(2)} kWh</td>
                        </tr>
                        <tr>
                          <td>Total Grid Import</td>
                          <td>{summary.battery.total_grid_import.toFixed(2)} kWh</td>
                        </tr>
                        <tr>
                          <td>Total Grid Export</td>
                          <td>{summary.battery.total_grid_export.toFixed(2)} kWh</td>
                        </tr>
                      </>
                    )}
                    {summary.economic && (
                      <>
                        <tr>
                          <td rowSpan="3">Economics</td>
                          <td>Total Import Cost</td>
                          <td>€{summary.economic.total_import_cost.toFixed(2)}</td>
                        </tr>
                        <tr>
                          <td>Total Export Revenue</td>
                          <td>€{summary.economic.total_export_revenue.toFixed(2)}</td>
                        </tr>
                        <tr>
                          <td>Net Cost</td>
                          <td>€{summary.economic.net_cost.toFixed(2)}</td>
                        </tr>
                      </>
                    )}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        <div className="actions">
          <button onClick={loadAllData} className="btn btn-secondary">Refresh All Data</button>
        </div>
      </div>
    </div>
  );
}

export default CompleteVisualization;

