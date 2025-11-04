import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { trainLSTMPipeline, getLSTMEvaluation, getLSTMTrainingHistory, getLSTMStatus } from '../services/api';
import './LSTMVisualization.css';

function LSTMVisualization() {
  const [training, setTraining] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [trainingHistory, setTrainingHistory] = useState(null);
  const [evaluation, setEvaluation] = useState(null);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [status, setStatus] = useState(null);
  
  // Training parameters
  const [epochs, setEpochs] = useState(30);
  const [batchSize, setBatchSize] = useState(32);
  const [lstmUnits, setLstmUnits] = useState(64);
  const [learningRate, setLearningRate] = useState(0.001);
  const [scaleTargets, setScaleTargets] = useState(true);

  useEffect(() => {
    loadStatus();
    loadEvaluation();
    loadTrainingHistory();
  }, []);

  const loadStatus = async () => {
    try {
      const response = await getLSTMStatus();
      if (response.data.success) {
        setStatus(response.data);
      }
    } catch (err) {
      console.log('Status not available');
    }
  };

  const loadEvaluation = async () => {
    try {
      const response = await getLSTMEvaluation();
      if (response.data.success) {
        setEvaluation(response.data.evaluation);
        setMetrics({
          mae: response.data.evaluation.mae,
          rmse: response.data.evaluation.rmse
        });
      }
    } catch (err) {
      console.log('Evaluation not available');
    }
  };

  const loadTrainingHistory = async () => {
    try {
      const response = await getLSTMTrainingHistory();
      if (response.data.success) {
        setTrainingHistory(response.data.training_history);
      }
    } catch (err) {
      console.log('Training history not available');
    }
  };

  const handleTrain = async () => {
    try {
      setTraining(true);
      setError(null);
      setSuccess(null);
      
      const response = await trainLSTMPipeline(epochs, batchSize, lstmUnits, learningRate, scaleTargets);
      
      if (response.data.success) {
        setSuccess('LSTM training completed successfully!');
        setMetrics(response.data.metrics);
        setTrainingHistory(response.data.training_history);
        setEvaluation(response.data.evaluation);
        setStatus({
          model_trained: true,
          has_history: true,
          has_evaluation: true
        });
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Training failed');
      console.error('Training error:', err);
    } finally {
      setTraining(false);
    }
  };

  // Prepare training history chart data
  const prepareTrainingData = () => {
    if (!trainingHistory) return [];
    
    return trainingHistory.epochs.map((epoch, idx) => ({
      epoch,
      'Train Loss': trainingHistory.train_loss[idx],
      'Val Loss': trainingHistory.val_loss[idx],
      'Train MAE': trainingHistory.train_mae[idx],
      'Val MAE': trainingHistory.val_mae[idx]
    }));
  };

  // Prepare evaluation chart data (last 200 hours)
  const prepareEvaluationData = () => {
    if (!evaluation || !evaluation.y_pred || !evaluation.y_true) return [];
    
    const n_hours = Math.min(200, evaluation.y_pred.length);
    const start_idx = Math.max(0, evaluation.y_pred.length - n_hours);
    
    return evaluation.y_pred.slice(start_idx).map((pred, idx) => ({
      hour: idx + 1,
      'True Load': evaluation.y_true[start_idx + idx],
      'Predicted Load': pred
    }));
  };

  return (
    <div className="lstm-visualization">
      <div className="container">
        <h1>🔮 LSTM Model Training & Visualization</h1>
        
        {error && <div className="error-message">{error}</div>}
        {success && <div className="success-message">{success}</div>}

        {/* Status Card */}
        {status && (
          <div className="status-card">
            <h3>Model Status</h3>
            <div className="status-grid">
              <div className={`status-item ${status.model_trained ? 'active' : ''}`}>
                <span className="status-label">Model:</span>
                <span className="status-value">{status.model_trained ? '✅ Trained' : '❌ Not Trained'}</span>
              </div>
              <div className={`status-item ${status.has_history ? 'active' : ''}`}>
                <span className="status-label">History:</span>
                <span className="status-value">{status.has_history ? '✅ Available' : '❌ Not Available'}</span>
              </div>
              <div className={`status-item ${status.has_evaluation ? 'active' : ''}`}>
                <span className="status-label">Evaluation:</span>
                <span className="status-value">{status.has_evaluation ? '✅ Available' : '❌ Not Available'}</span>
              </div>
            </div>
          </div>
        )}

        {/* Training Configuration */}
        <div className="training-config">
          <h2>Training Configuration</h2>
          <div className="config-grid">
            <div className="config-item">
              <label>Epochs</label>
              <input
                type="number"
                min="1"
                max="200"
                value={epochs}
                onChange={(e) => setEpochs(parseInt(e.target.value))}
                disabled={training}
              />
            </div>
            <div className="config-item">
              <label>Batch Size</label>
              <input
                type="number"
                min="8"
                max="128"
                value={batchSize}
                onChange={(e) => setBatchSize(parseInt(e.target.value))}
                disabled={training}
              />
            </div>
            <div className="config-item">
              <label>LSTM Units</label>
              <input
                type="number"
                min="16"
                max="256"
                value={lstmUnits}
                onChange={(e) => setLstmUnits(parseInt(e.target.value))}
                disabled={training}
              />
            </div>
            <div className="config-item">
              <label>Learning Rate</label>
              <input
                type="number"
                min="0.0001"
                max="0.01"
                step="0.0001"
                value={learningRate}
                onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                disabled={training}
              />
            </div>
            <div className="config-item">
              <label>
                <input
                  type="checkbox"
                  checked={scaleTargets}
                  onChange={(e) => setScaleTargets(e.target.checked)}
                  disabled={training}
                />
                Scale Targets
              </label>
            </div>
          </div>
          <button
            onClick={handleTrain}
            disabled={training}
            className="btn-train"
          >
            {training ? 'Training...' : '🚀 Train LSTM Model'}
          </button>
        </div>

        {/* Metrics */}
        {metrics && (
          <div className="metrics-card">
            <h2>Model Performance (Test Set)</h2>
            <div className="metrics-grid">
              <div className="metric-item">
                <span className="metric-label">MAE</span>
                <span className="metric-value">{metrics.mae?.toFixed(4)} kW</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">RMSE</span>
                <span className="metric-value">{metrics.rmse?.toFixed(4)} kW</span>
              </div>
              {metrics.final_train_loss && (
                <>
                  <div className="metric-item">
                    <span className="metric-label">Final Train Loss</span>
                    <span className="metric-value">{metrics.final_train_loss?.toFixed(6)}</span>
                  </div>
                  <div className="metric-item">
                    <span className="metric-label">Final Val Loss</span>
                    <span className="metric-value">{metrics.final_val_loss?.toFixed(6)}</span>
                  </div>
                  <div className="metric-item">
                    <span className="metric-label">Best Epoch</span>
                    <span className="metric-value">{metrics.best_epoch}</span>
                  </div>
                  <div className="metric-item">
                    <span className="metric-label">Epochs Trained</span>
                    <span className="metric-value">{metrics.epochs_trained}</span>
                  </div>
                </>
              )}
            </div>
          </div>
        )}

        {/* Training History Chart */}
        {trainingHistory && (
          <div className="chart-card">
            <h2>Training History</h2>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={prepareTrainingData()}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Loss / MAE', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="Train Loss" stroke="#8884d8" strokeWidth={2} />
                <Line type="monotone" dataKey="Val Loss" stroke="#82ca9d" strokeWidth={2} />
                <Line type="monotone" dataKey="Train MAE" stroke="#ffc658" strokeWidth={2} />
                <Line type="monotone" dataKey="Val MAE" stroke="#ff7300" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Evaluation Chart */}
        {evaluation && evaluation.y_pred && evaluation.y_true && (
          <div className="chart-card">
            <h2>True vs Predicted Load (Last 200 Hours of Test Set)</h2>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={prepareEvaluationData()}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="hour" label={{ value: 'Hour (Last 200 hours)', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Load (kW)', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="True Load" stroke="#8884d8" strokeWidth={2} />
                <Line type="monotone" dataKey="Predicted Load" stroke="#82ca9d" strokeWidth={2} strokeDasharray="5 5" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  );
}

export default LSTMVisualization;

