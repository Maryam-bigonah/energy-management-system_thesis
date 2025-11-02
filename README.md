# LSTM Energy Forecasting - Full Stack Application

This project implements a complete full-stack application for energy forecasting using LSTM neural networks. It includes a Flask backend API and a React frontend with interactive visualizations.

## 🏗️ Project Structure

```
.
├── backend/                  # Flask API server
│   ├── app.py               # Main Flask application
│   ├── requirements.txt     # Python dependencies
│   └── Dockerfile          # Backend Docker image
├── frontend/                # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── pages/          # Page components
│   │   └── services/       # API services
│   ├── package.json        # Node dependencies
│   └── Dockerfile         # Frontend Docker image
├── lstm_energy_forecast.py # LSTM model implementation
├── data_loader.py          # Data loading utilities
├── docker-compose.yml      # Docker Compose configuration
└── README.md              # This file
```

## ✨ Features

### Backend (Flask API)
- **Data Management**: Load sample data or upload CSV files from PVGIS and Load Profile Generator
- **Model Training**: Train LSTM model with configurable hyperparameters
- **Forecasting**: Generate predictions for next hour or multiple hours
- **Metrics**: Get model performance metrics (MAE, RMSE, R², MAPE)
- **RESTful API**: All endpoints follow REST conventions

### Frontend (React)
- **Dashboard**: Visualize historical load and PV generation data
- **Training Page**: Train models with real-time progress and metrics
- **Forecasts Page**: Compare predictions vs actual values with interactive charts
- **Real-time Updates**: Live metrics and model status
- **Responsive Design**: Works on desktop and mobile devices

## 🚀 Quick Start

### Option 1: Using Docker (Recommended)

1. **Build and run with Docker Compose**:
```bash
docker-compose up --build
```

2. **Access the application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:5000

### Option 2: Manual Setup

#### Backend Setup

1. **Navigate to backend directory**:
```bash
cd backend
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Run the Flask server**:
```bash
python app.py
```

The backend will run on http://localhost:5000

#### Frontend Setup

1. **Navigate to frontend directory**:
```bash
cd frontend
```

2. **Install dependencies**:
```bash
npm install
```

3. **Run the development server**:
```bash
npm start
```

The frontend will run on http://localhost:3000

## 📡 API Endpoints

### Data Endpoints
- `GET /api/health` - Health check and model status
- `POST /api/data/load-sample` - Load sample Torino data
- `POST /api/data/upload` - Upload CSV files (PV and Load)
- `GET /api/data/historical` - Get historical data for visualization

### Model Endpoints
- `POST /api/model/train` - Train the LSTM model
- `POST /api/model/predict` - Get predictions for N hours
- `GET /api/model/metrics` - Get model performance metrics
- `GET /api/model/forecast-next` - Forecast next hour

## 🎯 Usage Guide

### 1. Load Data

**Option A: Load Sample Data**
- Go to Dashboard or Training page
- Click "Load Sample Data" button

**Option B: Upload Your Data**
- Prepare CSV files from PVGIS (PV) and Load Profile Generator (Load)
- Use the upload endpoint with your files

### 2. Train Model

1. Go to **Training** page
2. Configure training parameters:
   - Epochs (default: 50)
   - Batch Size (default: 32)
   - Validation Split (default: 0.2)
3. Click **"Train Model"**
4. Monitor training progress and metrics

### 3. View Forecasts

1. Go to **Forecasts** page
2. Select number of hours to forecast
3. Click **"Load Forecasts"**
4. Compare predictions vs actual values
5. View next-hour forecast

## 📊 Visualizations

The application provides several interactive charts:

- **Historical Data**: Load and PV generation over time
- **Training History**: Loss and MAE during training
- **Forecast Comparison**: Predicted vs actual values
- **Performance Metrics**: MAE, RMSE, R², MAPE

## 🔧 Configuration

### Backend Configuration

Edit `backend/app.py` to modify:
- Port (default: 5000)
- Host (default: 0.0.0.0)
- Data directory

### Frontend Configuration

Edit `frontend/src/services/api.js` to modify:
- API base URL (default: http://localhost:5000/api)

Or set environment variable:
```bash
REACT_APP_API_URL=http://your-backend-url/api
```

## 📦 Dependencies

### Backend
- Flask 3.0.0
- flask-cors 4.0.0
- TensorFlow 2.13.0+
- pandas 2.0.0+
- scikit-learn 1.3.0+

### Frontend
- React 18.2.0
- React Router 6.20.0
- Recharts 2.10.0
- Axios 1.6.0

## 🐳 Docker Commands

```bash
# Build and start containers
docker-compose up --build

# Start in background
docker-compose up -d

# Stop containers
docker-compose down

# View logs
docker-compose logs -f

# Rebuild specific service
docker-compose build backend
docker-compose build frontend
```

## 📝 Data Format

### Expected CSV Format

**PVGIS Data**:
- Must have datetime column (as index or column)
- PV generation column (will be renamed to 'pv')

**Load Profile Generator Data**:
- Must have datetime column (as index or column)
- Load column(s) (will be summed and renamed to 'load')

Adjust column names in `data_loader.py` to match your format.

## 🎓 Model Architecture

- **Input**: Past 24 hours of load, PV, and calendar features
- **LSTM Layers**: 
  - First: 64 units (return_sequences=True)
  - Second: 32 units
  - Dropout: 0.2 after each layer
- **Output**: Next-hour load and PV predictions

## 🐛 Troubleshooting

### Backend not starting
- Check if port 5000 is available
- Ensure all Python dependencies are installed
- Check backend logs for errors

### Frontend not connecting to backend
- Verify backend is running on port 5000
- Check CORS settings in `backend/app.py`
- Update API URL in `frontend/src/services/api.js`

### Model training fails
- Ensure data is loaded first
- Check that data has at least 24 hours
- Verify data contains 'load' and 'pv' columns

## 📄 License

This project is for research/educational purposes.

## 🤝 Contributing

Feel free to submit issues or pull requests for improvements!
