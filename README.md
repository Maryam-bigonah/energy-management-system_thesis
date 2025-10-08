# 🌞 Torino Solar Building Analysis

A full-stack web application for analyzing solar energy potential of buildings in Torino, Italy. This application provides interactive visualizations, detailed solar analysis, and technology comparisons for residential and commercial buildings.

## 🚀 Features

- **Interactive Dashboard**: Overview of solar potential across all buildings
- **Building Directory**: Searchable and filterable list of buildings
- **Detailed Analysis**: Individual building solar energy calculations using PVGIS
- **Technology Comparison**: Compare different solar cell technologies
- **Interactive Map**: Visual representation of buildings and their solar potential
- **Statistics & Analytics**: Comprehensive data visualization and insights

## 🏗️ Architecture

### Backend (FastAPI)
- RESTful API for building data and solar analysis
- PVGIS integration for accurate solar calculations
- Building data processing and validation
- Technology comparison algorithms

### Frontend (React)
- Modern, responsive UI with Material-UI
- Interactive charts and visualizations
- Map integration with Leaflet
- Real-time data filtering and search

## 📁 Project Structure

```
├── backend/                 # FastAPI backend
│   ├── main.py             # Main API application
│   └── requirements.txt    # Python dependencies
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/     # Reusable UI components
│   │   ├── pages/          # Page components
│   │   ├── services/       # API service layer
│   │   └── App.js          # Main application
│   ├── public/             # Static assets
│   └── package.json        # Node.js dependencies
├── torino_energy/          # Core data processing modules
│   └── data_sources/
│       ├── osm_roofs.py           # OSM building data processor
│       └── pvgis_integration.py   # PVGIS solar analysis
├── building_solar_analysis.py     # Interactive CLI tool
├── demo_building_analysis.py      # Demo script
├── example_building_analysis.py   # Example usage
├── processed_building_data.csv    # Sample building data
└── requirements.txt               # Python dependencies
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the backend server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

The application will be available at `http://localhost:3000`

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

1. Create a `docker-compose.yml` file in the project root:
```yaml
version: '3.8'
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - PYTHONPATH=/app
    volumes:
      - ./torino_energy:/app/torino_energy
      - ./processed_building_data.csv:/app/processed_building_data.csv

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_URL=http://localhost:8000
    depends_on:
      - backend
```

2. Build and run:
```bash
docker-compose up --build
```

### Individual Docker Containers

#### Backend Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Frontend Dockerfile
```dockerfile
FROM node:16-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

EXPOSE 3000

CMD ["npm", "start"]
```

## 📊 API Endpoints

### Buildings
- `GET /buildings` - List all buildings with optional filtering
- `GET /buildings/{building_id}` - Get building details
- `GET /buildings/{building_id}/solar-analysis` - Solar analysis for specific building
- `GET /buildings/{building_id}/technology-comparison` - Technology comparison

### Statistics
- `GET /statistics` - Overall dataset statistics
- `GET /plot-data/roof-area-distribution` - Roof area distribution data
- `GET /plot-data/solar-potential` - Solar potential data for visualization

## 🔬 Solar Analysis Features

### Supported Technologies
- Mono-crystalline Silicon
- Poly-crystalline Silicon
- Thin Film
- Perovskite

### Analysis Metrics
- Annual energy production
- Optimal tilt and azimuth angles
- System cost and payback period
- Capacity factor
- Energy per square meter

## 🗺️ Map Features

- Interactive map with building locations
- Color-coded solar potential visualization
- Filtering by solar potential range and building type
- Detailed popup information for each building

## 📈 Data Sources

- **OpenStreetMap (OSM)**: Building geometry and basic information
- **PVGIS**: Solar irradiance and energy calculations
- **Custom Processing**: Population estimates and roof area calculations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PVGIS for solar energy calculations
- OpenStreetMap contributors for building data
- Material-UI for the frontend components
- React Leaflet for map functionality

## 📞 Support

For support and questions, please open an issue in the GitHub repository.

---

**Built with ❤️ for sustainable energy analysis in Torino, Italy**
