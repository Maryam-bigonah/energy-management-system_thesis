# Collaboration Guide for Energy Management System

## 🎯 Quick Start for New Contributors

Welcome! This guide will help you get started with the Torino Building Energy Forecasting project.

### Project Overview

- **Purpose**: LSTM-based energy forecasting for Torino buildings
- **Tech Stack**: Python (Flask), React, TensorFlow/Keras
- **Main Features**: Load forecasting, PV generation forecasting, Web dashboard

### Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Maryam-bigonah/energy-management-system-.git
   cd energy-management-system-
   ```

2. **Set up development environment:**
   ```bash
   # Install backend dependencies
   cd backend
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   
   # Install frontend dependencies
   cd ../frontend
   npm install
   ```

3. **Run the application:**
   ```bash
   # From project root
   docker-compose up
   # OR manually start backend and frontend
   ```

### Project Structure

```
.
├── backend/           # Flask API server
├── frontend/          # React web application
├── data/              # Data files (not in git)
├── lstm_energy_forecast.py    # LSTM model
├── build_master_dataset.py   # Dataset builder
└── query_pvgis_for_buildings.py  # PVGIS integration
```

### Development Workflow

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**

3. **Test your changes:**
   ```bash
   python3 test_setup.py
   ```

4. **Commit:**
   ```bash
   git add .
   git commit -m "Add: Description of your feature"
   ```

5. **Push and create Pull Request:**
   ```bash
   git push origin feature/your-feature-name
   ```

### Code Style

- **Python**: Follow PEP 8
- **JavaScript**: Follow ESLint rules
- **Comments**: Document complex functions
- **Naming**: Use descriptive variable names

### Need Help?

- Check `README.md` for detailed documentation
- Check `TEST_RESULTS.md` for testing information
- Create an issue on GitHub for questions

### Contributing

Thank you for contributing! Please:
1. Test your changes
2. Update documentation if needed
3. Keep commits focused and descriptive
4. Create pull requests for review

---

Happy coding! 🚀

