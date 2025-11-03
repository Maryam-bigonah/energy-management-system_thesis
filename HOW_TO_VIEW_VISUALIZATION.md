# How to View the Complete Visualization

## 🚀 Quick Start

### Step 1: Start the Backend Server

Open a terminal and run:

```bash
cd /Users/mariabigonah/Desktop/thesis/code/backend
python3 app.py
```

You should see:
```
Initializing with sample data...
Loaded XXXX records
 * Running on http://0.0.0.0:5000
```

**Backend will be available at:** http://localhost:5000

### Step 2: Start the Frontend Server

Open **another terminal** and run:

```bash
cd /Users/mariabigonah/Desktop/thesis/code/frontend
npm start
```

You should see:
```
Compiled successfully!
You can now view the app in the browser.
  Local:            http://localhost:3000
```

**Frontend will automatically open at:** http://localhost:3000

---

## 🌐 Links to Access

### Main Application Links:

1. **Complete Visualization Page:**
   - **URL:** http://localhost:3000/complete
   - **Direct link:** Click "Complete View" in the navbar
   - **What you'll see:** All data visualization (family profiles, battery, economics)

2. **Dashboard:**
   - **URL:** http://localhost:3000/
   - **Direct link:** Click "Dashboard" in the navbar
   - **What you'll see:** Basic load and PV visualization

3. **Training:**
   - **URL:** http://localhost:3000/training
   - **What you'll see:** Model training interface

4. **Forecasts:**
   - **URL:** http://localhost:3000/forecasts
   - **What you'll see:** Prediction visualizations

### API Endpoints (for testing):

1. **Health Check:**
   - http://localhost:5000/api/health

2. **Master Dataset:**
   - http://localhost:5000/api/data/master-dataset?limit=100

3. **Battery Data:**
   - http://localhost:5000/api/battery/data?limit=100

4. **Summary:**
   - http://localhost:5000/api/data/summary

---

## 📊 What You'll See in Complete Visualization

The **Complete View** page has 5 tabs:

### 1. **Overview Tab**
- System summary statistics
- Total data points, apartments, averages
- Battery and economic metrics
- Quick overview chart

### 2. **Family Load Profiles Tab**
- Load consumption for each family type:
  - Couple working (apartments 01-05)
  - Family one child (apartments 06-10)
  - One working (apartments 11-15)
  - Retired (apartments 16-20)
- Combined comparison chart
- Individual family type charts

### 3. **Battery & PV Tab**
- Battery SOC (State of Charge)
- Charge/discharge power
- PV generation vs total load
- Grid import/export
- **Action:** Click "Run Battery Simulation" if battery data not loaded

### 4. **Economics Tab**
- Grid import costs
- Grid export revenues
- Net cost calculation
- (Requires tariffs data)

### 5. **Summary Tab**
- Complete data summary table
- All statistics in one place

---

## 🔧 If You See Errors

### Error: "Cannot connect to backend API"

**Solution:**
1. Make sure backend is running on port 5000
2. Check: http://localhost:5000/api/health
3. If not accessible, restart backend:
   ```bash
   cd backend
   python3 app.py
   ```

### Error: "No data loaded"

**Solution:**
1. Backend should auto-load sample data
2. If not, check backend terminal for errors
3. Or use Dashboard page to load sample data first

### Error: Frontend not starting

**Solution:**
1. Make sure you're in frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies if needed:
   ```bash
   npm install
   ```
3. Start again:
   ```bash
   npm start
   ```

---

## 📱 Alternative: Use Docker

If you prefer Docker:

```bash
cd /Users/mariabigonah/Desktop/thesis/code
docker-compose up
```

Then access:
- Frontend: http://localhost:3000
- Backend: http://localhost:5000

---

## 🎯 Quick Test

1. **Start backend:**
   ```bash
   cd backend && python3 app.py
   ```

2. **Start frontend (in new terminal):**
   ```bash
   cd frontend && npm start
   ```

3. **Open browser:**
   - Go to: http://localhost:3000/complete
   - Click "Complete View" in navbar

4. **View the tabs:**
   - Overview: System stats
   - Family Load Profiles: Family consumption patterns
   - Battery & PV: Energy system
   - Economics: Costs and revenues
   - Summary: All statistics

---

## ✅ Success Indicators

✅ Backend shows: "Running on http://0.0.0.0:5000"  
✅ Frontend shows: "Compiled successfully" and opens in browser  
✅ Complete View page loads with tabs visible  
✅ Data appears in charts (may take a moment to load)

---

**That's it! Your complete visualization is ready to view!** 🎉

