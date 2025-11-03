# Troubleshooting - Servers Not Opening

## 🔍 Quick Diagnosis

### Problem: Links don't open / Servers not starting

Let's check what's wrong:

---

## ✅ Step-by-Step Fix

### 1. Check Backend Dependencies

Run this command:
```bash
cd /Users/mariabigonah/Desktop/thesis/code/backend
python3 -c "import flask, flask_cors, pandas, numpy; print('OK')"
```

**If you see an error:**
```bash
# Install missing dependencies
pip3 install flask flask-cors pandas numpy
```

---

### 2. Check Frontend Dependencies

Run this command:
```bash
cd /Users/mariabigonah/Desktop/thesis/code/frontend
npm --version
```

**If npm is not found:**
- Install Node.js from: https://nodejs.org/

**If npm is installed:**
```bash
# Install frontend dependencies
cd frontend
npm install
```

---

### 3. Start Backend Manually

Open **Terminal 1**:
```bash
cd /Users/mariabigonah/Desktop/thesis/code/backend
python3 app.py
```

**What you should see:**
```
Initializing with sample data...
Loaded XXXX records
 * Running on http://0.0.0.0:5000
```

**If you see errors:**
- Check Python version: `python3 --version` (need 3.7+)
- Install dependencies: `pip3 install -r requirements.txt`
- Check for import errors in the terminal

---

### 4. Start Frontend Manually

Open **Terminal 2** (new terminal window):
```bash
cd /Users/mariabigonah/Desktop/thesis/code/frontend
npm start
```

**What you should see:**
```
Compiled successfully!

You can now view the app in the browser.

  Local:            http://localhost:3000
```

**If you see errors:**
- Run: `npm install` first
- Check Node.js version: `node --version` (need 14+)

---

## 🔧 Common Issues

### Issue 1: "Port 5000 already in use"

**Fix:**
```bash
# Find what's using port 5000
lsof -ti:5000

# Kill it
kill -9 $(lsof -ti:5000)

# Or use different port in backend/app.py
# Change: app.run(host='0.0.0.0', port=5000)
# To: app.run(host='0.0.0.0', port=5001)
```

### Issue 2: "Port 3000 already in use"

**Fix:**
```bash
# Find what's using port 3000
lsof -ti:3000

# Kill it
kill -9 $(lsof -ti:3000)

# Or in frontend, it will automatically ask to use another port
```

### Issue 3: "Cannot connect to backend API"

**Fix:**
1. Make sure backend is running (check Terminal 1)
2. Test backend directly:
   ```bash
   curl http://localhost:5000/api/health
   ```
3. If it works, backend is fine - check frontend
4. If it doesn't work, check backend errors in Terminal 1

### Issue 4: "ModuleNotFoundError" in Backend

**Fix:**
```bash
cd backend
pip3 install flask flask-cors pandas numpy requests
```

### Issue 5: Frontend won't compile

**Fix:**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm start
```

---

## 🚀 Quick Start Script

I've created a script to start both servers:

```bash
cd /Users/mariabigonah/Desktop/thesis/code
./START_SERVERS.sh
```

Or manually:
```bash
chmod +x START_SERVERS.sh
./START_SERVERS.sh
```

---

## 📋 Manual Start (Step-by-Step)

### Terminal 1 (Backend):
```bash
cd /Users/mariabigonah/Desktop/thesis/code/backend
python3 app.py
```

**Wait for:** "Running on http://0.0.0.0:5000"

### Terminal 2 (Frontend):
```bash
cd /Users/mariabigonah/Desktop/thesis/code/frontend
npm start
```

**Wait for:** "Compiled successfully" and browser to open

### Browser:
- Go to: http://localhost:3000/complete

---

## 🧪 Test Backend First

Before starting frontend, test backend:

```bash
# In Terminal 1
cd /Users/mariabigonah/Desktop/thesis/code/backend
python3 app.py
```

**In another terminal:**
```bash
curl http://localhost:5000/api/health
```

**Expected output:**
```json
{"model_loaded":false,"model_trained":false,"status":"healthy"}
```

If this works, backend is fine!

---

## 💡 Still Not Working?

Tell me:
1. What error message do you see?
2. In which terminal (backend or frontend)?
3. Can you access http://localhost:5000/api/health in browser?
4. Can you access http://localhost:3000 in browser?

Then I can help you fix the specific issue!

