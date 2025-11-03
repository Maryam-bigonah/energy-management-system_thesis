# 🔧 Debug Guide - "It Doesn't Work"

## Common Issues & Fixes

### Issue 1: Page Won't Load / 404 Error

**Symptoms:**
- Browser shows "Cannot GET /complete"
- Page not found error

**Fix:**
1. Make sure **frontend is running**:
   ```bash
   cd frontend
   npm start
   ```
2. Wait for "Compiled successfully!"
3. Browser should open automatically
4. Go to: http://localhost:3000/complete

---

### Issue 2: "Cannot connect to backend API"

**Symptoms:**
- Red error message in browser
- "Cannot connect to backend API"

**Fix:**
1. **Start backend first:**
   ```bash
   cd backend
   python3 app.py
   ```
2. **Wait for:** "Running on http://0.0.0.0:5000"
3. **Test backend:**
   - Open: http://localhost:5000/api/health
   - Should see JSON response
4. **Then start frontend** (in another terminal)

---

### Issue 3: No Data Showing

**Symptoms:**
- Page loads but shows "No data" or empty charts

**Fix:**
1. **Check backend terminal** - should show "Loaded XXXX records"
2. **If no data:**
   ```bash
   # Backend loads sample data automatically
   # If not working, restart backend:
   cd backend
   python3 app.py
   ```
3. **For master dataset:**
   - Need to create it first: `python3 build_master_dataset_final.py`
   - Then load via API

---

### Issue 4: Syntax Errors / Compilation Errors

**Symptoms:**
- Terminal shows red errors
- "Compiled with problems"

**Fix:**
1. Check terminal for specific error
2. Already fixed Forecasts.js error
3. If new errors, share the error message

---

### Issue 5: Port Already in Use

**Symptoms:**
- "Port 5000 is already in use"
- "Port 3000 is already in use"

**Fix:**

**For port 5000:**
```bash
# Find and kill process
kill -9 $(lsof -ti:5000)
# Then restart backend
cd backend && python3 app.py
```

**For port 3000:**
```bash
# Find and kill process
kill -9 $(lsof -ti:3000)
# Or npm will ask to use another port (say yes)
```

---

### Issue 6: Blank Page / White Screen

**Symptoms:**
- Browser opens but shows blank page
- No content visible

**Fix:**
1. **Check browser console:**
   - Press F12 (or Cmd+Option+I on Mac)
   - Look for errors in Console tab
   - Share error messages

2. **Check if React compiled:**
   - Terminal should show "Compiled successfully!"
   - If errors, fix them first

3. **Hard refresh:**
   - Cmd+Shift+R (Mac) or Ctrl+Shift+R (Windows)

---

## 🔍 Step-by-Step Debug

### Step 1: Check Both Servers Running

**Terminal 1 - Backend:**
```bash
cd /Users/mariabigonah/Desktop/thesis/code/backend
python3 app.py
```

**Should see:**
```
Initializing with sample data...
Loaded XXXX records
 * Running on http://0.0.0.0:5000
```

**Terminal 2 - Frontend:**
```bash
cd /Users/mariabigonah/Desktop/thesis/code/frontend
npm start
```

**Should see:**
```
Compiled successfully!

You can now view the app in the browser.

  Local:            http://localhost:3000
```

---

### Step 2: Test Backend Directly

Open browser and go to:
```
http://localhost:5000/api/health
```

**Should see:**
```json
{"status":"healthy","model_loaded":false,"model_trained":false}
```

**If this doesn't work:** Backend isn't running properly

---

### Step 3: Test Frontend

Open browser and go to:
```
http://localhost:3000
```

**Should see:** Energy Forecast app (Dashboard page)

**If this doesn't work:** Frontend isn't running properly

---

### Step 4: Test Complete View

From http://localhost:3000, click **"Complete View"** in navbar

**OR go directly to:**
```
http://localhost:3000/complete
```

**Should see:** Complete Visualization page with tabs

---

## 🆘 Still Not Working?

**Tell me:**

1. **What exactly happens?**
   - Page won't load?
   - Page loads but shows error?
   - Page loads but no data?

2. **What error message do you see?**
   - In browser?
   - In terminal?

3. **Which page are you trying to open?**
   - http://localhost:3000/complete?
   - http://localhost:3000?
   - http://localhost:5000?

4. **Are both servers running?**
   - Backend terminal shows "Running on..."?
   - Frontend terminal shows "Compiled successfully"?

**Share these details and I'll fix it!** 🔧

