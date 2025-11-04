# 🔍 Complete System Check

## What I Just Checked

I ran a comprehensive check of your system. Here's what needs to be done:

---

## ✅ What Should Work

1. **Backend Python Dependencies** - ✅ All installed
2. **Backend Code Files** - ✅ All present
3. **Frontend Code Files** - ✅ All present
4. **Ports** - Should be available

---

## ❌ What Needs to be Fixed

### 1. Node.js Installation
**Status:** Need to verify installation completed

**Check:**
```bash
node --version
npm --version
```

**If it doesn't show versions:**
- The .pkg installer may not have completed
- Or you need to restart Terminal
- Or Node.js wasn't added to PATH

**Fix:**
1. Restart your Terminal completely
2. Then try `node --version` again
3. If still not working, reinstall Node.js

### 2. Frontend Dependencies
**Status:** Not installed yet

**Fix:**
```bash
cd /Users/mariabigonah/Desktop/thesis/code/frontend
npm install
```

This must be done AFTER Node.js is confirmed working.

### 3. Servers Not Running
**Status:** Both servers need to be started

**Fix:** Start both servers (see below)

---

## 🚀 Step-by-Step Fix

### Step 1: Verify Node.js (Do This First!)

Open a NEW Terminal window and run:
```bash
node --version
npm --version
```

**Expected output:**
```
v20.10.0
10.2.3
```

**If you see errors:**
- Node.js installation didn't complete
- Try restarting Terminal
- Or reinstall Node.js from nodejs.org

### Step 2: Install Frontend Dependencies

Once Node.js works:
```bash
cd /Users/mariabigonah/Desktop/thesis/code/frontend
npm install
```

**Wait 1-2 minutes** - you'll see packages downloading.

### Step 3: Start Backend Server

**Terminal 1:**
```bash
cd /Users/mariabigonah/Desktop/thesis/code/backend
python3 app.py
```

**Wait for:** "Running on http://0.0.0.0:5000"

**Test it:** Open browser to http://localhost:5000/api/health

You should see JSON response!

### Step 4: Start Frontend Server

**Terminal 2 (NEW terminal):**
```bash
cd /Users/mariabigonah/Desktop/thesis/code/frontend
npm start
```

**Wait for:** Browser to open automatically to http://localhost:3000

---

## 🧪 Test Before Opening Links

### Test Backend First:
```bash
# In terminal
curl http://localhost:5000/api/health
```

**Expected:**
```json
{"status":"healthy","model_loaded":false,"model_trained":false}
```

**If this works:** Backend is ready! ✅

**If this fails:** Backend isn't running - check Terminal 1 for errors

### Test Frontend:
- Browser should open automatically
- Or go to: http://localhost:3000
- You should see the Energy Forecast app

---

## ❓ Common Issues

### Issue: "node: command not found"
**Cause:** Node.js not installed or Terminal not restarted
**Fix:** 
1. Restart Terminal completely
2. Try `node --version` again
3. If still fails, reinstall Node.js

### Issue: Backend won't start
**Check Terminal 1 for error messages:**
- Import errors? Install: `pip3 install flask flask-cors pandas numpy`
- Port in use? Kill existing process: `kill -9 $(lsof -ti:5000)`

### Issue: Frontend won't start
**Check Terminal 2:**
- "npm: command not found"? Node.js not installed
- Compilation errors? Check the error message
- Port 3000 in use? npm will suggest another port

### Issue: Links don't work
**Before opening links:**
1. ✅ Backend running? (check Terminal 1)
2. ✅ Frontend running? (check Terminal 2)
3. ✅ Both show no errors?

**Only then:** Open http://localhost:3000/complete

---

## 📋 Pre-Flight Checklist

Before opening any links, verify:

- [ ] `node --version` works
- [ ] `npm --version` works
- [ ] `npm install` completed in frontend folder
- [ ] Backend server running (Terminal 1 shows "Running on...")
- [ ] Frontend server running (Terminal 2 shows "Compiled successfully")
- [ ] Browser opens automatically OR you can manually go to http://localhost:3000

**Only open links when ALL items above are checked!** ✅

---

## 🆘 Still Having Issues?

Tell me:
1. What does `node --version` show?
2. Are both servers running?
3. What error messages do you see?

