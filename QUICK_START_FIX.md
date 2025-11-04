# 🚀 Quick Fix - Get Servers Running

## The Problem
Frontend dependencies are missing. Here's how to fix it:

---

## ✅ Solution (2 Simple Steps)

### Step 1: Install Frontend Dependencies

Open Terminal and run:
```bash
cd /Users/mariabigonah/Desktop/thesis/code/frontend
npm install
```

**Wait for:** This will take 1-2 minutes. You'll see packages being installed.

---

### Step 2: Start Both Servers

You need **2 terminals** running at the same time:

#### Terminal 1 - Backend:
```bash
cd /Users/mariabigonah/Desktop/thesis/code/backend
python3 app.py
```

**You should see:**
```
Initializing with sample data...
Loaded XXXX records
 * Running on http://0.0.0.0:5000
```

**Keep this terminal open!** ✅

#### Terminal 2 - Frontend:
```bash
cd /Users/mariabigonah/Desktop/thesis/code/frontend
npm start
```

**You should see:**
```
Compiled successfully!
You can now view the app in the browser.
  Local:            http://localhost:3000
```

**Browser should open automatically!** ✅

---

## 🌐 Then Open in Browser

Once both servers are running:

1. **Complete Visualization:**
   - http://localhost:3000/complete
   - Or click "Complete View" in navbar

2. **Dashboard:**
   - http://localhost:3000/

---

## ⚠️ Common Issues

### If "npm: command not found"
Install Node.js: https://nodejs.org/

### If "port 5000 in use"
Kill existing process:
```bash
kill -9 $(lsof -ti:5000)
```

### If "port 3000 in use"
npm will ask to use another port (like 3001). Say yes!

### If backend errors
Check Terminal 1 for error messages. Usually it's a missing Python package:
```bash
cd backend
pip3 install flask flask-cors pandas numpy
```

---

## ✅ Success Checklist

- [ ] Frontend: `npm install` completed
- [ ] Terminal 1: Backend shows "Running on http://0.0.0.0:5000"
- [ ] Terminal 2: Frontend shows "Compiled successfully"
- [ ] Browser opens to http://localhost:3000
- [ ] Can click "Complete View" in navbar

---

**Once both terminals are running, your visualization will work!** 🎉

