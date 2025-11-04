# 🔧 Fix: Servers Won't Start

## The Problem

**npm is not installed** - You need Node.js to run the frontend.

---

## ✅ Solution

### Step 1: Install Node.js

1. **Download Node.js:**
   - Go to: https://nodejs.org/
   - Download the **LTS version** (recommended)
   - Install it (just click through the installer)

2. **Verify installation:**
   Open Terminal and run:
   ```bash
   node --version
   npm --version
   ```
   
   You should see version numbers (e.g., `v18.17.0`)

### Step 2: Install Frontend Dependencies

Once Node.js is installed:
```bash
cd /Users/mariabigonah/Desktop/thesis/code/frontend
npm install
```

**Wait 1-2 minutes** for packages to install.

### Step 3: Start Servers

**Terminal 1 - Backend:**
```bash
cd /Users/mariabigonah/Desktop/thesis/code/backend
python3 app.py
```

**Terminal 2 - Frontend:**
```bash
cd /Users/mariabigonah/Desktop/thesis/code/frontend
npm start
```

---

## 🚀 Alternative: Use Docker (If Node.js Install Fails)

If you have Docker installed:

```bash
cd /Users/mariabigonah/Desktop/thesis/code
docker-compose up
```

This starts both servers automatically!

---

## 📋 Quick Checklist

- [ ] Install Node.js from nodejs.org
- [ ] Verify: `npm --version` works
- [ ] Run: `cd frontend && npm install`
- [ ] Start backend in Terminal 1
- [ ] Start frontend in Terminal 2
- [ ] Open: http://localhost:3000/complete

---

**Once Node.js is installed, everything should work!** 🎉

