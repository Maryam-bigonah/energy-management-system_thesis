#!/bin/bash

echo "=========================================="
echo "🔍 SYSTEM CHECK & START GUIDE"
echo "=========================================="
echo ""

# Check Node.js
echo "1. Checking Node.js..."
if command -v node &> /dev/null; then
    echo "   ✅ Node.js version: $(node --version)"
    echo "   ✅ npm version: $(npm --version)"
else
    echo "   ❌ Node.js NOT FOUND"
    echo "   → Install from: https://nodejs.org/"
    echo "   → Then restart Terminal and run this again"
    exit 1
fi

echo ""

# Check Python
echo "2. Checking Python..."
if command -v python3 &> /dev/null; then
    echo "   ✅ Python version: $(python3 --version)"
else
    echo "   ❌ Python NOT FOUND"
    exit 1
fi

echo ""

# Check frontend dependencies
echo "3. Checking frontend dependencies..."
if [ -d "frontend/node_modules" ]; then
    echo "   ✅ Frontend dependencies installed"
else
    echo "   ⚠️  Frontend dependencies missing"
    echo "   → Running: cd frontend && npm install"
    cd frontend && npm install && cd ..
fi

echo ""

# Check ports
echo "4. Checking ports..."
if lsof -ti:5000 &> /dev/null; then
    echo "   ⚠️  Port 5000 is in use"
    read -p "   Kill process on port 5000? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kill -9 $(lsof -ti:5000) 2>/dev/null
        echo "   ✅ Port 5000 cleared"
    fi
else
    echo "   ✅ Port 5000 is available"
fi

if lsof -ti:3000 &> /dev/null; then
    echo "   ⚠️  Port 3000 is in use"
    echo "   → npm will ask to use another port"
else
    echo "   ✅ Port 3000 is available"
fi

echo ""
echo "=========================================="
echo "✅ ALL CHECKS PASSED!"
echo "=========================================="
echo ""
echo "Now start servers:"
echo ""
echo "Terminal 1 (Backend):"
echo "  cd backend && python3 app.py"
echo ""
echo "Terminal 2 (Frontend):"
echo "  cd frontend && npm start"
echo ""
echo "Then open: http://localhost:3000/complete"
echo ""

