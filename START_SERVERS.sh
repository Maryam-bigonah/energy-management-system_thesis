#!/bin/bash

# Script to start both backend and frontend servers
# Usage: ./START_SERVERS.sh

echo "=========================================="
echo "Starting Energy Management System Servers"
echo "=========================================="

# Check if we're in the right directory
if [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    echo "Error: Must run from code directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Start Backend in background
echo ""
echo "1. Starting Backend Server..."
cd backend
python3 app.py &
BACKEND_PID=$!
cd ..

echo "Backend started (PID: $BACKEND_PID)"
echo "Backend URL: http://localhost:5000"
echo "Waiting 3 seconds for backend to start..."

sleep 3

# Check if backend is running
if curl -s http://localhost:5000/api/health > /dev/null 2>&1; then
    echo "✅ Backend is running!"
else
    echo "⚠️  Backend might not be ready yet"
fi

# Start Frontend
echo ""
echo "2. Starting Frontend Server..."
echo "Frontend will open in your browser automatically"
echo ""
cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

echo "Starting frontend..."
npm start

# Note: npm start will block, so we won't see this
echo ""
echo "=========================================="
echo "Servers are starting!"
echo "=========================================="
echo ""
echo "Backend: http://localhost:5000"
echo "Frontend: http://localhost:3000"
echo "Complete View: http://localhost:3000/complete"
echo ""
echo "To stop servers:"
echo "  - Press Ctrl+C in this terminal"
echo "  - Or kill backend: kill $BACKEND_PID"

