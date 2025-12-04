#!/usr/bin/env python3
"""
Startup script for the Energy Management System Dashboard.
Launches the Flask backend server.
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent)  # Change to project root
    from backend.app import app
    print("=" * 80)
    print("Energy Management System Dashboard")
    print("=" * 80)
    print()
    print("Starting Flask backend server...")
    print("Access the dashboard at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print()
    app.run(debug=True, host='0.0.0.0', port=5000)

