#!/bin/bash

echo "🛑 Stopping Project..."

# 1. Kill the Master Script (start_project.sh)
# We do this first to stop it from spawning new things or waiting.
echo "🔪 Killing start scripts..."
pkill -f "start_project.sh" || echo "No start script running."

# 2. Kill Application Processes (Worker and API)
echo "🔪 Killing application processes..."
pkill -f "python worker.py" || echo "No worker running."
pkill -f "uvicorn main:app" || echo "No API running."

# 3. Stop Docker Services
echo "🐳 Stopping Docker services..."
# We use 'down' to stop and remove containers/networks. 
# We do NOT use '-v' here by default to preserve data, but users can add it manually if they want a wipe.
docker-compose down

# 4. Cleanup Artifacts (optional)
echo "🧹 Cleaning up temporary files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete

echo "✅ All processes terminated and services stopped."
