#!/bin/bash

# Enterprise Federated RAG - Debug Launcher

echo "🐛 Launching in DEBUG mode..."
export LOG_LEVEL=DEBUG

# Ensure log files exist so tail doesn't complain
touch worker.log api.log

# Start tailing logs in the background
echo "👀 Tailing logs (worker.log, api.log)..."
tail -f worker.log api.log &
TAIL_PID=$!

# Run the standard start script
./start_project.sh

# Cleanup: Kill the tail process when start_project.sh exits
echo "🛑 Stopping log monitoring..."
kill $TAIL_PID
