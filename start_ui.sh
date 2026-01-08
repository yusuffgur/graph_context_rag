#!/bin/bash

echo "📦 Installing Dependencies (if missing)..."
pip install -r requirements.txt > /dev/null 2>&1

echo "🚀 Starting Infrastructure..."
docker-compose up -d

echo "⏳ Waiting for containers..."
sleep 10

echo "🦙 Ensuring Ollama models are ready..."
docker exec ollama_service ollama pull mistral
docker exec ollama_service ollama pull nomic-embed-text

echo "🐍 Starting Backend Services..."
# Start Worker
python worker.py > worker.log 2>&1 &
WORKER_PID=$!

# Start API
uvicorn main:app --port 8000 > api.log 2>&1 &
API_PID=$!

echo "⏳ Waiting for API..."
sleep 5

echo "🎨 Starting Gradio UI..."
echo "👉 Open http://localhost:7860 in your browser"

# Run UI in foreground
# Run UI in foreground (captured)
python ui.py > ui.log 2>&1

# Cleanup on exit
kill $WORKER_PID $API_PID
