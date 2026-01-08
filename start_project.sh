#!/bin/bash

# Enterprise Federated RAG - Automation Script

echo "🛑 Stopping existing services and cleaning volumes..."
docker-compose down -v

echo "🚀 Starting infrastructure..."
docker-compose up -d

echo "⏳ Waiting for containers to stabilize (15s)..."
sleep 15

echo "🦙 Pulling Mistral model..."
docker exec ollama_service ollama pull mistral
docker exec ollama_service ollama pull nomic-embed-text

echo "🐍 Starting Python Worker (Background)..."
export LLM_PROVIDER=${LLM_PROVIDER:-ollama}
python worker.py > worker.log 2>&1 &
WORKER_PID=$!
echo "   PID: $WORKER_PID"

echo "🌐 Starting API (Background)..."
uvicorn main:app --port 8000 > api.log 2>&1 &
API_PID=$!
echo "   PID: $API_PID"

echo "⏳ Waiting for API to be ready (10s)..."
sleep 10

# Trap to kill background processes on exit
trap "kill $WORKER_PID $API_PID" EXIT

echo "🧪 Running Tests..."
./tests/run.sh

echo "✅ Done! (Press Ctrl+C to stop services)"
wait
