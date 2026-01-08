#!/bin/bash

# Enterprise Federated RAG - Infrastructure Launcher (Debug Mode)

echo "🚀 Starting infrastructure (Docker only)..."
docker-compose up -d

echo "⏳ Waiting for containers to stabilize (15s)..."
sleep 15

echo "🦙 Ensuring Ollama models are ready..."
docker exec ollama_service ollama pull mistral
docker exec ollama_service ollama pull nomic-embed-text

echo "✅ Infrastructure Ready!"
echo ""
echo "You can now run Python services interactively in your IDE."
echo "Examples:"
echo "  python worker.py"
echo "  uvicorn main:app --port 8000 --reload"
echo ""
echo "Set breakpoints as needed. Run './stop_project.sh' when you are done to tear down services."
