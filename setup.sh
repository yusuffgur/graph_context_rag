#!/bin/bash
echo "⏳ Waiting for Ollama Service..."
until curl -s http://localhost:11434/api/tags > /dev/null; do
    sleep 2
done

echo "⬇️ Pulling Mistral Model (Small LLM)..."
docker exec ollama_service ollama pull mistral

echo "✅ Environment Ready!"