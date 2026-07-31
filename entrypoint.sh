#!/bin/bash
set -e

echo "=== Starting AI-Map All-in-One Container ==="

# 1. Start Ollama service in the background
echo "[1/3] Starting Ollama background daemon..."
ollama serve &

# 2. Wait for Ollama service to respond
echo "[2/3] Waiting for Ollama API..."
until curl -s http://127.0.0.1:11434/api/tags > /dev/null; do
    sleep 1
done

echo "Ollama API is online!"

# Check if nomic-embed-text model exists, if not pull it
if ! curl -s http://127.0.0.1:11434/api/tags | grep -q "nomic-embed-text"; then
    echo "Pulling embedding model (nomic-embed-text)..."
    ollama pull nomic-embed-text
fi

# Check if llama3.2 model exists, if not pull it
if ! curl -s http://127.0.0.1:11434/api/tags | grep -q "llama3.2"; then
    echo "Pulling generative LLM model (llama3.2)..."
    ollama pull llama3.2
fi

echo "[3/3] AI Models ready! Launching AI-Map Server on http://localhost:8080..."

# 3. Start AI-Map Python server in foreground
exec python main.py
