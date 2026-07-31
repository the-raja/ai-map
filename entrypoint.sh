#!/bin/bash
set -e

echo "=== Starting AI-Map All-in-One Container ==="

export OLLAMA_HOST="127.0.0.1:11434"

echo "[1/3] Starting Ollama background daemon..."
ollama serve &

echo "[2/3] Waiting for Ollama API..."
until curl -s http://127.0.0.1:11434/api/tags > /dev/null; do
    sleep 1
done

echo "Ollama API is online!"

if ! curl -s http://127.0.0.1:11434/api/tags | grep -q "nomic-embed-text"; then
    echo "Pulling embedding model (nomic-embed-text)..."
    ollama pull nomic-embed-text
fi

if ! curl -s http://127.0.0.1:11434/api/tags | grep -q "llama3.2"; then
    echo "Pulling generative LLM model (llama3.2)..."
    ollama pull llama3.2
fi

echo "[3/3] AI Models ready! Launching AI-Map Server on http://localhost:8080..."
exec python main.py
