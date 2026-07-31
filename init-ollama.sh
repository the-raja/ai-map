#!/bin/sh
echo "Waiting for Ollama service to start..."
while ! curl -s http://ollama:11434/api/tags > /dev/null; do
    sleep 2
done

echo "Ollama is online! Checking and pulling required AI models..."

echo "Pulling embedding model: nomic-embed-text..."
curl -s -X POST http://ollama:11434/api/pull -d '{"name": "nomic-embed-text", "stream": false}'

echo "\nPulling generative model: llama3.2..."
curl -s -X POST http://ollama:11434/api/pull -d '{"name": "llama3.2", "stream": false}'

echo "\nAll AI models downloaded successfully!"
