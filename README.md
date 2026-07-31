# VectorDB — Vector Database in Python

A fully working **Vector Database** built from scratch in **Python** (using standard library) with an interactive web UI.  
Implements **HNSW**, **KD-Tree**, and **Brute Force** search algorithms side-by-side, plus a **RAG pipeline** powered by a local LLM via Ollama.

> Converted from the original C++ VectorDB implementation into clean, idiomatic Python.

---

## Features

| Feature | Description |
|---|---|
| **3 Search Algorithms** | HNSW (production-grade graph search), KD-Tree, Brute Force — run all three and compare latency |
| **3 Distance Metrics** | Cosine similarity, Euclidean distance, Manhattan distance |
| **16D Demo Vectors** | 20 pre-loaded semantic vectors across 4 categories (CS, Math, Food, Sports) |
| **2D PCA Scatter Plot** | Live visualization of semantic space — watch clusters form |
| **Real Document Embedding** | Paste any text → Ollama embeds it with `nomic-embed-text` (768D) |
| **RAG Pipeline** | Ask questions about your documents → HNSW retrieves context → local LLM answers via Ollama (`llama3.2`) |
| **Full REST API** | CRUD endpoints: search, insert, delete, benchmark, hnsw-info, doc management, RAG |
| **Zero Dependencies** | Built using Python standard library (`http.server`, `heapq`, `urllib`, `threading`, `dataclasses`) |

---

## How It Works

```
Your Text
    │
    ▼
Ollama (nomic-embed-text)          ← converts text to a 768-dimensional vector
    │
    ▼
HNSW Index (Python)                ← indexes the vector in a multilayer graph
    │
    ▼
Semantic Search                    ← finds nearest neighbors in vector space
    │
    ▼
Ollama (llama3.2)                  ← reads retrieved chunks, generates an answer
    │
    ▼
Answer
```

---

## Prerequisites

1. **Python 3.7+** (No external packages required!)
2. **Ollama** (optional, for real document embeddings & RAG pipeline)
   - Download from: [https://ollama.com](https://ollama.com)
   - Pull embedding model: `ollama pull nomic-embed-text`
   - Pull LLM model: `ollama pull llama3.2`

---

## Quick Start

### 1. Run the Python Server

In your terminal / PowerShell:

```bash
python main.py
```

Output:
```
=== VectorDB Engine (Python) ===
http://localhost:8080
20 demo vectors | 16 dims | HNSW+KD-Tree+BruteForce
Ollama: ONLINE
  embed model: nomic-embed-text  gen model: llama3.2
```

### 2. Open Web Visualizer

Open your browser and visit:
```
http://localhost:8080
```

---

## REST API Specification

### Demo Vector Endpoints
- `GET /search?v=...&k=5&metric=cosine&algo=hnsw` — Vector k-NN search
- `POST /insert` — Insert a new demo vector (`{"metadata":"...","category":"...","embedding":[...]}`)
- `DELETE /delete/{id}` — Delete a vector by ID
- `GET /items` — List all demo vectors
- `GET /benchmark?v=...&k=5&metric=cosine` — Compare latency across HNSW, KD-Tree, and Brute Force
- `GET /hnsw-info` — Inspection data for multi-layer graph visualization

### Document & RAG Endpoints
- `POST /doc/insert` — Insert and chunk a document (`{"title":"...","text":"..."}`)
- `DELETE /doc/delete/{id}` — Delete a document chunk
- `GET /doc/list` — List all document chunks
- `POST /doc/search` — Perform semantic retrieval on docs (`{"question":"...","k":3}`)
- `POST /doc/ask` — Full RAG generation pipeline (`{"question":"...","k":3}`)
- `GET /status` — System health & Ollama connection status
