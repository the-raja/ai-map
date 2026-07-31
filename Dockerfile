# Use an official lightweight Python runtime
FROM python:3.11-slim

# Install system dependencies (curl, ca-certificates, procps, zstd) needed for Ollama
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    procps \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# Install official Ollama binary
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory inside container
WORKDIR /app

# Prevent Python from writing .pyc files and buffer logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy application files
COPY main.py .
COPY index.html .
COPY entrypoint.sh .

# Ensure entrypoint script is executable
RUN chmod +x entrypoint.sh

# Expose ports: 8080 (AI-Map Web UI) and 11434 (Ollama)
EXPOSE 8080 11434

# Set entrypoint to run Ollama + AI-Map all-in-one
ENTRYPOINT ["/app/entrypoint.sh"]
