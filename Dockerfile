# Use a lightweight, official Python runtime
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy project files into the container
COPY main.py .
COPY index.html .

# Expose port 8080 for the VectorDB server
EXPOSE 8080

# Command to run the application
CMD ["python", "main.py"]
