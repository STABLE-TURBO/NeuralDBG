# Dockerfile for Neural DSL Backend Bridge

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[api]"

# Expose port
EXPOSE 8000

# Set environment variables
ENV NEURAL_HOST=0.0.0.0
ENV NEURAL_PORT=8000
ENV NEURAL_LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run the server
CMD ["python", "-m", "neural.aquarium.backend.run"]
