# Anti-Fraud Detection System - Complete Deployment
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    openjdk-11-jre-headless \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME for Spark
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Set Spark environment
ENV SPARK_HOME=/opt/spark
ENV PYSPARK_PYTHON=python3

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data logs models feature_store

# Expose ports for services
EXPOSE 8000 8501 5000 4040

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Default command - Start API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
