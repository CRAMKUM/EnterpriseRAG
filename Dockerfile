# ================================================================================
# Multi-stage Dockerfile for Enterprise RAG System
# Optimized for Cloud Run deployment with caching best practices
# ================================================================================

# Define build argument for base image (allows override)
ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE} AS base

# Set working directory
WORKDIR /app

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=True

# Install system dependencies for OpenCV, Tesseract, PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements.txt first for docker layer caching
COPY requirements.txt /app/

# Install Python dependencies from public PyPI
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories with proper structure
RUN mkdir -p /app/data /app/logs /app/temp /app/.streamlit

# Copy code
COPY . /app/

# ==============================================================================
# STAGE 2: Runtime Image (minimal, secure)
# ==============================================================================
FROM base AS runtime

ENV HOME=/app

# Switch ownership to non-root user (uid 1000)
RUN chown -R 1000:1000 /app
USER 1000:1000

EXPOSE 8080

# Environment Variables
ENV GOOGLE_GENAI_USE_VERTEXAI="True"
ENV GOOGLE_CLOUD_LOCATION="europe-west4"
ENV GOOGLE_CLOUD_PROJECT="project-0f2740f8-acd1-43a1-a04"

ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV COOKIE_PASSWORD="enterprise_rag_secure_key_2026_v1"
ENV CONFIG_BUCKET_NAME="enterprise-rag-storage-avathon"

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/main.py", \
            "--server.port=8080",\
            "--server.address=0.0.0.0",\
            "--browser.gatherUsageStats=false"]
