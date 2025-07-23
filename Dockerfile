FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for HebSpacy and compilation
RUN apt-get update && apt-get install -y \
    gcc g++ \
    curl \
    git \
    pkg-config \
    cmake \
    make \
    build-essential \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (required for tokenizers)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip

# Force install exact Pydantic v1 and prevent any upgrades
RUN pip install --no-cache-dir --no-deps pydantic==1.10.18
RUN pip install --no-cache-dir --no-deps typing-extensions==4.8.0

# Install core web framework dependencies with --no-deps
RUN pip install --no-cache-dir --no-deps fastapi==0.85.0
RUN pip install --no-cache-dir --no-deps starlette==0.20.4
RUN pip install --no-cache-dir --no-deps uvicorn==0.18.3

# Install spaCy and hebspacy with --no-deps (most critical)
RUN pip install --no-cache-dir --no-deps spacy==3.2.2
RUN pip install --no-cache-dir --no-deps hebspacy==0.1.7

# Install tokenizers with exact version (pre-built wheel)
RUN pip install --no-cache-dir --no-deps tokenizers==0.13.3

# Install HTTP/API dependencies
RUN pip install --no-cache-dir --no-deps httpx==0.25.2
RUN pip install --no-cache-dir --no-deps aiohttp==3.8.6

# Install data processing dependencies
RUN pip install --no-cache-dir --no-deps pandas==2.0.3
RUN pip install --no-cache-dir --no-deps numpy==1.24.4

# Install caching
RUN pip install --no-cache-dir --no-deps redis==4.6.0

# Install utilities
RUN pip install --no-cache-dir --no-deps python-dotenv==1.0.1
RUN pip install --no-cache-dir --no-deps loguru==0.7.3
RUN pip install --no-cache-dir --no-deps PyYAML==6.0.2

# Install testing dependencies
RUN pip install --no-cache-dir --no-deps pytest==7.4.4
RUN pip install --no-cache-dir --no-deps pytest-asyncio==0.21.2

# Install any missing sub-dependencies that are safe
RUN pip install --no-cache-dir click h11 anyio sniffio idna certifi charset-normalizer

# Download HebSpacy model
RUN pip install --no-cache-dir https://github.com/8400TheHealthNetwork/HebSpacy/releases/download/he_ner_news_trf-3.2.1/he_ner_news_trf-3.2.1-py3-none-any.whl

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models/model_cache logs cache

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health/live || exit 1

# Run the application
CMD ["python", "app.py"]
