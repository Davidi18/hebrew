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

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip

# Force install exact Pydantic v1 and prevent any upgrades
RUN pip install --no-cache-dir --no-deps pydantic==1.10.18
RUN pip install --no-cache-dir --no-deps typing-extensions==4.8.0

# Install core dependencies with --no-deps to prevent conflicts
RUN pip install --no-cache-dir --no-deps fastapi==0.85.0
RUN pip install --no-cache-dir --no-deps starlette==0.20.4
RUN pip install --no-cache-dir --no-deps uvicorn==0.18.3

# Install spaCy and hebspacy with --no-deps
RUN pip install --no-cache-dir --no-deps spacy==3.2.2
RUN pip install --no-cache-dir --no-deps hebspacy==0.1.7

# Now install remaining dependencies normally (they shouldn't conflict)
RUN pip install --no-cache-dir -r requirements.txt --no-deps || true
RUN pip install --no-cache-dir tokenizers==0.13.3 httpx aiohttp pandas numpy redis python-dotenv loguru PyYAML pytest pytest-asyncio

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
