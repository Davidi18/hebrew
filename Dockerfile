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

# Explicitly prevent Pydantic v2 installation
RUN pip install --no-cache-dir "pydantic>=1.8.2,<2.0.0" --force-reinstall

RUN pip install --no-cache-dir -r requirements.txt

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
