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
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Remove HebSpacy from requirements to avoid conflicts
RUN pip uninstall hebspacy -y || true

# Install Transformers for advanced Hebrew NLP
RUN pip install --no-cache-dir transformers torch sentencepiece

# Note: Hebrew models (heBERT) will be downloaded at runtime to avoid build-time issues
# This approach is more flexible and avoids PyTorch security restrictions during build

# Verify transformers and torch installation
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); \
    import transformers; print(f'Transformers version: {transformers.__version__}'); \
    print('Hebrew Transformers stack ready!')"

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
