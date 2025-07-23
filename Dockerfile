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
RUN pip install --no-cache-dir -r requirements.txt

# Fix HebSpacy installation - clean up any corrupted versions
RUN pip uninstall hebspacy -y || true
RUN pip cache purge
RUN pip install hebspacy==0.1.7 --force-reinstall --no-cache-dir

# Verify HebSpacy installation
RUN python -c "import hebspacy; print('HebSpacy version:', hebspacy.__version__); print('Has load method:', hasattr(hebspacy, 'load')); assert hasattr(hebspacy, 'load'), 'HebSpacy load method missing'"

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
