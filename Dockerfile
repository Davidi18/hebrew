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

# AGGRESSIVE HebSpacy cleanup - remove all traces
RUN pip uninstall hebspacy -y || true
RUN pip uninstall hebspacy -y || true  # Run twice
RUN pip cache purge
RUN rm -rf /usr/local/lib/python3.9/site-packages/hebspacy*
RUN rm -rf /usr/local/lib/python3.9/site-packages/*hebspacy*
RUN rm -rf /root/.cache/pip
RUN find /usr/local/lib/python3.9/site-packages -name "*hebspacy*" -type d -exec rm -rf {} + || true
RUN find /usr/local/lib/python3.9/site-packages -name "*hebspacy*" -type f -delete || true

# Install HebSpacy from scratch with aggressive flags
RUN pip install --no-cache-dir --force-reinstall --no-deps hebspacy==0.1.7
RUN pip install --no-cache-dir spacy==3.2.2  # Ensure spacy is correct version

# Download and install HebSpacy model (he_ner_news_trf-3.2.1)
RUN pip install --no-cache-dir https://github.com/8400TheHealthNetwork/HebSpacy/releases/download/he_ner_news_trf-3.2.1/he_ner_news_trf-3.2.1-py3-none-any.whl

# Debug: Show what's actually installed
RUN echo "=== DEBUG: Checking HebSpacy installation ===" && \
    find /usr/local/lib/python3.9/site-packages -name "*hebspacy*" -ls && \
    pip list | grep -i hebspacy && \
    python -c "import sys; print('Python path:', sys.path)" && \
    python -c "import hebspacy; print('HebSpacy file:', hebspacy.__file__); print('Version:', hebspacy.__version__); print('Dir:', dir(hebspacy))"

# Verify HebSpacy installation and model
RUN python -c "import hebspacy; print('HebSpacy version:', hebspacy.__version__); print('Has load method:', hasattr(hebspacy, 'load')); assert hasattr(hebspacy, 'load'), 'HebSpacy load method missing'"
RUN python -c "import spacy; nlp = spacy.load('he_ner_news_trf'); doc = nlp('שלום עולם'); print('Model test successful:', [token.text for token in doc])"

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
