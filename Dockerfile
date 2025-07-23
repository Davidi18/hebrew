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

# Remove HebSpacy from requirements to avoid conflicts
RUN pip uninstall hebspacy -y || true

# Install Transformers for advanced Hebrew NLP
RUN pip install --no-cache-dir transformers torch sentencepiece

# Download and cache Hebrew models (heBERT and AlephBERT)
RUN python -c "
from transformers import AutoTokenizer, AutoModel, pipeline
import torch

print('Downloading heBERT model...')
tokenizer = AutoTokenizer.from_pretrained('avichr/heBERT')
model = AutoModel.from_pretrained('avichr/heBERT')
print('heBERT downloaded successfully')

print('Downloading AlephBERT model...')
tokenizer_aleph = AutoTokenizer.from_pretrained('onlplab/alephbert-base')
model_aleph = AutoModel.from_pretrained('onlplab/alephbert-base')
print('AlephBERT downloaded successfully')

print('Testing Hebrew NER pipeline...')
ner_pipeline = pipeline('ner', model='avichr/heBERT_NER', tokenizer='avichr/heBERT_NER')
result = ner_pipeline('שלום, אני דוד מתל אביב')
print('NER test successful:', result)
"

# Verify Transformers Hebrew installation
RUN python -c "
import transformers
import torch
print('Transformers version:', transformers.__version__)
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('Hebrew Transformers setup complete!')
"

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
