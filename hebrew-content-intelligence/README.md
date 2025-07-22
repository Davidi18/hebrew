# 🧠 Hebrew Content Intelligence Service

Advanced Hebrew content analysis service using HebSpacy, semantic clustering, and DataForSEO integration.

## 🎯 Features

- **Hebrew NLP Analysis**: Morphological analysis, root extraction, and NER using HebSpacy
- **Semantic Clustering**: Intelligent grouping of related Hebrew concepts
- **Keyword Expansion**: Generate Hebrew keyword variations and morphological forms
- **Search Intelligence**: Integration with DataForSEO for volume and difficulty data
- **Content Gap Analysis**: Identify missing content opportunities
- **Business Insights**: SEO recommendations for Hebrew content

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Redis (for caching)
- Docker & Docker Compose

### Installation

1. Clone and setup:
```bash
git clone <repo-url>
cd hebrew-content-intelligence
cp .env.example .env
```

2. Using Docker:
```bash
docker-compose up --build
```

3. Or local development:
```bash
pip install -r requirements.txt
python scripts/download_models.py
uvicorn app:app --reload --port 5000
```

### Usage

```bash
# Analyze Hebrew content
curl -X POST "http://localhost:5000/api/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "עיצוב מטבח מודרני עם פתרונות אחסון חכמים"}'
```

## 📁 Project Structure

```
hebrew-content-intelligence/
├── app.py                     # FastAPI main application
├── config/                    # Configuration management
├── services/                  # Core business logic
├── utils/                     # Helper utilities
├── models/                    # HebSpacy model management
├── api/                       # FastAPI routes & middleware
├── schemas/                   # Pydantic models
├── tests/                     # Testing suite
└── scripts/                   # Utility scripts
```

## 🔧 Configuration

Key environment variables:
- `HEBSPACY_MODEL`: Hebrew model name (default: 'he')
- `DATAFORSEO_API_URL`: DataForSEO MCP endpoint
- `REDIS_URL`: Redis connection string
- `CACHE_TTL`: Cache expiration time

## 📊 API Endpoints

- `POST /api/analyze` - Analyze Hebrew content
- `POST /api/clusters` - Generate semantic clusters
- `POST /api/keywords` - Expand Hebrew keywords
- `GET /health` - Service health check

## 🧪 Testing

```bash
pytest tests/
python scripts/benchmark.py
```

## 🐳 Deployment

Production deployment with Docker Compose includes Redis caching and health checks.

## 📝 License

MIT License
