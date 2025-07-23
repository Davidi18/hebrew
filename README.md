# Hebrew Content Intelligence Service 

Advanced Hebrew content analysis service using HebSpacy, semantic clustering, and DataForSEO integration. This service provides comprehensive Hebrew NLP capabilities including keyword expansion, semantic clustering, and content analysis.

## Features

- **Hebrew NLP Analysis** - Advanced Hebrew text processing using HebSpacy
- **Keyword Expansion** - Morphological and semantic Hebrew keyword variations
- **Semantic Clustering** - Group related Hebrew keywords intelligently
- **Search Volume Data** - Integration with DataForSEO for keyword metrics
- **Caching System** - Redis/Memory caching for optimal performance
- **Health Monitoring** - Comprehensive health checks and monitoring

## Architecture

- **FastAPI** - Modern async web framework
- **HebSpacy** - Hebrew NLP model for entity recognition and analysis
- **Pydantic** - Data validation and serialization
- **Redis** - Caching and session management
- **Docker** - Containerized deployment

## Installation & Setup

### Docker (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd hebrew

# Build and run with Docker
docker build -t hebrew-intelligence .
docker run -p 5000:5000 hebrew-intelligence
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DEBUG=true
export LOG_LEVEL=INFO

# Run the application
python app.py
```

## Configuration

The service uses environment variables for configuration:

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVICE_NAME` | `hebrew-content-intelligence` | Service name |
| `VERSION` | `1.0.0` | Service version |
| `DEBUG` | `false` | Enable debug mode |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `5000` | Server port |
| `HEBSPACY_MODEL` | `he` | HebSpacy model identifier |
| `CACHE_TYPE` | `memory` | Cache type (memory/redis/disk) |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `DATAFORSEO_API_URL` | - | DataForSEO API endpoint |
| `DATAFORSEO_API_KEY` | - | DataForSEO API key |

## API Documentation

### Base URL
```
http://localhost:5000
```

### Authentication
Currently no authentication required. Configure as needed for production.

---

## Health Endpoints

### 1. Health Check
**GET** `/health/`

Basic health check with service status and model information.

```bash
curl -X GET "http://localhost:5000/health/"
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-23T14:30:00Z",
  "version": "1.0.0",
  "model_status": {
    "loaded": true,
    "model_name": "he_ner_news_trf",
    "version": "3.2.1"
  },
  "uptime_seconds": 3600.5
}
```

### 2. Readiness Check
**GET** `/health/ready`

Check if service is ready to accept requests.

```bash
curl -X GET "http://localhost:5000/health/ready"
```

**Response:**
```json
{
  "status": "ready",
  "message": "Service is ready to accept requests"
}
```

### 3. Liveness Check
**GET** `/health/live`

Check if service is alive (used by Docker health checks).

```bash
curl -X GET "http://localhost:5000/health/live"
```

**Response:**
```json
{
  "status": "alive",
  "message": "Service is running"
}
```

---

## Keyword Expansion Endpoints

### 1. Expand Keywords
**POST** `/api/keywords/expand`

Generate morphological and semantic variations of Hebrew keywords.

```bash
curl -X POST "http://localhost:5000/api/keywords/expand" \
  -H "Content-Type: application/json" \
  -d '{
    "keywords": ["בית", "מכונית", "עבודה"],
    "options": {
      "include_morphological": true,
      "include_semantic": true,
      "include_commercial": true,
      "max_variations": 20
    }
  }'
```

**Response:**
```json
{
  "results": [
    {
      "original_keyword": "בית",
      "variations": [
        {
          "keyword": "בתים",
          "type": "morphological",
          "confidence": 0.95,
          "variation_type": "plural"
        },
        {
          "keyword": "בית משפחה",
          "type": "semantic",
          "confidence": 0.88,
          "variation_type": "compound"
        },
        {
          "keyword": "דירה",
          "type": "semantic",
          "confidence": 0.82,
          "variation_type": "synonym"
        }
      ],
      "total_variations": 15
    }
  ],
  "metadata": {
    "processing_time": 0.45,
    "total_keywords": 3,
    "total_variations": 42,
    "cached": false
  }
}
```

### 2. Search Volume Data
**POST** `/api/search-volume`

Get search volume and competition data for Hebrew keywords.

```bash
curl -X POST "http://localhost:5000/api/search-volume" \
  -H "Content-Type: application/json" \
  -d '{
    "keywords": ["בית", "מכונית", "עבודה"],
    "location": "Israel",
    "language": "he"
  }'
```

**Response:**
```json
{
  "volumes": [
    {
      "keyword": "בית",
      "search_volume": 12000,
      "competition": 0.65,
      "cpc": 2.3,
      "difficulty": 45,
      "trends": [8500, 9200, 11000, 12000]
    },
    {
      "keyword": "מכונית",
      "search_volume": 8500,
      "competition": 0.72,
      "cpc": 3.1,
      "difficulty": 52,
      "trends": [7200, 8100, 8800, 8500]
    }
  ],
  "metadata": {
    "processing_time": 1.2,
    "data_source": "DataForSEO",
    "location": "Israel",
    "language": "he",
    "cached": true
  }
}
```

### 3. Expansion Methods
**GET** `/api/keywords/expansion-methods`

Get available keyword expansion methods and their descriptions.

```bash
curl -X GET "http://localhost:5000/api/keywords/expansion-methods"
```

**Response:**
```json
{
  "methods": [
    {
      "name": "morphological",
      "description": "Hebrew morphological variations (plural, construct, possessive)",
      "examples": ["בית → בתים, בית של, בתי"]
    },
    {
      "name": "semantic",
      "description": "Semantic synonyms and related terms",
      "examples": ["בית → דירה, מגורים, נכס"]
    },
    {
      "name": "commercial",
      "description": "Commercial intent variations",
      "examples": ["בית → קניית בית, מכירת בית, השכרת בית"]
    },
    {
      "name": "locational",
      "description": "Israeli geographic variations",
      "examples": ["בית → בית בתל אביב, בית בירושלים"]
    }
  ]
}
```

### 4. Hebrew Roots Analysis
**GET** `/api/keywords/hebrew-roots`

Analyze Hebrew root patterns in keywords.

```bash
curl -X GET "http://localhost:5000/api/keywords/hebrew-roots?keywords=כתב,כתיבה,כותב"
```

**Response:**
```json
{
  "roots": [
    {
      "root": "כ-ת-ב",
      "keywords": ["כתב", "כתיבה", "כותב"],
      "meaning": "writing, inscription",
      "pattern_analysis": {
        "verb_forms": ["כתב", "כותב"],
        "noun_forms": ["כתיבה"],
        "related_concepts": ["מכתב", "כתובת", "כתב יד"]
      }
    }
  ]
}
```

---

## Semantic Clustering Endpoints

### 1. Generate Clusters
**POST** `/api/clusters`

Group related Hebrew keywords into semantic clusters.

```bash
curl -X POST "http://localhost:5000/api/clusters" \
  -H "Content-Type: application/json" \
  -d '{
    "keywords": [
      "בית", "דירה", "מגורים", "נכס",
      "מכונית", "רכב", "אוטו", "משאית",
      "עבודה", "משרה", "קריירה", "מקצוע"
    ],
    "options": {
      "max_clusters": 5,
      "min_cluster_size": 2,
      "similarity_threshold": 0.7
    }
  }'
```

**Response:**
```json
{
  "clusters": [
    {
      "id": 1,
      "name": "נדלן ומגורים",
      "keywords": [
        {
          "keyword": "בית",
          "confidence": 0.95,
          "centrality": 0.88
        },
        {
          "keyword": "דירה",
          "confidence": 0.92,
          "centrality": 0.85
        },
        {
          "keyword": "מגורים",
          "confidence": 0.89,
          "centrality": 0.82
        },
        {
          "keyword": "נכס",
          "confidence": 0.87,
          "centrality": 0.79
        }
      ],
      "size": 4,
      "coherence_score": 0.91
    },
    {
      "id": 2,
      "name": "כלי רכב",
      "keywords": [
        {
          "keyword": "מכונית",
          "confidence": 0.94,
          "centrality": 0.89
        },
        {
          "keyword": "רכב",
          "confidence": 0.91,
          "centrality": 0.86
        },
        {
          "keyword": "אוטו",
          "confidence": 0.88,
          "centrality": 0.83
        }
      ],
      "size": 3,
      "coherence_score": 0.89
    }
  ],
  "metadata": {
    "processing_time": 0.75,
    "total_clusters": 3,
    "total_keywords": 12,
    "algorithm": "hebrew_semantic_clustering",
    "cached": false
  }
}
```

### 2. Clustering Methods
**GET** `/api/clusters/methods`

Get available clustering methods and their parameters.

```bash
curl -X GET "http://localhost:5000/api/clusters/methods"
```

**Response:**
```json
{
  "methods": [
    {
      "name": "hebrew_semantic",
      "description": "Hebrew-aware semantic clustering using root analysis",
      "parameters": {
        "similarity_threshold": "float (0.0-1.0)",
        "max_clusters": "int (1-20)",
        "min_cluster_size": "int (2-10)"
      }
    },
    {
      "name": "morphological",
      "description": "Clustering based on Hebrew morphological patterns",
      "parameters": {
        "root_weight": "float (0.0-1.0)",
        "pattern_weight": "float (0.0-1.0)"
      }
    }
  ]
}
```

---

## Content Analysis Endpoints

### 1. Analyze Content
**POST** `/api/analyze`

Comprehensive Hebrew content analysis with keyword extraction and insights.

```bash
curl -X POST "http://localhost:5000/api/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "בית חדש למכירה בתל אביב. הדירה כוללת 4 חדרים, מרפסת גדולה ומקום חניה. המחיר 2.5 מיליון שקל. צרו קשר לפרטים נוספים.",
    "url": "https://example.com/property-listing",
    "options": {
      "include_volumes": true,
      "include_clusters": true,
      "include_expansion": true,
      "max_keywords": 20
    }
  }'
```

**Response:**
```json
{
  "analysis": {
    "keywords": [
      {
        "keyword": "בית",
        "frequency": 1,
        "importance": 0.92,
        "entities": ["PRODUCT"],
        "search_volume": 12000,
        "competition": 0.65
      },
      {
        "keyword": "דירה",
        "frequency": 1,
        "importance": 0.89,
        "entities": ["PRODUCT"],
        "search_volume": 8500,
        "competition": 0.58
      },
      {
        "keyword": "תל אביב",
        "frequency": 1,
        "importance": 0.85,
        "entities": ["GPE"],
        "search_volume": 45000,
        "competition": 0.72
      }
    ],
    "entities": [
      {
        "text": "תל אביב",
        "label": "GPE",
        "confidence": 0.99,
        "start": 25,
        "end": 32
      },
      {
        "text": "2.5 מיליון שקל",
        "label": "MONEY",
        "confidence": 0.95,
        "start": 95,
        "end": 109
      }
    ],
    "clusters": [
      {
        "name": "נדלן",
        "keywords": ["בית", "דירה", "מכירה", "חדרים"],
        "relevance": 0.94
      }
    ],
    "sentiment": {
      "polarity": 0.1,
      "subjectivity": 0.3,
      "classification": "neutral"
    }
  },
  "metadata": {
    "processing_time": 1.8,
    "text_length": 142,
    "language": "he",
    "model_version": "he_ner_news_trf-3.2.1"
  }
}
```

---

## Batch Processing

### Batch Analysis
**POST** `/api/batch-analyze`

Analyze multiple texts in a single request.

```bash
curl -X POST "http://localhost:5000/api/batch-analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "טקסט ראשון לניתוח",
      "טקסט שני לניתוח",
      "טקסט שלישי לניתוח"
    ],
    "options": {
      "include_volumes": false,
      "max_keywords": 10
    }
  }'
```

---

## Performance & Caching

The service includes intelligent caching to optimize performance:

- **Memory Cache** - Fast in-memory caching for frequent requests
- **Redis Cache** - Distributed caching for scaled deployments
- **Model Caching** - HebSpacy model results are cached
- **API Response Caching** - DataForSEO responses are cached

### Cache Headers

Responses include cache information:
```
X-Cache-Status: HIT|MISS
X-Cache-TTL: 3600
X-Processing-Time: 0.45
```

## Error Handling

The API returns structured error responses:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "field": "keywords",
      "issue": "List cannot be empty"
    }
  },
  "request_id": "req_123456789",
  "timestamp": "2025-01-23T14:30:00Z"
}
```

### Common Error Codes

| Code | Description |
|------|-------------|
| `VALIDATION_ERROR` | Request validation failed |
| `MODEL_ERROR` | HebSpacy model error |
| `CACHE_ERROR` | Caching system error |
| `EXTERNAL_API_ERROR` | DataForSEO API error |
| `RATE_LIMIT_ERROR` | Rate limit exceeded |

## Monitoring & Logging

### Structured Logging
All requests are logged with structured data:
```json
{
  "timestamp": "2025-01-23T14:30:00Z",
  "level": "INFO",
  "message": "Request processed",
  "request_id": "req_123456789",
  "endpoint": "/api/keywords/expand",
  "processing_time": 0.45,
  "cache_hit": true
}
```

### Metrics
Key metrics are tracked:
- Request latency
- Cache hit rates
- Model inference time
- Error rates
- Throughput

## Docker Deployment

### Production Docker Setup

```dockerfile
# Multi-stage build for optimized production image
FROM python:3.9-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.9-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  hebrew-intelligence:
    build: .
    ports:
      - "5000:5000"
    environment:
      - REDIS_URL=redis://redis:6379/0
      - DEBUG=false
    depends_on:
      - redis
    
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

## Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For support and questions:
- Create an issue on GitHub
- Check the API documentation at `/docs`
- Review the health endpoints for service status

---

**Made with ❤️ for Hebrew NLP**
