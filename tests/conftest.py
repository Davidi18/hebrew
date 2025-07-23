"""
Pytest configuration and fixtures for Hebrew Content Intelligence Service tests.
"""

import pytest
import asyncio
from typing import AsyncGenerator, Dict, Any
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient
from httpx import AsyncClient

# Import the main app
from app import app
from config.settings import settings
from models.hebrew_loader import hebrew_loader
from services.hebrew_analyzer import hebrew_analyzer
from services.semantic_clusters import clustering_service
from services.keyword_expander import keyword_expander
from services.search_data_service import search_data_service
from utils.cache_manager import cache_manager


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client for the FastAPI app."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture(scope="session")
async def mock_hebrew_model():
    """Mock Hebrew model for testing."""
    mock_model = MagicMock()
    
    # Mock model attributes
    mock_model.lang = "he"
    mock_model.meta = {
        "name": "hebrew-transformers",
        "version": "1.0.0",
        "description": "Hebrew Transformers model"
    }
    
    # Mock tokenization
    def mock_tokenize(text):
        tokens = text.split()
        mock_tokens = []
        for i, token_text in enumerate(tokens):
            mock_token = MagicMock()
            mock_token.text = token_text
            mock_token.lemma_ = token_text
            mock_token.pos_ = "NOUN"
            mock_token.tag_ = "NN"
            mock_token.is_alpha = token_text.isalpha()
            mock_token.is_stop = token_text in ["של", "את", "על", "עם"]
            mock_token.i = i
            mock_tokens.append(mock_token)
        return mock_tokens
    
    mock_model.return_value = mock_tokenize
    return mock_model


@pytest.fixture
def sample_hebrew_texts():
    """Sample Hebrew texts for testing."""
    return [
        "זהו טקסט בדיקה בעברית",
        "עיצוב מטבח מודרני עם פתרונות אחסון חכמים",
        "בית יפה עם חדרים גדולים ונוף מרהיב",
        "טכנולוגיה מתקדמת לעסקים קטנים ובינוניים",
        "שירותי ייעוץ עסקי מקצועיים בישראל"
    ]


@pytest.fixture
def sample_hebrew_keywords():
    """Sample Hebrew keywords for testing."""
    return [
        "עיצוב מטבח",
        "בית יפה",
        "טכנולוgiה",
        "ייעוץ עסקי",
        "פתרונות אחסון",
        "חדרים גדולים",
        "נוף מרהיב",
        "עסקים קטנים"
    ]


@pytest.fixture
def mock_analysis_result():
    """Mock Hebrew analysis result for testing."""
    return {
        "extracted_roots": {
            "roots": ["עצב", "מטבח", "בית", "יפה"],
            "root_frequencies": {"עצב": 2, "מטבח": 1, "בית": 3, "יפה": 1},
            "confidence_scores": {"עצב": 0.9, "מטבח": 0.95, "בית": 0.88, "יפה": 0.92}
        },
        "morphological_analysis": {
            "tokens": [
                {"text": "עיצוב", "lemma": "עיצוב", "pos": "NOUN", "features": {}},
                {"text": "מטבח", "lemma": "מטבח", "pos": "NOUN", "features": {}}
            ],
            "morphological_patterns": ["NOUN+NOUN"],
            "construct_chains": [["עיצוב", "מטבח"]]
        },
        "extracted_keywords": {
            "top_keywords": ["עיצוב מטבח", "פתרונות אחסון", "בית יפה"],
            "keyword_scores": {"עיצוב מטבח": 0.95, "פתרונות אחסון": 0.87, "בית יפה": 0.82},
            "total_keywords": 3
        },
        "content_themes": {
            "dominant_themes": ["עיצוב", "בית", "טכנולוגיה"],
            "theme_scores": {"עיצוב": 0.9, "בית": 0.85, "טכנולוגיה": 0.8},
            "theme_keywords": {
                "עיצוב": ["מטבח", "פתרונות", "אחסון"],
                "בית": ["חדרים", "נוף", "יפה"],
                "טכנולוגיה": ["מתקדמת", "עסקים"]
            }
        }
    }


@pytest.fixture
def mock_clustering_result():
    """Mock clustering result for testing."""
    return {
        "clusters": [
            {
                "id": "cluster_1",
                "name": "עיצוב ובית",
                "keywords": ["עיצוב מטבח", "בית יפה", "פתרונות אחסון"],
                "cluster_type": "theme_based",
                "score": 0.92,
                "metadata": {"dominant_theme": "עיצוב", "keyword_count": 3}
            },
            {
                "id": "cluster_2", 
                "name": "טכנולוגיה ועסקים",
                "keywords": ["טכנולוגיה מתקדמת", "ייעוץ עסקי", "עסקים קטנים"],
                "cluster_type": "semantic_similarity",
                "score": 0.88,
                "metadata": {"dominant_theme": "טכנולוגיה", "keyword_count": 3}
            }
        ],
        "metadata": {
            "total_clusters": 2,
            "clustering_method": "mixed",
            "average_cluster_score": 0.9
        }
    }


@pytest.fixture
def mock_keyword_expansion():
    """Mock keyword expansion result for testing."""
    return {
        "keyword_expansions": {
            "עיצוב מטבח": {
                "original": "עיצוב מטבח",
                "morphological_info": {
                    "root": "עצב",
                    "pattern": "פיעול",
                    "pos": "NOUN"
                },
                "variations_by_type": {
                    "morphological": ["עיצובי מטבח", "עיצוב מטבחים", "עיצוב המטבח"],
                    "semantic": ["תכנון מטבח", "עיצוב פנים מטבח", "מטבח מעוצב"],
                    "commercial": ["עיצוב מטבח מחיר", "עיצוב מטבח זול", "עיצוב מטבח איכותי"],
                    "locational": ["עיצוב מטבח תל אביב", "עיצוב מטבח ירושלים"],
                    "question_based": ["איך לעצב מטבח", "כמה עולה עיצוב מטבח"]
                },
                "all_variations": [
                    "עיצובי מטבח", "עיצוב מטבחים", "תכנון מטבח", 
                    "עיצוב מטבח מחיר", "עיצוב מטבח תל אביב"
                ],
                "expansion_score": 0.91
            }
        },
        "metadata": {
            "total_original_keywords": 1,
            "total_variations": 5,
            "expansion_ratio": 5.0
        }
    }


@pytest.fixture
def mock_search_volume_data():
    """Mock search volume data for testing."""
    return {
        "keywords": {
            "עיצוב מטבח": {
                "search_volume": 2400,
                "competition": 0.67,
                "cpc": 3.45,
                "difficulty": 45,
                "trends": [
                    {"month": 1, "search_volume": 2200},
                    {"month": 2, "search_volume": 2600}
                ],
                "related_keywords": ["תכנון מטבח", "מטבח מעוצב"]
            }
        },
        "metadata": {
            "total_keywords": 1,
            "location": "Israel",
            "language": "he",
            "mock_data": True
        }
    }


@pytest.fixture(autouse=True)
async def setup_test_environment():
    """Set up test environment with mocked services."""
    # Use memory cache for tests
    cache_manager.cache_type = "memory"
    cache_manager.memory_cache = {}
    
    # Initialize cache manager
    await cache_manager.initialize()
    
    yield
    
    # Cleanup after tests
    await cache_manager.clear_cache()


@pytest.fixture
async def mock_services():
    """Mock all services for isolated testing."""
    # Mock Hebrew loader
    hebrew_loader.load_model = AsyncMock()
    hebrew_loader.analyze_text = AsyncMock()
    
    # Mock Hebrew analyzer
    hebrew_analyzer.analyze_content = AsyncMock()
    
    # Mock clustering service
    clustering_service.generate_clusters = AsyncMock()
    
    # Mock keyword expander
    keyword_expander.expand_keywords = AsyncMock()
    
    # Mock search data service
    search_data_service.get_comprehensive_keyword_data = AsyncMock()
    
    yield {
        "hebrew_loader": hebrew_loader,
        "hebrew_analyzer": hebrew_analyzer,
        "clustering_service": clustering_service,
        "keyword_expander": keyword_expander,
        "search_data_service": search_data_service
    }


# Test data constants
TEST_HEBREW_TEXT = "זהו טקסט בדיקה בעברית עם מילים שונות ומגוונות"
TEST_KEYWORDS = ["עיצוב מטבח", "בית יפה", "טכנולוגיה מתקדמת"]
TEST_LOCATION = "Israel"
TEST_LANGUAGE = "he"
