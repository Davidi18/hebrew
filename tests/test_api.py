"""
API endpoint tests for Hebrew Content Intelligence Service.
"""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi import status
from httpx import AsyncClient

from tests.conftest import (
    TEST_HEBREW_TEXT, TEST_KEYWORDS, TEST_LOCATION, TEST_LANGUAGE
)


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    async def test_health_check(self, async_client: AsyncClient):
        """Test basic health check."""
        response = await async_client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "service" in data
        assert "version" in data
        assert "timestamp" in data
    
    async def test_readiness_check(self, async_client: AsyncClient):
        """Test readiness check."""
        with patch("models.hebspacy_loader.hebrew_loader.get_model_info") as mock_info:
            mock_info.return_value = {"model_loaded": True, "model_name": "he_core_news_lg"}
            
            response = await async_client.get("/health/ready")
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert data["ready"] is True
            assert "checks" in data
    
    async def test_liveness_check(self, async_client: AsyncClient):
        """Test liveness check."""
        response = await async_client.get("/health/live")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["alive"] is True
        assert "uptime_seconds" in data


class TestAnalysisEndpoints:
    """Test content analysis endpoints."""
    
    async def test_analyze_content_single(self, async_client: AsyncClient, mock_services, mock_analysis_result):
        """Test single content analysis."""
        # Mock the Hebrew analyzer
        mock_services["hebrew_analyzer"].analyze_content.return_value = mock_analysis_result
        mock_services["clustering_service"].generate_clusters.return_value = {
            "clusters": [], "metadata": {}
        }
        mock_services["keyword_expander"].expand_keywords.return_value = {
            "keyword_expansions": {}, "metadata": {}
        }
        
        request_data = {
            "text": TEST_HEBREW_TEXT,
            "options": {
                "include_roots": True,
                "include_morphology": True,
                "include_keywords": True,
                "include_themes": True
            }
        }
        
        response = await async_client.post("/analysis/analyze", json=request_data)
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "analysis" in data
        assert "clusters" in data
        assert "keyword_expansions" in data
        assert "metadata" in data
        
        # Verify analysis structure
        analysis = data["analysis"]
        assert "extracted_roots" in analysis
        assert "morphological_analysis" in analysis
        assert "extracted_keywords" in analysis
        assert "content_themes" in analysis
    
    async def test_analyze_content_batch(self, async_client: AsyncClient, mock_services, mock_analysis_result):
        """Test batch content analysis."""
        mock_services["hebrew_analyzer"].analyze_content.return_value = mock_analysis_result
        mock_services["clustering_service"].generate_clusters.return_value = {
            "clusters": [], "metadata": {}
        }
        mock_services["keyword_expander"].expand_keywords.return_value = {
            "keyword_expansions": {}, "metadata": {}
        }
        
        request_data = {
            "texts": [TEST_HEBREW_TEXT, "טקסט נוסף לבדיקה"],
            "options": {
                "include_roots": True,
                "include_morphology": False,
                "include_keywords": True,
                "include_themes": False
            }
        }
        
        response = await async_client.post("/analysis/batch", json=request_data)
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 2
        assert "metadata" in data
        
        # Check each result
        for result in data["results"]:
            assert "analysis" in result
            assert "clusters" in result
            assert "keyword_expansions" in result
    
    async def test_analyze_content_invalid_text(self, async_client: AsyncClient):
        """Test analysis with invalid text."""
        request_data = {
            "text": "",  # Empty text
            "options": {}
        }
        
        response = await async_client.post("/analysis/analyze", json=request_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    async def test_analyze_content_too_long(self, async_client: AsyncClient):
        """Test analysis with text that's too long."""
        long_text = "א" * 50001  # Exceeds max length
        
        request_data = {
            "text": long_text,
            "options": {}
        }
        
        response = await async_client.post("/analysis/analyze", json=request_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestClusteringEndpoints:
    """Test semantic clustering endpoints."""
    
    async def test_generate_clusters(self, async_client: AsyncClient, mock_services, mock_clustering_result):
        """Test cluster generation."""
        mock_services["clustering_service"].generate_clusters.return_value = mock_clustering_result
        
        request_data = {
            "keywords": TEST_KEYWORDS,
            "options": {
                "clustering_method": "mixed",
                "min_cluster_size": 2,
                "max_clusters": 10
            }
        }
        
        response = await async_client.post("/clusters/generate", json=request_data)
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "clusters" in data
        assert "metadata" in data
        
        # Verify cluster structure
        clusters = data["clusters"]
        assert len(clusters) > 0
        
        for cluster in clusters:
            assert "id" in cluster
            assert "name" in cluster
            assert "keywords" in cluster
            assert "cluster_type" in cluster
            assert "score" in cluster
    
    async def test_clustering_methods_info(self, async_client: AsyncClient):
        """Test clustering methods info endpoint."""
        response = await async_client.get("/clusters/methods")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "methods" in data
        assert len(data["methods"]) > 0
        
        for method in data["methods"]:
            assert "name" in method
            assert "description" in method
    
    async def test_generate_clusters_empty_keywords(self, async_client: AsyncClient):
        """Test clustering with empty keywords."""
        request_data = {
            "keywords": [],
            "options": {}
        }
        
        response = await async_client.post("/clusters/generate", json=request_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestKeywordEndpoints:
    """Test keyword expansion endpoints."""
    
    async def test_expand_keywords(self, async_client: AsyncClient, mock_services, mock_keyword_expansion):
        """Test keyword expansion."""
        mock_services["keyword_expander"].expand_keywords.return_value = mock_keyword_expansion
        
        request_data = {
            "keywords": TEST_KEYWORDS[:1],  # Use first keyword
            "options": {
                "include_morphological": True,
                "include_semantic": True,
                "include_commercial": True,
                "include_locational": True,
                "include_question_based": True,
                "max_variations_per_type": 5
            }
        }
        
        response = await async_client.post("/keywords/expand", json=request_data)
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "expansions" in data
        assert "metadata" in data
        
        # Verify expansion structure
        expansions = data["expansions"]
        assert len(expansions) > 0
        
        for keyword, expansion in expansions.items():
            assert "original" in expansion
            assert "morphological_info" in expansion
            assert "variations_by_type" in expansion
            assert "all_variations" in expansion
            assert "expansion_score" in expansion
    
    async def test_search_volumes(self, async_client: AsyncClient, mock_services, mock_search_volume_data):
        """Test search volume retrieval."""
        mock_services["search_data_service"].get_comprehensive_keyword_data.return_value = mock_search_volume_data
        
        request_data = {
            "keywords": TEST_KEYWORDS[:1],
            "location": TEST_LOCATION,
            "language": TEST_LANGUAGE
        }
        
        response = await async_client.post("/search-volume", json=request_data)
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "volumes" in data
        assert "metadata" in data
        
        # Verify volume structure
        volumes = data["volumes"]
        assert len(volumes) > 0
        
        for keyword, volume_data in volumes.items():
            assert "search_volume" in volume_data
            assert "competition" in volume_data
            assert "cpc" in volume_data
            assert "difficulty" in volume_data
    
    async def test_expansion_methods_info(self, async_client: AsyncClient):
        """Test expansion methods info endpoint."""
        response = await async_client.get("/keywords/methods")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "methods" in data
        assert len(data["methods"]) > 0
        
        for method in data["methods"]:
            assert "name" in method
            assert "description" in method
    
    async def test_expand_keywords_empty_list(self, async_client: AsyncClient):
        """Test expansion with empty keyword list."""
        request_data = {
            "keywords": [],
            "options": {}
        }
        
        response = await async_client.post("/keywords/expand", json=request_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    async def test_invalid_json(self, async_client: AsyncClient):
        """Test handling of invalid JSON."""
        response = await async_client.post(
            "/analysis/analyze",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    async def test_missing_required_fields(self, async_client: AsyncClient):
        """Test handling of missing required fields."""
        request_data = {
            # Missing 'text' field
            "options": {}
        }
        
        response = await async_client.post("/analysis/analyze", json=request_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    async def test_service_error_handling(self, async_client: AsyncClient, mock_services):
        """Test handling of service errors."""
        # Mock service to raise an exception
        mock_services["hebrew_analyzer"].analyze_content.side_effect = Exception("Service error")
        
        request_data = {
            "text": TEST_HEBREW_TEXT,
            "options": {}
        }
        
        response = await async_client.post("/analysis/analyze", json=request_data)
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        
        data = response.json()
        assert "detail" in data
    
    async def test_nonexistent_endpoint(self, async_client: AsyncClient):
        """Test handling of nonexistent endpoints."""
        response = await async_client.get("/nonexistent")
        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestCaching:
    """Test caching functionality."""
    
    async def test_cache_hit_analysis(self, async_client: AsyncClient, mock_services, mock_analysis_result):
        """Test cache hit for analysis."""
        mock_services["hebrew_analyzer"].analyze_content.return_value = mock_analysis_result
        mock_services["clustering_service"].generate_clusters.return_value = {
            "clusters": [], "metadata": {}
        }
        mock_services["keyword_expander"].expand_keywords.return_value = {
            "keyword_expansions": {}, "metadata": {}
        }
        
        request_data = {
            "text": TEST_HEBREW_TEXT,
            "options": {"include_roots": True}
        }
        
        # First request - should call service
        response1 = await async_client.post("/analysis/analyze", json=request_data)
        assert response1.status_code == status.HTTP_200_OK
        
        # Second request - should hit cache
        response2 = await async_client.post("/analysis/analyze", json=request_data)
        assert response2.status_code == status.HTTP_200_OK
        
        # Verify responses are identical
        assert response1.json() == response2.json()
    
    async def test_cache_different_options(self, async_client: AsyncClient, mock_services, mock_analysis_result):
        """Test that different options don't hit same cache."""
        mock_services["hebrew_analyzer"].analyze_content.return_value = mock_analysis_result
        mock_services["clustering_service"].generate_clusters.return_value = {
            "clusters": [], "metadata": {}
        }
        mock_services["keyword_expander"].expand_keywords.return_value = {
            "keyword_expansions": {}, "metadata": {}
        }
        
        request_data1 = {
            "text": TEST_HEBREW_TEXT,
            "options": {"include_roots": True}
        }
        
        request_data2 = {
            "text": TEST_HEBREW_TEXT,
            "options": {"include_roots": False}
        }
        
        # Both requests should call service (different cache keys)
        response1 = await async_client.post("/analysis/analyze", json=request_data1)
        response2 = await async_client.post("/analysis/analyze", json=request_data2)
        
        assert response1.status_code == status.HTTP_200_OK
        assert response2.status_code == status.HTTP_200_OK
        
        # Should have been called twice due to different options
        assert mock_services["hebrew_analyzer"].analyze_content.call_count == 2


class TestPerformance:
    """Test performance-related functionality."""
    
    async def test_response_time_headers(self, async_client: AsyncClient, mock_services, mock_analysis_result):
        """Test that response time headers are included."""
        mock_services["hebrew_analyzer"].analyze_content.return_value = mock_analysis_result
        mock_services["clustering_service"].generate_clusters.return_value = {
            "clusters": [], "metadata": {}
        }
        mock_services["keyword_expander"].expand_keywords.return_value = {
            "keyword_expansions": {}, "metadata": {}
        }
        
        request_data = {
            "text": TEST_HEBREW_TEXT,
            "options": {}
        }
        
        response = await async_client.post("/analysis/analyze", json=request_data)
        assert response.status_code == status.HTTP_200_OK
        
        # Check for performance headers
        assert "X-Process-Time" in response.headers
        
        # Check metadata includes processing time
        data = response.json()
        assert "metadata" in data
        assert "processing_time_ms" in data["metadata"]
    
    async def test_concurrent_requests(self, async_client: AsyncClient, mock_services, mock_analysis_result):
        """Test handling of concurrent requests."""
        import asyncio
        
        mock_services["hebrew_analyzer"].analyze_content.return_value = mock_analysis_result
        mock_services["clustering_service"].generate_clusters.return_value = {
            "clusters": [], "metadata": {}
        }
        mock_services["keyword_expander"].expand_keywords.return_value = {
            "keyword_expansions": {}, "metadata": {}
        }
        
        request_data = {
            "text": TEST_HEBREW_TEXT,
            "options": {}
        }
        
        # Send multiple concurrent requests
        tasks = [
            async_client.post("/analysis/analyze", json=request_data)
            for _ in range(5)
        ]
        
        responses = await asyncio.gather(*tasks)
        
        # All should succeed
        for response in responses:
            assert response.status_code == status.HTTP_200_OK
