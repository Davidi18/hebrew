"""
Search Data Service for Hebrew Content Intelligence.
Integrates with DataForSEO MCP for keyword volumes and difficulty data.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from loguru import logger

import httpx
from config.settings import settings
from utils.cache_manager import cache_manager


class SearchDataService:
    """
    Service for retrieving search volume and keyword difficulty data.
    Integrates with DataForSEO MCP and provides intelligent caching.
    """
    
    def __init__(self):
        self.client: Optional[httpx.AsyncClient] = None
        self.api_url = settings.dataforseo_api_url
        self.api_key = settings.dataforseo_api_key
        self.rate_limit_delay = 0.5  # Seconds between requests
        self.last_request_time = 0.0
        
    async def initialize(self):
        """Initialize HTTP client and cache."""
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        )
        await cache_manager.initialize()
        logger.info("Search Data Service initialized")
    
    async def close(self):
        """Close HTTP client."""
        if self.client:
            await self.client.aclose()
    
    async def _rate_limit(self):
        """Implement rate limiting for API requests."""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        
        self.last_request_time = asyncio.get_event_loop().time()
    
    async def get_keyword_volumes(self, keywords: List[str], location: str = "Israel", language: str = "he") -> Dict[str, Any]:
        """
        Get search volume data for Hebrew keywords.
        
        Args:
            keywords: List of Hebrew keywords
            location: Target location (default: Israel)
            language: Target language (default: Hebrew)
            
        Returns:
            Dictionary with volume data for each keyword
        """
        if not keywords:
            return {"volumes": {}, "metadata": {"total_keywords": 0}}
        
        # Check cache first
        cache_key = f"{sorted(keywords)}_{location}_{language}"
        cached_result = await cache_manager.get_search_volumes(keywords)
        if cached_result:
            logger.debug(f"Cache hit for {len(keywords)} keywords")
            return cached_result
        
        try:
            # If DataForSEO API is not configured, return mock data
            if not self.api_key or self.api_key == "your_dataforseo_api_key":
                logger.info("DataForSEO API not configured, returning mock data")
                return await self._get_mock_volume_data(keywords, location, language)
            
            # Make API request to DataForSEO
            await self._rate_limit()
            
            payload = {
                "keywords": keywords,
                "location_name": location,
                "language_name": language,
                "include_serp_info": True,
                "include_clickstream_data": True
            }
            
            headers = {
                "Authorization": f"Basic {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = await self.client.post(
                f"{self.api_url}/keywords_data/google/search_volume/live",
                json=[payload],
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                result = await self._process_volume_response(data, keywords)
                
                # Cache the result
                await cache_manager.cache_search_volumes(keywords, result)
                
                return result
            else:
                logger.warning(f"DataForSEO API error: {response.status_code}")
                return await self._get_mock_volume_data(keywords, location, language)
                
        except Exception as e:
            logger.error(f"Error fetching search volumes: {e}")
            return await self._get_mock_volume_data(keywords, location, language)
    
    async def get_keyword_difficulty(self, keywords: List[str], location: str = "Israel") -> Dict[str, Any]:
        """
        Get keyword difficulty scores for Hebrew keywords.
        
        Args:
            keywords: List of Hebrew keywords
            location: Target location
            
        Returns:
            Dictionary with difficulty scores for each keyword
        """
        if not keywords:
            return {"difficulty": {}, "metadata": {"total_keywords": 0}}
        
        try:
            # If DataForSEO API is not configured, return mock data
            if not self.api_key or self.api_key == "your_dataforseo_api_key":
                logger.info("DataForSEO API not configured, returning mock difficulty data")
                return await self._get_mock_difficulty_data(keywords, location)
            
            # Make API request to DataForSEO
            await self._rate_limit()
            
            payload = {
                "keywords": keywords,
                "location_name": location,
                "include_serp_info": True
            }
            
            headers = {
                "Authorization": f"Basic {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = await self.client.post(
                f"{self.api_url}/keywords_data/google/keyword_difficulty/live",
                json=[payload],
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                return await self._process_difficulty_response(data, keywords)
            else:
                logger.warning(f"DataForSEO API error: {response.status_code}")
                return await self._get_mock_difficulty_data(keywords, location)
                
        except Exception as e:
            logger.error(f"Error fetching keyword difficulty: {e}")
            return await self._get_mock_difficulty_data(keywords, location)
    
    async def get_comprehensive_keyword_data(self, keywords: List[str], location: str = "Israel", language: str = "he") -> Dict[str, Any]:
        """
        Get comprehensive keyword data including volumes and difficulty.
        
        Args:
            keywords: List of Hebrew keywords
            location: Target location
            language: Target language
            
        Returns:
            Combined dictionary with volumes and difficulty data
        """
        # Run volume and difficulty requests concurrently
        volume_task = self.get_keyword_volumes(keywords, location, language)
        difficulty_task = self.get_keyword_difficulty(keywords, location)
        
        volume_data, difficulty_data = await asyncio.gather(
            volume_task, difficulty_task, return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(volume_data, Exception):
            logger.error(f"Volume data error: {volume_data}")
            volume_data = {"volumes": {}, "metadata": {"error": str(volume_data)}}
        
        if isinstance(difficulty_data, Exception):
            logger.error(f"Difficulty data error: {difficulty_data}")
            difficulty_data = {"difficulty": {}, "metadata": {"error": str(difficulty_data)}}
        
        # Combine results
        combined_data = {}
        for keyword in keywords:
            volume_info = volume_data.get("volumes", {}).get(keyword, {})
            difficulty_info = difficulty_data.get("difficulty", {}).get(keyword, {})
            
            combined_data[keyword] = {
                "search_volume": volume_info.get("search_volume", 0),
                "competition": volume_info.get("competition", 0.0),
                "cpc": volume_info.get("cpc", 0.0),
                "difficulty": difficulty_info.get("difficulty", 0),
                "serp_features": volume_info.get("serp_features", []),
                "trends": volume_info.get("trends", []),
                "related_keywords": difficulty_info.get("related_keywords", [])
            }
        
        return {
            "keywords": combined_data,
            "metadata": {
                "total_keywords": len(keywords),
                "location": location,
                "language": language,
                "timestamp": datetime.now().isoformat(),
                "volume_metadata": volume_data.get("metadata", {}),
                "difficulty_metadata": difficulty_data.get("metadata", {})
            }
        }
    
    async def _process_volume_response(self, data: Dict[str, Any], keywords: List[str]) -> Dict[str, Any]:
        """Process DataForSEO volume response."""
        volumes = {}
        
        if data.get("status_code") == 20000 and data.get("tasks"):
            task_result = data["tasks"][0]
            if task_result.get("result"):
                for item in task_result["result"]:
                    keyword = item.get("keyword", "")
                    if keyword in keywords:
                        volumes[keyword] = {
                            "search_volume": item.get("search_volume", 0),
                            "competition": item.get("competition", 0.0),
                            "cpc": item.get("cpc", {}).get("value", 0.0),
                            "trends": item.get("monthly_searches", []),
                            "serp_features": item.get("keyword_info", {}).get("serp_features", [])
                        }
        
        return {
            "volumes": volumes,
            "metadata": {
                "total_keywords": len(keywords),
                "found_keywords": len(volumes),
                "timestamp": datetime.now().isoformat()
            }
        }
    
    async def _process_difficulty_response(self, data: Dict[str, Any], keywords: List[str]) -> Dict[str, Any]:
        """Process DataForSEO difficulty response."""
        difficulty = {}
        
        if data.get("status_code") == 20000 and data.get("tasks"):
            task_result = data["tasks"][0]
            if task_result.get("result"):
                for item in task_result["result"]:
                    keyword = item.get("keyword", "")
                    if keyword in keywords:
                        difficulty[keyword] = {
                            "difficulty": item.get("keyword_difficulty", 0),
                            "related_keywords": item.get("related_keywords", [])[:5]  # Limit to top 5
                        }
        
        return {
            "difficulty": difficulty,
            "metadata": {
                "total_keywords": len(keywords),
                "found_keywords": len(difficulty),
                "timestamp": datetime.now().isoformat()
            }
        }
    
    async def _get_mock_volume_data(self, keywords: List[str], location: str, language: str) -> Dict[str, Any]:
        """Generate realistic mock volume data for testing."""
        import random
        
        volumes = {}
        for keyword in keywords:
            # Generate realistic Hebrew keyword volumes
            base_volume = random.randint(100, 10000)
            
            # Hebrew keywords typically have lower volumes
            if len(keyword) > 10:  # Longer phrases
                base_volume = int(base_volume * 0.3)
            
            volumes[keyword] = {
                "search_volume": base_volume,
                "competition": round(random.uniform(0.1, 0.9), 2),
                "cpc": round(random.uniform(0.5, 5.0), 2),
                "trends": [
                    {"month": i, "search_volume": int(base_volume * random.uniform(0.7, 1.3))}
                    for i in range(1, 13)
                ],
                "serp_features": random.sample(
                    ["featured_snippet", "local_pack", "images", "videos", "shopping"],
                    random.randint(0, 3)
                )
            }
        
        return {
            "volumes": volumes,
            "metadata": {
                "total_keywords": len(keywords),
                "found_keywords": len(volumes),
                "location": location,
                "language": language,
                "timestamp": datetime.now().isoformat(),
                "mock_data": True
            }
        }
    
    async def _get_mock_difficulty_data(self, keywords: List[str], location: str) -> Dict[str, Any]:
        """Generate realistic mock difficulty data for testing."""
        import random
        
        difficulty = {}
        for keyword in keywords:
            # Generate realistic difficulty scores
            base_difficulty = random.randint(20, 80)
            
            # Shorter, more generic keywords are typically harder
            if len(keyword) < 8:
                base_difficulty = min(90, int(base_difficulty * 1.3))
            
            difficulty[keyword] = {
                "difficulty": base_difficulty,
                "related_keywords": [
                    f"{keyword} {suffix}" for suffix in random.sample(
                        ["מחיר", "ביקורות", "איכות", "זול", "טוב", "מומלץ"],
                        random.randint(2, 4)
                    )
                ]
            }
        
        return {
            "difficulty": difficulty,
            "metadata": {
                "total_keywords": len(keywords),
                "found_keywords": len(difficulty),
                "location": location,
                "timestamp": datetime.now().isoformat(),
                "mock_data": True
            }
        }


# Global search data service instance
search_data_service = SearchDataService()
