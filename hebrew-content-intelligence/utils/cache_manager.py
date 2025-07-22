"""
Cache Manager for Hebrew Content Intelligence Service.
Provides caching layer for expensive NLP operations using Redis or memory cache.
"""

import json
import hashlib
import asyncio
from typing import Any, Optional, Dict, Union
from datetime import datetime, timedelta
from loguru import logger

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available, using memory cache only")

from config.settings import settings


class CacheManager:
    """
    Unified cache manager supporting Redis and in-memory caching.
    Optimized for Hebrew NLP operations with intelligent key generation.
    """
    
    def __init__(self):
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.redis_client: Optional[redis.Redis] = None
        self.cache_type = settings.cache_type.lower()
        self.default_ttl = settings.cache_ttl
        
    async def initialize(self):
        """Initialize cache connections."""
        if self.cache_type == "redis" and REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(
                    settings.redis_url,
                    decode_responses=True,
                    socket_timeout=5,
                    socket_connect_timeout=5
                )
                # Test connection
                await self.redis_client.ping()
                logger.info("Redis cache initialized successfully")
            except Exception as e:
                logger.warning(f"Redis connection failed, falling back to memory cache: {e}")
                self.cache_type = "memory"
        else:
            self.cache_type = "memory"
            logger.info("Using memory cache")
    
    def _generate_cache_key(self, prefix: str, data: Union[str, Dict[str, Any]]) -> str:
        """Generate consistent cache key from data."""
        if isinstance(data, str):
            content = data
        else:
            content = json.dumps(data, sort_keys=True, ensure_ascii=False)
        
        # Create hash of content for consistent keys
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
        return f"hebrew_intel:{prefix}:{content_hash}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            if self.cache_type == "redis" and self.redis_client:
                value = await self.redis_client.get(key)
                if value:
                    return json.loads(value)
            else:
                # Memory cache
                if key in self.memory_cache:
                    cache_entry = self.memory_cache[key]
                    if cache_entry['expires_at'] > datetime.now():
                        return cache_entry['value']
                    else:
                        # Expired, remove from cache
                        del self.memory_cache[key]
            
            return None
            
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with TTL."""
        try:
            ttl = ttl or self.default_ttl
            serialized_value = json.dumps(value, ensure_ascii=False, default=str)
            
            if self.cache_type == "redis" and self.redis_client:
                await self.redis_client.setex(key, ttl, serialized_value)
            else:
                # Memory cache
                expires_at = datetime.now() + timedelta(seconds=ttl)
                self.memory_cache[key] = {
                    'value': value,
                    'expires_at': expires_at
                }
                
                # Clean up expired entries periodically
                await self._cleanup_memory_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            if self.cache_type == "redis" and self.redis_client:
                await self.redis_client.delete(key)
            else:
                self.memory_cache.pop(key, None)
            
            return True
            
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def cache_hebrew_analysis(self, text: str, analysis_result: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Cache Hebrew text analysis results."""
        key = self._generate_cache_key("analysis", text)
        return await self.set(key, analysis_result, ttl)
    
    async def get_hebrew_analysis(self, text: str) -> Optional[Dict[str, Any]]:
        """Get cached Hebrew text analysis."""
        key = self._generate_cache_key("analysis", text)
        return await self.get(key)
    
    async def cache_keyword_expansion(self, keywords: list, expansion_result: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Cache keyword expansion results."""
        key = self._generate_cache_key("expansion", {"keywords": keywords})
        return await self.set(key, expansion_result, ttl)
    
    async def get_keyword_expansion(self, keywords: list) -> Optional[Dict[str, Any]]:
        """Get cached keyword expansion."""
        key = self._generate_cache_key("expansion", {"keywords": keywords})
        return await self.get(key)
    
    async def cache_semantic_clusters(self, analysis_data: Dict[str, Any], clusters_result: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Cache semantic clustering results."""
        # Use a subset of analysis data for key generation to avoid huge keys
        key_data = {
            "keywords": analysis_data.get('extracted_keywords', {}).get('top_keywords', [])[:10],
            "themes": analysis_data.get('content_themes', {}).get('dominant_themes', [])
        }
        key = self._generate_cache_key("clusters", key_data)
        return await self.set(key, clusters_result, ttl)
    
    async def get_semantic_clusters(self, analysis_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached semantic clustering."""
        key_data = {
            "keywords": analysis_data.get('extracted_keywords', {}).get('top_keywords', [])[:10],
            "themes": analysis_data.get('content_themes', {}).get('dominant_themes', [])
        }
        key = self._generate_cache_key("clusters", key_data)
        return await self.get(key)
    
    async def cache_search_volumes(self, keywords: list, volumes_result: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Cache search volume data."""
        key = self._generate_cache_key("volumes", {"keywords": sorted(keywords)})
        # Use longer TTL for search volume data as it changes less frequently
        volume_ttl = ttl or (self.default_ttl * 24)  # 24x longer
        return await self.set(key, volumes_result, volume_ttl)
    
    async def get_search_volumes(self, keywords: list) -> Optional[Dict[str, Any]]:
        """Get cached search volume data."""
        key = self._generate_cache_key("volumes", {"keywords": sorted(keywords)})
        return await self.get(key)
    
    async def _cleanup_memory_cache(self):
        """Clean up expired entries from memory cache."""
        if len(self.memory_cache) > 1000:  # Only cleanup when cache gets large
            now = datetime.now()
            expired_keys = [
                key for key, entry in self.memory_cache.items()
                if entry['expires_at'] <= now
            ]
            for key in expired_keys:
                del self.memory_cache[key]
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "cache_type": self.cache_type,
            "default_ttl": self.default_ttl
        }
        
        if self.cache_type == "redis" and self.redis_client:
            try:
                info = await self.redis_client.info()
                stats.update({
                    "redis_connected": True,
                    "redis_memory_used": info.get('used_memory_human', 'unknown'),
                    "redis_keys": await self.redis_client.dbsize()
                })
            except Exception as e:
                stats.update({
                    "redis_connected": False,
                    "redis_error": str(e)
                })
        else:
            stats.update({
                "memory_cache_size": len(self.memory_cache),
                "memory_cache_keys": list(self.memory_cache.keys())[:10]  # Sample keys
            })
        
        return stats
    
    async def clear_cache(self, pattern: Optional[str] = None) -> bool:
        """Clear cache entries, optionally by pattern."""
        try:
            if self.cache_type == "redis" and self.redis_client:
                if pattern:
                    keys = await self.redis_client.keys(f"hebrew_intel:{pattern}:*")
                    if keys:
                        await self.redis_client.delete(*keys)
                else:
                    await self.redis_client.flushdb()
            else:
                if pattern:
                    keys_to_delete = [
                        key for key in self.memory_cache.keys()
                        if pattern in key
                    ]
                    for key in keys_to_delete:
                        del self.memory_cache[key]
                else:
                    self.memory_cache.clear()
            
            logger.info(f"Cache cleared{f' for pattern: {pattern}' if pattern else ''}")
            return True
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False


# Global cache manager instance
cache_manager = CacheManager()
