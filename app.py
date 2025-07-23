"""
Hebrew Content Intelligence Service - Main FastAPI Application
Advanced Hebrew content analysis with Hebrew Transformers, semantic clustering, and DataForSEO integration.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Dict, Any
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import uvicorn

from config.settings import settings
from models.hebspacy_loader import hebrew_loader
from utils.cache_manager import cache_manager
from services.search_data_service import search_data_service
from api.routes import analysis, clusters, keywords, health
from api.middleware.error_handler import setup_error_handlers
from api.middleware.request_logger import setup_request_logging


async def startup_event():
    """Initialize services on startup."""
    try:
        logger.info("Starting Hebrew Content Intelligence Service...")
        
        # Initialize cache manager
        await cache_manager.initialize()
        
        # Initialize search data service
        await search_data_service.initialize()
        
        # Load Hebrew Transformers models
        logger.info("Loading Hebrew Transformers models...")
        await hebrew_loader.load_model()
        logger.success("Hebrew Transformers models loaded successfully")
        
        # Store startup time
        app.state.startup_time = datetime.now()
        
        logger.success("✅ Hebrew Content Intelligence Service started successfully!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise


async def shutdown_event():
    """Cleanup on shutdown."""
    try:
        logger.info("Shutting down Hebrew Content Intelligence Service...")
        
        # Close search data service
        await search_data_service.close()
        
        logger.info("✅ Shutdown completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")


# Create FastAPI application
app = FastAPI(
    title="Hebrew Content Intelligence Service",
    description="Advanced Hebrew content analysis using Hebrew Transformers, semantic clustering, and DataForSEO integration",
    version=settings.version,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    on_startup=[startup_event],
    on_shutdown=[shutdown_event]
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup error handling and logging
setup_error_handlers(app)
setup_request_logging(app)

# Include API routes
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(analysis.router, prefix="/api", tags=["analysis"])
app.include_router(clusters.router, prefix="/api", tags=["clusters"])
app.include_router(keywords.router, prefix="/api", tags=["keywords"])


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": settings.service_name,
        "version": settings.version,
        "status": "running",
        "description": "Hebrew Content Intelligence Service",
        "endpoints": {
            "health": "/health",
            "analyze": "/api/analyze",
            "clusters": "/api/clusters",
            "keywords": "/api/keywords",
            "docs": "/docs" if settings.debug else "disabled"
        }
    }


@app.get("/info")
async def service_info():
    """Detailed service information and model status."""
    model_info = await hebrew_loader.get_model_info()
    
    return {
        "service": {
            "name": "Hebrew Content Intelligence Service",
            "version": settings.version,
            "status": "running",
            "startup_time": app.state.startup_time.isoformat() if hasattr(app.state, 'startup_time') else None,
            "debug_mode": settings.debug
        },
        "model": model_info,
        "cache": {
            "enabled": settings.cache_enabled,
            "ttl": settings.cache_ttl
        },
        "limits": {
            "max_content_length": settings.max_content_length,
            "max_keywords": settings.max_keywords
        },
        "capabilities": [
            "Hebrew tokenization with heBERT",
            "Named Entity Recognition",
            "Hebrew embeddings and contextual analysis",
            "Semantic clustering",
            "Keyword expansion",
            "Content theme analysis",
            "Mixed Hebrew-English processing"
        ]
    }


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        "logs/app.log",
        rotation="1 day",
        retention="30 days",
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    )
    
    if settings.debug:
        logger.add(
            lambda msg: print(msg, end=""),
            level=settings.log_level,
            format="{time:HH:mm:ss} | {level} | {message}"
        )
    
    # Run the application
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1,  # Single worker for model sharing
        log_level=settings.log_level.lower()
    )
