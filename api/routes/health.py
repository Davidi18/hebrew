"""
Health check endpoint for Hebrew Content Intelligence Service.
Provides service status, model health, and system information.
"""

import time
from datetime import datetime
from fastapi import APIRouter, HTTPException
from loguru import logger

from config.settings import settings
from models.hebspacy_loader import hebspacy_loader
from schemas.responses import HealthResponse

router = APIRouter()

# Track service start time for uptime calculation
SERVICE_START_TIME = time.time()


@router.get("/", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns service status, model information, and uptime.
    """
    try:
        # Get model status
        model_info = await hebspacy_loader.get_model_info()
        
        # Calculate uptime
        uptime_seconds = time.time() - SERVICE_START_TIME
        
        # Determine overall status
        status = "healthy"
        if not model_info.get("loaded", False):
            status = "degraded"
        
        return HealthResponse(
            status=status,
            timestamp=datetime.now(),
            version=settings.version,
            model_status=model_info,
            uptime_seconds=round(uptime_seconds, 2)
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get("/ready")
async def readiness_check():
    """
    Readiness check for Kubernetes/Docker health probes.
    Returns 200 if service is ready to handle requests.
    """
    try:
        # Check if HebSpacy model is loaded
        if not hebspacy_loader.is_loaded():
            raise HTTPException(
                status_code=503,
                detail="HebSpacy model not loaded"
            )
        
        # Test basic model functionality
        test_result = await hebspacy_loader.analyze_text("בדיקה")
        if not test_result or not test_result.get("tokens"):
            raise HTTPException(
                status_code=503,
                detail="Model functionality test failed"
            )
        
        return {"status": "ready", "timestamp": datetime.now()}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Service not ready: {str(e)}"
        )


@router.get("/live")
async def liveness_check():
    """
    Liveness check for Kubernetes/Docker health probes.
    Returns 200 if service is alive (basic functionality).
    """
    return {
        "status": "alive",
        "timestamp": datetime.now(),
        "service": settings.service_name,
        "version": settings.version
    }
