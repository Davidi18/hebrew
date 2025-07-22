"""
Request logging middleware for Hebrew Content Intelligence Service.
Provides comprehensive request/response logging and performance monitoring.
"""

import time
import json
from typing import Callable
from fastapi import FastAPI, Request, Response
from loguru import logger

from config.settings import settings


def setup_request_logging(app: FastAPI):
    """Setup request logging middleware for the FastAPI application."""
    
    @app.middleware("http")
    async def log_requests(request: Request, call_next: Callable) -> Response:
        """Log all HTTP requests and responses with timing information."""
        
        # Skip logging for health check endpoints in production
        if not settings.debug and request.url.path in ["/health", "/health/live", "/health/ready"]:
            return await call_next(request)
        
        start_time = time.time()
        
        # Log request
        request_body = None
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                # Read request body for logging (be careful with large payloads)
                body = await request.body()
                if len(body) < 10000:  # Only log small payloads
                    request_body = body.decode('utf-8')
                else:
                    request_body = f"<large payload: {len(body)} bytes>"
            except Exception:
                request_body = "<unable to read body>"
        
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        if settings.debug and request_body:
            logger.debug(f"Request body: {request_body}")
        
        # Process request
        try:
            response = await call_next(request)
            processing_time = time.time() - start_time
            
            # Log response
            logger.info(
                f"Response: {response.status_code} "
                f"in {processing_time:.3f}s "
                f"for {request.method} {request.url.path}"
            )
            
            # Add performance headers
            response.headers["X-Process-Time"] = str(round(processing_time * 1000, 2))
            response.headers["X-Service-Version"] = settings.version
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            logger.error(
                f"Request failed: {request.method} {request.url.path} "
                f"after {processing_time:.3f}s - {str(e)}"
            )
            raise
