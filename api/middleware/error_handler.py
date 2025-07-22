"""
Error handling middleware for Hebrew Content Intelligence Service.
Provides centralized error handling and response formatting.
"""

import traceback
from datetime import datetime
from typing import Union
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from loguru import logger
from pydantic import ValidationError

from schemas.responses import ErrorResponse


def setup_error_handlers(app: FastAPI):
    """Setup global error handlers for the FastAPI application."""
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions."""
        logger.warning(f"HTTP {exc.status_code}: {exc.detail} - {request.url}")
        
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error="HTTPException",
                message=str(exc.detail),
                details={"status_code": exc.status_code}
            ).dict()
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors."""
        logger.warning(f"Validation error: {exc.errors()} - {request.url}")
        
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="ValidationError",
                message="Request validation failed",
                details={
                    "validation_errors": exc.errors(),
                    "body": exc.body
                }
            ).dict()
        )
    
    @app.exception_handler(ValidationError)
    async def pydantic_validation_handler(request: Request, exc: ValidationError):
        """Handle Pydantic validation errors."""
        logger.warning(f"Pydantic validation error: {exc.errors()} - {request.url}")
        
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="PydanticValidationError",
                message="Data validation failed",
                details={"validation_errors": exc.errors()}
            ).dict()
        )
    
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        """Handle ValueError exceptions."""
        logger.error(f"ValueError: {str(exc)} - {request.url}")
        
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error="ValueError",
                message=str(exc),
                details={"request_url": str(request.url)}
            ).dict()
        )
    
    @app.exception_handler(TimeoutError)
    async def timeout_error_handler(request: Request, exc: TimeoutError):
        """Handle timeout errors."""
        logger.error(f"Timeout error: {str(exc)} - {request.url}")
        
        return JSONResponse(
            status_code=408,
            content=ErrorResponse(
                error="TimeoutError",
                message="Request timed out",
                details={"request_url": str(request.url)}
            ).dict()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle all other exceptions."""
        error_id = f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.error(
            f"Unhandled exception [{error_id}]: {str(exc)} - {request.url}\n"
            f"Traceback: {traceback.format_exc()}"
        )
        
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="InternalServerError",
                message="An internal server error occurred",
                details={
                    "error_id": error_id,
                    "request_url": str(request.url)
                }
            ).dict()
        )
