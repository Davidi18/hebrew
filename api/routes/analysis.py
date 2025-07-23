"""
Analysis routes for Hebrew Content Intelligence Service.
Provides content analysis and batch processing endpoints.
"""

import time
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from loguru import logger

from schemas.requests import ContentAnalysisRequest, BatchAnalysisRequest
from schemas.responses import ContentAnalysisResponse, BatchAnalysisResponse, ErrorResponse
from services.hebrew_analyzer import hebrew_analyzer
from services.semantic_clusters import semantic_clustering
from utils.cache_manager import cache_manager

router = APIRouter()


@router.post("/analyze", response_model=ContentAnalysisResponse)
async def analyze_content(request: ContentAnalysisRequest):
    """
    Analyze Hebrew content for semantic insights.
    
    Provides comprehensive analysis including:
    - Hebrew morphological analysis
    - Root extraction (שורש המילה)
    - Named Entity Recognition
    - Semantic clustering
    - Keyword extraction
    - Content theme analysis
    """
    start_time = time.time()
    
    try:
        logger.info(f"Starting content analysis for {len(request.text)} characters")
        
        # Check cache first
        cached_result = await cache_manager.get_hebrew_analysis(request.text)
        if cached_result and not request.options.get('force_refresh', False):
            logger.info("Returning cached analysis result")
            return ContentAnalysisResponse(**cached_result)
        
        # Perform Hebrew content analysis
        analysis_results = await hebrew_analyzer.analyze_content(
            request.text, 
            request.options or {}
        )
        
        # Generate semantic clusters
        clustering_results = await semantic_clustering.generate_clusters(
            analysis_results,
            request.options or {}
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Build response
        response_data = {
            "success": True,
            "semantic_clusters": clustering_results.get('semantic_clusters', []),
            "content_gaps": [],  # Would be populated by business logic
            "business_insights": [],  # Would be populated by business logic
            "metadata": {
                "processing_time_ms": processing_time,
                "text_length": len(request.text),
                "hebrew_ratio": analysis_results.get('language_stats', {}).get('hebrew_ratio', 0),
                "analysis_timestamp": time.time()
            },
            "raw_analysis": analysis_results if request.options.get('include_raw', False) else None
        }
        
        # Cache the result
        await cache_manager.cache_hebrew_analysis(request.text, response_data)
        
        logger.info(f"Content analysis completed in {processing_time:.2f}ms")
        
        return ContentAnalysisResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Content analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Content analysis failed: {str(e)}"
        )


@router.post("/batch-analyze", response_model=BatchAnalysisResponse)
async def batch_analyze_content(request: BatchAnalysisRequest):
    """
    Analyze multiple Hebrew texts in batch.
    
    Efficiently processes multiple texts with:
    - Parallel processing
    - Shared model loading
    - Intelligent caching
    - Error handling per text
    """
    start_time = time.time()
    
    try:
        logger.info(f"Starting batch analysis for {len(request.texts)} texts")
        
        if len(request.texts) > 50:  # Reasonable limit
            raise HTTPException(
                status_code=400,
                detail="Batch size too large. Maximum 50 texts per batch."
            )
        
        results = []
        failed_analyses = []
        
        # Process each text
        for i, text in enumerate(request.texts):
            try:
                # Create individual analysis request
                individual_request = ContentAnalysisRequest(
                    text=text,
                    options=request.options
                )
                
                # Analyze content
                analysis_result = await analyze_content(individual_request)
                results.append(analysis_result)
                
            except Exception as e:
                logger.error(f"Failed to analyze text {i}: {e}")
                failed_analyses.append({
                    "text_index": i,
                    "error": str(e),
                    "text_preview": text[:100] + "..." if len(text) > 100 else text
                })
        
        processing_time = (time.time() - start_time) * 1000
        
        batch_metadata = {
            "total_texts": len(request.texts),
            "successful_analyses": len(results),
            "failed_analyses": len(failed_analyses),
            "processing_time_ms": processing_time,
            "avg_time_per_text": processing_time / len(request.texts) if request.texts else 0,
            "batch_timestamp": time.time()
        }
        
        logger.info(f"Batch analysis completed: {len(results)}/{len(request.texts)} successful in {processing_time:.2f}ms")
        
        return BatchAnalysisResponse(
            success=len(failed_analyses) == 0,
            results=results,
            batch_metadata=batch_metadata,
            failed_analyses=failed_analyses
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch analysis failed: {str(e)}"
        )
