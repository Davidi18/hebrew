"""
Keyword Expansion API endpoints for Hebrew Content Intelligence Service.
Provides Hebrew keyword expansion and variation generation.
"""

import time
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException
from loguru import logger

from schemas.requests import KeywordExpansionRequest, SearchVolumeRequest
from schemas.responses import KeywordExpansionResponse, SearchVolumeResponse, KeywordExpansionResult, KeywordData, KeywordVolume, ResponseMetadata
from services.keyword_expander import keyword_expander
from services.search_data_service import search_data_service
from utils.cache_manager import cache_manager

router = APIRouter()


@router.post("/keywords/expand", response_model=KeywordExpansionResponse)
async def expand_keywords(request: KeywordExpansionRequest):
    """
    Expand Hebrew keywords with morphological and semantic variations.
    
    Generates:
    - Morphological variations (plural, construct, possessive)
    - Semantic variations (synonyms, related terms)
    - Commercial intent variations
    - Locational variations (Israeli cities/regions)
    - Question-based variations
    """
    start_time = time.time()
    
    try:
        logger.info(f"Starting keyword expansion for {len(request.keywords)} keywords")
        
        # Perform keyword expansion
        expansion_result = await keyword_expander.expand_keywords(
            request.keywords,
            options=request.options
        )
        
        # Convert to response format
        keyword_expansions = {}
        for keyword, expansion_data in expansion_result.get('keyword_expansions', {}).items():
            keyword_expansions[keyword] = KeywordExpansionResult(
                original=expansion_data.get('original', keyword),
                morphological_info=expansion_data.get('morphological_info', {}),
                variations_by_type=expansion_data.get('variations_by_type', {}),
                all_variations=expansion_data.get('all_variations', []),
                expansion_score=expansion_data.get('expansion_score', 0.0)
            )
        
        processing_time = int((time.time() - start_time) * 1000)
        expansion_metadata = expansion_result.get('expansion_metadata', {})
        expansion_metadata['processing_time_ms'] = processing_time
        
        logger.info(f"Keyword expansion completed in {processing_time}ms")
        
        return KeywordExpansionResponse(
            keyword_expansions=keyword_expansions,
            combined_variations=expansion_result.get('combined_variations', []),
            expansion_metadata=expansion_metadata
        )
        
    except Exception as e:
        logger.error(f"Keyword expansion failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Keyword expansion failed: {str(e)}"
        )


@router.post("/search-volume", response_model=SearchVolumeResponse)
async def get_search_volumes(request: SearchVolumeRequest):
    """
    Get search volume data for Hebrew keywords.
    Integrates with DataForSEO for real search data.
    """
    try:
        start_time = time.time()
        
        # Get comprehensive keyword data including volumes and difficulty
        search_data = await search_data_service.get_comprehensive_keyword_data(
            keywords=request.keywords,
            location=request.location,
            language=request.language
        )
        
        # Format response
        keyword_volumes = {}
        for keyword, data in search_data.get("keywords", {}).items():
            keyword_volumes[keyword] = KeywordVolume(
                keyword=keyword,
                search_volume=data.get("search_volume", 0),
                competition=data.get("competition", 0.0),
                cpc=data.get("cpc", 0.0),
                difficulty=data.get("difficulty", 0),
                trends=data.get("trends", []),
                related_keywords=data.get("related_keywords", [])
            )
        
        processing_time = time.time() - start_time
        
        return SearchVolumeResponse(
            volumes=keyword_volumes,
            metadata=ResponseMetadata(
                processing_time_ms=int(processing_time * 1000),
                total_keywords=len(request.keywords),
                found_keywords=len(keyword_volumes),
                location=request.location,
                language=request.language,
                cached=search_data.get("metadata", {}).get("cached", False),
                additional_info={
                    "search_metadata": search_data.get("metadata", {}),
                    "mock_data": search_data.get("metadata", {}).get("mock_data", False)
                }
            )
        )
        
    except Exception as e:
        logger.error(f"Search volume error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get search volumes: {str(e)}"
        )


@router.get("/keywords/expansion-methods")
async def get_expansion_methods():
    """
    Get information about available keyword expansion methods.
    """
    return {
        "expansion_types": [
            {
                "name": "morphological",
                "description": "Hebrew morphological variations (plural, construct, possessive)",
                "examples": ["בית → בתים, בתי, ביתו"]
            },
            {
                "name": "semantic",
                "description": "Synonyms and semantically related terms",
                "examples": ["בית → דירה, מגורים, משכן"]
            },
            {
                "name": "commercial",
                "description": "Commercial intent variations",
                "examples": ["מטבח → מחיר מטבח, קניית מטבח, מטבח זול"]
            },
            {
                "name": "locational",
                "description": "Israeli geographic variations",
                "examples": ["עיצוב → עיצוב תל אביב, עיצוב ירושלים"]
            },
            {
                "name": "question",
                "description": "Question-based variations",
                "examples": ["מטבח → מה זה מטבח, איך לבחור מטבח"]
            }
        ],
        "morphological_patterns": {
            "nouns": ["plural forms", "construct state", "possessive forms", "diminutive"],
            "verbs": ["tense variations", "participles", "infinitive forms"],
            "adjectives": ["gender/number agreement", "comparative forms"]
        },
        "supported_features": [
            "Hebrew root extraction",
            "Mixed Hebrew-English content",
            "Commercial intent detection",
            "Israeli location targeting",
            "Question pattern generation"
        ]
    }


@router.get("/keywords/hebrew-roots")
async def get_hebrew_roots_info():
    """
    Get information about Hebrew root analysis capabilities.
    """
    return {
        "description": "Hebrew root (שורש) extraction and analysis",
        "capabilities": [
            "3-letter root extraction",
            "Morphological pattern recognition",
            "Root-based keyword clustering",
            "Semantic relationship mapping"
        ],
        "common_patterns": {
            "כתב": "writing, document related",
            "למד": "learning, education related", 
            "עבד": "work, labor related",
            "בנה": "building, construction related",
            "שמר": "keeping, guarding related"
        },
        "morphological_transformations": [
            "Prefix addition (ה, ו, ב, כ, ל, מ, ש)",
            "Suffix addition (ים, ות, ה, ת, ן, ך)",
            "Vowel pattern changes",
            "Construct state formation"
        ],
        "use_cases": [
            "Semantic clustering",
            "Keyword expansion",
            "Content theme analysis",
            "Related term discovery"
        ]
    }
