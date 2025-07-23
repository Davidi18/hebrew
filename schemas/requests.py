"""
Request schemas for Hebrew Content Intelligence Service API.
Pydantic models for validating incoming requests.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator


class AnalysisOptions(BaseModel):
    """Options for content analysis."""
    include_volumes: bool = Field(default=True, description="Include search volume data from DataForSEO")
    include_difficulty: bool = Field(default=True, description="Include keyword difficulty scores")
    include_clusters: bool = Field(default=True, description="Generate semantic clusters")
    include_related: bool = Field(default=True, description="Include related keywords")
    include_expansion: bool = Field(default=True, description="Include keyword expansions")
    max_keywords: int = Field(default=50, ge=1, le=200, description="Maximum keywords to analyze")
    max_clusters: int = Field(default=10, ge=1, le=20, description="Maximum clusters to generate")


class ContentAnalysisRequest(BaseModel):
    """Request for Hebrew content analysis."""
    text: str = Field(..., min_length=1, max_length=50000, description="Hebrew text to analyze")
    url: Optional[str] = Field(default=None, description="Optional source URL")
    options: Optional[AnalysisOptions] = Field(default_factory=AnalysisOptions, description="Analysis options")
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()


class ClusterGenerationRequest(BaseModel):
    """Request for semantic cluster generation."""
    keywords: List[str] = Field(..., min_items=1, max_items=100, description="Keywords to cluster")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Clustering options")
    
    @field_validator('keywords')
    @classmethod
    def validate_keywords(cls, v):
        if not v:
            raise ValueError('Keywords list cannot be empty')
        # Remove empty strings and duplicates
        cleaned = list(set(kw.strip() for kw in v if kw.strip()))
        if not cleaned:
            raise ValueError('No valid keywords provided')
        return cleaned


class KeywordExpansionRequest(BaseModel):
    """Request for Hebrew keyword expansion."""
    keywords: List[str] = Field(..., min_items=1, max_items=50, description="Keywords to expand")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Expansion options")
    
    @field_validator('keywords')
    @classmethod
    def validate_keywords(cls, v):
        if not v:
            raise ValueError('Keywords list cannot be empty')
        # Remove empty strings and duplicates
        cleaned = list(set(kw.strip() for kw in v if kw.strip()))
        if not cleaned:
            raise ValueError('No valid keywords provided')
        return cleaned


class BatchAnalysisRequest(BaseModel):
    """Request for batch content analysis."""
    texts: List[str] = Field(..., min_items=1, max_items=10, description="List of texts to analyze")
    options: Optional[AnalysisOptions] = Field(default_factory=AnalysisOptions, description="Analysis options")
    
    @field_validator('texts')
    @classmethod
    def validate_texts(cls, v):
        if not v:
            raise ValueError('Texts list cannot be empty')
        # Validate each text
        cleaned = []
        for text in v:
            if text and text.strip():
                if len(text.strip()) > 50000:
                    raise ValueError(f'Text too long: {len(text)} characters (max 50000)')
                cleaned.append(text.strip())
        
        if not cleaned:
            raise ValueError('No valid texts provided')
        return cleaned


class SearchVolumeRequest(BaseModel):
    """Request for search volume data."""
    keywords: List[str] = Field(..., min_items=1, max_items=100, description="Keywords for volume lookup")
    location: Optional[str] = Field(default="Israel", description="Target location for search data")
    language: Optional[str] = Field(default="he", description="Target language")
    
    @field_validator('keywords')
    @classmethod
    def validate_keywords(cls, v):
        if not v:
            raise ValueError('Keywords list cannot be empty')
        return [kw.strip() for kw in v if kw.strip()]
