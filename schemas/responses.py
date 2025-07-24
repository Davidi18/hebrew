"""
Response schemas for Hebrew Content Intelligence Service API.
Pydantic models for API responses.
"""

import json
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime


class KeywordData(BaseModel):
    """Individual keyword with SEO data."""
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat() if v else None}
    )
    
    keyword: str = Field(..., description="The keyword")
    volume: Optional[int] = Field(default=None, description="Monthly search volume")
    difficulty: Optional[float] = Field(default=None, description="Keyword difficulty score (0-100)")
    cpc: Optional[float] = Field(default=None, description="Cost per click")
    competition: Optional[str] = Field(default=None, description="Competition level")


class KeywordVolume(BaseModel):
    """Keyword volume data."""
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat() if v else None}
    )
    
    keyword: str = Field(..., description="The keyword")
    volume: Optional[int] = Field(default=None, description="Monthly search volume")
    data_source: str = Field(default="DataForSEO", description="Data source")
    last_updated: Optional[datetime] = Field(default=None, description="Last update timestamp")


class ResponseMetadata(BaseModel):
    """Response metadata information."""
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat() if v else None}
    )
    
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    processing_time_ms: Optional[float] = Field(default=None, description="Processing time in milliseconds")
    request_id: Optional[str] = Field(default=None, description="Request ID for tracking")
    data_source: Optional[str] = Field(default=None, description="Data source used")
    cache_hit: bool = Field(default=False, description="Whether response was served from cache")


class SemanticCluster(BaseModel):
    """Semantic cluster of related keywords."""
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat() if v else None}
    )
    
    cluster_name: str = Field(..., description="Name of the cluster")
    root_concept: str = Field(..., description="Root concept or theme")
    primary_keywords: List[KeywordData] = Field(..., description="Main keywords in cluster")
    variations: List[KeywordData] = Field(default_factory=list, description="Keyword variations")
    related_concepts: List[str] = Field(default_factory=list, description="Related concepts")
    cluster_type: str = Field(..., description="Type of clustering used")
    coherence_score: float = Field(..., description="Cluster coherence score (0-1)")


class ContentGap(BaseModel):
    """Identified content gap opportunity."""
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat() if v else None}
    )
    
    gap_type: str = Field(..., description="Type of content gap")
    description: str = Field(..., description="Description of the gap")
    keywords: List[KeywordData] = Field(..., description="Related keywords")
    opportunity_score: float = Field(..., description="Opportunity score (0-1)")
    recommendation: str = Field(..., description="Recommended action")


class BusinessInsight(BaseModel):
    """Business insight from content analysis."""
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat() if v else None}
    )
    
    insight_type: str = Field(..., description="Type of insight")
    title: str = Field(..., description="Insight title")
    description: str = Field(..., description="Detailed description")
    impact: str = Field(..., description="Expected impact level")
    effort: str = Field(..., description="Implementation effort level")
    keywords: List[str] = Field(default_factory=list, description="Related keywords")


class AnalysisMetadata(BaseModel):
    """Metadata about the analysis process."""
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat() if v else None}
    )
    
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    hebrew_ratio: float = Field(..., description="Ratio of Hebrew content (0-1)")
    total_keywords_analyzed: int = Field(..., description="Total keywords analyzed")
    total_search_volume: Optional[int] = Field(default=None, description="Total search volume")
    dominant_themes: List[str] = Field(default_factory=list, description="Main content themes")


class ContentAnalysisResponse(BaseModel):
    """Complete content analysis response."""
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat() if v else None}
    )
    
    success: bool = Field(default=True, description="Analysis success status")
    semantic_clusters: List[SemanticCluster] = Field(..., description="Generated semantic clusters")
    content_gaps: List[ContentGap] = Field(default_factory=list, description="Identified content gaps")
    business_insights: List[BusinessInsight] = Field(default_factory=list, description="Business insights")
    metadata: AnalysisMetadata = Field(..., description="Analysis metadata")
    raw_analysis: Optional[Dict[str, Any]] = Field(default=None, description="Raw analysis data")


class KeywordExpansionResult(BaseModel):
    """Result of keyword expansion."""
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat() if v else None}
    )
    
    original: str = Field(..., description="Original keyword")
    morphological_info: Dict[str, Any] = Field(..., description="Morphological information")
    variations_by_type: Dict[str, List[str]] = Field(..., description="Variations by type")
    all_variations: List[str] = Field(..., description="All variations combined")
    expansion_score: float = Field(..., description="Expansion quality score")


class KeywordExpansionResponse(BaseModel):
    """Response for keyword expansion."""
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat() if v else None}
    )
    
    success: bool = Field(default=True, description="Expansion success status")
    keyword_expansions: Dict[str, KeywordExpansionResult] = Field(..., description="Expansions per keyword")
    combined_variations: List[str] = Field(..., description="Combined keyword variations")
    expansion_metadata: Dict[str, Any] = Field(..., description="Expansion metadata")


class ClusteringResponse(BaseModel):
    """Response for semantic clustering."""
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat() if v else None}
    )
    
    success: bool = Field(default=True, description="Clustering success status")
    semantic_clusters: List[SemanticCluster] = Field(..., description="Generated clusters")
    clustering_metadata: Dict[str, Any] = Field(..., description="Clustering metadata")


class HealthResponse(BaseModel):
    """Health check response."""
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat() if v else None}
    )
    
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Check timestamp")
    version: str = Field(..., description="Service version")
    model_status: Dict[str, Any] = Field(..., description="Model loading status")
    uptime_seconds: Optional[float] = Field(default=None, description="Service uptime")


class ErrorResponse(BaseModel):
    """Error response schema."""
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat() if v else None}
    )
    
    success: bool = Field(default=False, description="Request success status")
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")


class BatchAnalysisResponse(BaseModel):
    """Response for batch analysis."""
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat() if v else None}
    )
    
    success: bool = Field(default=True, description="Batch analysis success status")
    results: List[ContentAnalysisResponse] = Field(..., description="Analysis results per text")
    batch_metadata: Dict[str, Any] = Field(..., description="Batch processing metadata")
    failed_analyses: List[Dict[str, Any]] = Field(default_factory=list, description="Failed analysis details")


class SearchVolumeResponse(BaseModel):
    """Response for search volume lookup."""
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat() if v else None}
    )
    
    success: bool = Field(default=True, description="Lookup success status")
    keywords: List[KeywordData] = Field(..., description="Keywords with volume data")
    metadata: Dict[str, Any] = Field(..., description="Lookup metadata")
    data_source: str = Field(default="DataForSEO", description="Data source")


class ServiceInfoResponse(BaseModel):
    """Service information response."""
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat() if v else None}
    )
    
    service: Dict[str, Any] = Field(..., description="Service information")
    model: Dict[str, Any] = Field(..., description="Model information")
    configuration: Dict[str, Any] = Field(..., description="Configuration details")
    capabilities: List[str] = Field(..., description="Service capabilities")
