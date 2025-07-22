"""
Semantic Clustering API endpoints for Hebrew Content Intelligence Service.
Provides clustering functionality for Hebrew keywords and concepts.
"""

import time
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException
from loguru import logger

from schemas.requests import ClusterGenerationRequest
from schemas.responses import ClusteringResponse, SemanticCluster, KeywordData
from services.semantic_clusters import semantic_clustering

router = APIRouter()


@router.post("/clusters", response_model=ClusteringResponse)
async def generate_clusters(request: ClusterGenerationRequest):
    """
    Generate semantic clusters from Hebrew keywords.
    
    Groups related keywords using:
    - Hebrew root analysis
    - Thematic similarity
    - Morphological relationships
    - Semantic similarity
    """
    start_time = time.time()
    
    try:
        logger.info(f"Starting clustering for {len(request.keywords)} keywords")
        
        # Create mock analysis results for clustering
        # In a real implementation, this would come from the Hebrew analyzer
        mock_analysis = {
            'extracted_keywords': {
                'top_keywords': [{'keyword': kw, 'score': 1.0} for kw in request.keywords]
            },
            'hebrew_roots': {
                'dominant_roots': [],
                'root_details': {}
            },
            'content_themes': {
                'theme_distribution': {},
                'dominant_themes': []
            },
            'tokens': [
                {
                    'text': kw,
                    'lemma': kw,
                    'pos': 'NOUN',
                    'is_hebrew': True
                } for kw in request.keywords
            ]
        }
        
        # Generate clusters
        clustering_result = await semantic_clustering.generate_clusters(
            mock_analysis,
            options=request.options
        )
        
        # Convert to response format
        semantic_clusters = []
        for cluster in clustering_result.get('semantic_clusters', []):
            # Convert keywords to KeywordData format
            primary_keywords = []
            for kw in cluster.get('keywords', []):
                primary_keywords.append(KeywordData(
                    keyword=kw.get('text', ''),
                    volume=None,  # Will be populated with DataForSEO
                    difficulty=None,
                    cpc=None,
                    competition=None
                ))
            
            semantic_clusters.append(SemanticCluster(
                cluster_name=cluster.get('cluster_name', ''),
                root_concept=cluster.get('root_concept', ''),
                primary_keywords=primary_keywords,
                variations=[],
                related_concepts=[],
                cluster_type=cluster.get('cluster_type', 'unknown'),
                coherence_score=cluster.get('coherence_score', 0.0)
            ))
        
        processing_time = int((time.time() - start_time) * 1000)
        clustering_metadata = clustering_result.get('clustering_metadata', {})
        clustering_metadata['processing_time_ms'] = processing_time
        
        logger.info(f"Clustering completed in {processing_time}ms, generated {len(semantic_clusters)} clusters")
        
        return ClusteringResponse(
            semantic_clusters=semantic_clusters,
            clustering_metadata=clustering_metadata
        )
        
    except Exception as e:
        logger.error(f"Clustering failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Clustering failed: {str(e)}"
        )


@router.get("/clusters/methods")
async def get_clustering_methods():
    """
    Get information about available clustering methods.
    """
    return {
        "available_methods": [
            {
                "name": "root_based",
                "description": "Clusters keywords by Hebrew root relationships (שורש המילה)",
                "best_for": "Morphologically related Hebrew words"
            },
            {
                "name": "theme_based", 
                "description": "Clusters keywords by thematic similarity",
                "best_for": "Content with clear topical themes"
            },
            {
                "name": "similarity_based",
                "description": "Clusters keywords using TF-IDF and cosine similarity",
                "best_for": "General semantic similarity"
            },
            {
                "name": "pos_based",
                "description": "Clusters keywords by part-of-speech patterns",
                "best_for": "Grammatical categorization"
            }
        ],
        "default_parameters": {
            "min_cluster_size": 3,
            "max_clusters": 10,
            "similarity_threshold": 0.3
        },
        "supported_languages": ["Hebrew", "Mixed Hebrew-English"]
    }
