"""
Semantic Clustering Service
Groups related Hebrew concepts and keywords into meaningful clusters
for content analysis and SEO optimization.
"""

import asyncio
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict, Counter
from loguru import logger
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

from services.hebrew_analyzer import hebrew_analyzer


class SemanticClusteringService:
    """
    Advanced semantic clustering for Hebrew content.
    Groups related keywords and concepts for SEO and content optimization.
    """
    
    def __init__(self):
        self.min_cluster_size = 3
        self.max_clusters = 10
        self.similarity_threshold = 0.3
        
    async def generate_clusters(self, analysis_results: Dict[str, Any], options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate semantic clusters from Hebrew content analysis.
        
        Args:
            analysis_results: Results from hebrew_analyzer
            options: Clustering options
            
        Returns:
            Semantic clusters with related keywords and concepts
        """
        if not options:
            options = {}
            
        logger.info("Starting semantic clustering analysis")
        
        # Extract keywords and phrases for clustering
        keywords = self._extract_clustering_candidates(analysis_results)
        
        if len(keywords) < self.min_cluster_size:
            logger.warning(f"Not enough keywords ({len(keywords)}) for clustering")
            return self._create_single_cluster(keywords)
        
        # Perform different clustering approaches
        clustering_tasks = [
            self._cluster_by_roots(keywords, analysis_results.get('hebrew_roots', {})),
            self._cluster_by_themes(keywords, analysis_results.get('content_themes', {})),
            self._cluster_by_similarity(keywords),
            self._cluster_by_pos_patterns(keywords, analysis_results.get('tokens', []))
        ]
        
        cluster_results = await asyncio.gather(*clustering_tasks)
        
        # Combine and optimize clusters
        final_clusters = self._merge_and_optimize_clusters(cluster_results)
        
        return {
            'semantic_clusters': final_clusters,
            'clustering_metadata': {
                'total_keywords': len(keywords),
                'clusters_generated': len(final_clusters),
                'clustering_methods': ['root_based', 'theme_based', 'similarity_based', 'pos_based'],
                'avg_cluster_size': np.mean([len(c['keywords']) for c in final_clusters]) if final_clusters else 0
            }
        }
    
    def _extract_clustering_candidates(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract keywords and phrases suitable for clustering."""
        candidates = []
        
        # Add top keywords
        if 'extracted_keywords' in analysis_results:
            for kw in analysis_results['extracted_keywords'].get('top_keywords', []):
                candidates.append({
                    'text': kw['keyword'],
                    'type': 'keyword',
                    'score': kw['score'],
                    'source': 'keyword_extraction'
                })
        
        # Add noun phrases
        if 'semantic_phrases' in analysis_results:
            for phrase in analysis_results['semantic_phrases'].get('noun_phrases', []):
                if len(phrase['text'].split()) >= 2:  # Multi-word phrases
                    candidates.append({
                        'text': phrase['text'],
                        'type': 'phrase',
                        'score': phrase['length'] * 0.5,  # Length-based scoring
                        'source': 'phrase_extraction'
                    })
        
        # Add dominant roots as concepts
        if 'hebrew_roots' in analysis_results:
            for root in analysis_results['hebrew_roots'].get('dominant_roots', []):
                candidates.append({
                    'text': root,
                    'type': 'root',
                    'score': 1.0,
                    'source': 'root_extraction'
                })
        
        # Sort by score and return top candidates
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:50]  # Limit for performance
    
    async def _cluster_by_roots(self, keywords: List[Dict], roots_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Cluster keywords by Hebrew root relationships."""
        clusters = []
        root_details = roots_data.get('root_details', {})
        
        for root, root_info in root_details.items():
            if len(root_info) >= 2:  # At least 2 related words
                cluster_keywords = []
                
                # Find keywords related to this root
                for kw in keywords:
                    for root_word in root_info:
                        if (root_word['lemma'] in kw['text'] or 
                            kw['text'] in root_word['lemma'] or
                            self._are_morphologically_related(kw['text'], root_word['lemma'])):
                            cluster_keywords.append(kw)
                            break
                
                if len(cluster_keywords) >= 2:
                    clusters.append({
                        'cluster_name': f"שורש_{root}",
                        'root_concept': root,
                        'keywords': cluster_keywords,
                        'cluster_type': 'morphological',
                        'coherence_score': self._calculate_root_coherence(cluster_keywords, root)
                    })
        
        return clusters
    
    async def _cluster_by_themes(self, keywords: List[Dict], themes_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Cluster keywords by thematic similarity."""
        clusters = []
        theme_distribution = themes_data.get('theme_distribution', {})
        
        # Define theme-specific keyword patterns
        theme_patterns = {
            'technology': ['טכנולוגי', 'דיגיטלי', 'מחשב', 'אינטרנט', 'תוכנה', 'מערכת'],
            'business': ['עסקי', 'שיווק', 'מכירות', 'לקוח', 'שירות', 'רווח'],
            'education': ['לימודי', 'חינוכי', 'קורס', 'הכשרה', 'מורה', 'תלמיד'],
            'health': ['בריאות', 'רפואי', 'טיפול', 'מחלה', 'רופא'],
            'home': ['בית', 'מטבח', 'עיצוב', 'ריהוט', 'דירה', 'חדר']
        }
        
        for theme, patterns in theme_patterns.items():
            if theme_distribution.get(theme, 0) > 5:  # Theme is significant
                theme_keywords = []
                
                for kw in keywords:
                    if any(pattern in kw['text'] for pattern in patterns):
                        theme_keywords.append(kw)
                
                if len(theme_keywords) >= 2:
                    clusters.append({
                        'cluster_name': f"נושא_{theme}",
                        'root_concept': theme,
                        'keywords': theme_keywords,
                        'cluster_type': 'thematic',
                        'coherence_score': theme_distribution.get(theme, 0) / 100
                    })
        
        return clusters
    
    async def _cluster_by_similarity(self, keywords: List[Dict]) -> List[Dict[str, Any]]:
        """Cluster keywords by semantic similarity using TF-IDF."""
        if len(keywords) < 3:
            return []
        
        # Prepare text data
        texts = [kw['text'] for kw in keywords]
        
        try:
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Use DBSCAN for density-based clustering
            clustering = DBSCAN(
                eps=1 - self.similarity_threshold,
                min_samples=2,
                metric='precomputed'
            )
            
            distance_matrix = 1 - similarity_matrix
            cluster_labels = clustering.fit_predict(distance_matrix)
            
            # Group keywords by cluster
            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                if label != -1:  # Not noise
                    clusters[label].append(keywords[i])
            
            # Convert to result format
            result_clusters = []
            for cluster_id, cluster_keywords in clusters.items():
                if len(cluster_keywords) >= 2:
                    # Find most representative keyword as cluster name
                    cluster_name = max(cluster_keywords, key=lambda x: x['score'])['text']
                    
                    result_clusters.append({
                        'cluster_name': f"דמיון_{cluster_name}",
                        'root_concept': cluster_name,
                        'keywords': cluster_keywords,
                        'cluster_type': 'similarity',
                        'coherence_score': self._calculate_similarity_coherence(cluster_keywords, similarity_matrix)
                    })
            
            return result_clusters
            
        except Exception as e:
            logger.error(f"Similarity clustering failed: {e}")
            return []
    
    async def _cluster_by_pos_patterns(self, keywords: List[Dict], tokens: List[Dict]) -> List[Dict[str, Any]]:
        """Cluster keywords by part-of-speech patterns."""
        clusters = defaultdict(list)
        
        # Create POS mapping from tokens
        pos_mapping = {}
        for token in tokens:
            # Safe access to token fields
            is_hebrew = token.get('is_hebrew', False)
            lemma = token.get('lemma', '')
            pos = token.get('pos', 'UNKNOWN')
            
            if is_hebrew and lemma:
                pos_mapping[lemma] = pos
        
        # Group keywords by POS
        for kw in keywords:
            kw_text = kw.get('keyword', kw.get('text', str(kw) if isinstance(kw, str) else ''))
            pos = pos_mapping.get(kw_text, 'UNKNOWN')
            if pos in ['NOUN', 'ADJ', 'VERB']:
                clusters[pos].append(kw)
        
        # Convert to result format
        result_clusters = []
        pos_names = {'NOUN': 'שמות_עצם', 'ADJ': 'תארים', 'VERB': 'פעלים'}
        
        for pos, cluster_keywords in clusters.items():
            if len(cluster_keywords) >= 3:
                result_clusters.append({
                    'cluster_name': pos_names.get(pos, pos),
                    'root_concept': pos,
                    'keywords': cluster_keywords,
                    'cluster_type': 'grammatical',
                    'coherence_score': self._calculate_grammatical_coherence(cluster_keywords)  # Real calculation instead of fixed 0.7
                })
        
        return result_clusters
    
    def _merge_and_optimize_clusters(self, cluster_results: List[List[Dict]]) -> List[Dict[str, Any]]:
        """Merge and optimize clusters from different methods."""
        all_clusters = []
        for result in cluster_results:
            all_clusters.extend(result)
        
        if not all_clusters:
            return []
        
        # Remove duplicate keywords across clusters
        seen_keywords = set()
        optimized_clusters = []
        
        # Sort clusters by coherence score
        all_clusters.sort(key=lambda x: x['coherence_score'], reverse=True)
        
        for cluster in all_clusters:
            # Filter out already seen keywords
            unique_keywords = []
            for kw in cluster['keywords']:
                if kw.get('keyword', kw.get('text', '')) not in seen_keywords:
                    unique_keywords.append(kw)
                    seen_keywords.add(kw.get('keyword', kw.get('text', '')))
            
            # Keep cluster if it still has enough keywords
            if len(unique_keywords) >= 2:
                cluster['keywords'] = unique_keywords
                optimized_clusters.append(cluster)
                
                # Limit total number of clusters
                if len(optimized_clusters) >= self.max_clusters:
                    break
        
        return optimized_clusters
    
    def _are_morphologically_related(self, word1: str, word2: str) -> float:
        """Check if two Hebrew words are morphologically related."""
        # Simple heuristic: check for common substrings of length 2+
        if len(word1) < 2 or len(word2) < 2:
            return 0.0
            
        # Find longest common substring
        common_length = 0
        for i in range(len(word1)):
            for j in range(len(word2)):
                k = 0
                while (i + k < len(word1) and 
                       j + k < len(word2) and 
                       word1[i + k] == word2[j + k]):
                    k += 1
                common_length = max(common_length, k)
        
        # Return similarity score (0.0 to 1.0)
        max_length = max(len(word1), len(word2))
        if max_length == 0:
            return 0.0
        
        # Base similarity on common substring ratio
        similarity = common_length / max_length
        
        # Bonus for exact match
        if word1 == word2:
            return 1.0
            
        # Minimum threshold for considering words related
        return max(similarity, 0.1)  # Give minimum score for any pair
    
    def _calculate_root_coherence(self, keywords: List[Dict], root: str) -> float:
        """Calculate coherence score for root-based cluster."""
        if not keywords:
            return 0.0
            
        # Higher score for more keywords and higher individual scores
        avg_score = np.mean([kw['score'] for kw in keywords])
        size_bonus = min(len(keywords) / 5, 1.0)
        
        return (avg_score + size_bonus) / 2
    
    def _calculate_similarity_coherence(self, keywords: List[Dict], similarity_matrix: np.ndarray) -> float:
        """Calculate coherence score for similarity-based cluster."""
        if len(keywords) < 2:
            return 0.0
            
        # Calculate average pairwise similarity within cluster
        indices = list(range(len(keywords)))
        similarities = []
        
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                similarities.append(similarity_matrix[indices[i]][indices[j]])
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_grammatical_coherence(self, keywords: List[Dict]) -> float:
        """Calculate coherence score for grammatical cluster."""
        if len(keywords) < 2:
            return 0.0
            
        # Calculate average pairwise similarity within cluster
        similarities = []
        
        for i in range(len(keywords)):
            for j in range(i + 1, len(keywords)):
                word1 = keywords[i].get('keyword', keywords[i].get('text', ''))
                word2 = keywords[j].get('keyword', keywords[j].get('text', ''))
                similarities.append(self._are_morphologically_related(word1, word2))
        
        return np.mean(similarities) if similarities else 0.0
    
    def _create_single_cluster(self, keywords: List[Dict]) -> Dict[str, Any]:
        """Create a single cluster when not enough keywords for clustering."""
        if not keywords:
            return {'semantic_clusters': [], 'clustering_metadata': {'total_keywords': 0}}
            
        return {
            'semantic_clusters': [{
                'cluster_name': 'תוכן_כללי',
                'root_concept': 'general_content',
                'keywords': keywords,
                'cluster_type': 'single',
                'coherence_score': self._calculate_grammatical_coherence(keywords)  # Real calculation instead of fixed 0.5
            }],
            'clustering_metadata': {
                'total_keywords': len(keywords),
                'clusters_generated': 1,
                'clustering_methods': ['single_cluster'],
                'avg_cluster_size': len(keywords)
            }
        }


# Global clustering service instance
semantic_clustering = SemanticClusteringService()
