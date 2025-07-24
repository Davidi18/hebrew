"""
Hebrew Semantic Analyzer
Advanced Hebrew text analysis using Hebrew Transformers with morphological analysis,
root extraction, NER, and semantic phrase extraction.
"""

import re
import asyncio
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import defaultdict, Counter
from loguru import logger
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from models.hebspacy_loader import hebrew_loader


class HebrewSemanticAnalyzer:
    """
    Advanced Hebrew content analysis engine.
    Provides morphological analysis, root extraction, NER, and semantic clustering.
    """
    
    def __init__(self):
        self.stop_words = self._load_hebrew_stop_words()
        self.common_roots = self._load_common_hebrew_roots()
        
    def _load_hebrew_stop_words(self) -> Set[str]:
        """Load Hebrew stop words."""
        return {
            'של', 'את', 'על', 'אל', 'כל', 'לא', 'זה', 'זו', 'זאת', 'הוא', 'היא', 'הם', 'הן',
            'אני', 'אתה', 'את', 'אנחנו', 'אתם', 'אתן', 'יש', 'אין', 'היה', 'הייה', 'יהיה',
            'תהיה', 'עם', 'בין', 'אחר', 'אחרי', 'לפני', 'תחת', 'מעל', 'ליד', 'בתוך', 'מחוץ',
            'גם', 'רק', 'אבל', 'או', 'כי', 'אם', 'מה', 'מי', 'איך', 'איפה', 'מתי', 'למה',
            'כמה', 'איזה', 'איזו', 'אילו', 'הזה', 'הזו', 'הזאת', 'ההוא', 'ההיא', 'ההם', 'ההן'
        }
    
    def _load_common_hebrew_roots(self) -> Dict[str, List[str]]:
        """Load common Hebrew roots and their variations."""
        return {
            'כתב': ['כתב', 'כותב', 'כתיבה', 'מכתב', 'כתבה'],
            'למד': ['למד', 'לומד', 'למידה', 'מלמד', 'תלמיד'],
            'עבד': ['עבד', 'עובד', 'עבודה', 'מעבד', 'עובדת'],
            'בנה': ['בנה', 'בונה', 'בניה', 'מבנה', 'בנין'],
            'שמר': ['שמר', 'שומר', 'שמירה', 'משמר', 'שמרה'],
            'דבר': ['דבר', 'דובר', 'דיבור', 'מדבר', 'דברה'],
            'קרא': ['קרא', 'קורא', 'קריאה', 'מקרא', 'קראה'],
            'ראה': ['ראה', 'רואה', 'ראיה', 'מראה', 'ראתה']
        }
    
    async def analyze_content(self, text: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Comprehensive Hebrew content analysis.
        
        Args:
            text: Hebrew text to analyze
            options: Analysis options
            
        Returns:
            Comprehensive analysis results
        """
        if not options:
            options = {}
            
        logger.info(f"Starting Hebrew content analysis for {len(text)} characters")
        
        # Get basic Hebrew Transformers analysis
        basic_analysis = await hebrew_loader.analyze_text(text)
        
        # Get tokens safely with fallback
        tokens = basic_analysis.get('tokens', [])
        
        # Perform advanced analysis only if we have tokens
        if tokens:
            analysis_tasks = [
                self._extract_hebrew_roots(tokens),
                self._extract_semantic_phrases(tokens),
                self._analyze_morphology(tokens),
                self._extract_keywords(tokens),
                self._analyze_content_themes(tokens)
            ]
            
            results = await asyncio.gather(*analysis_tasks)
        else:
            # Fallback results if no tokens available
            results = [
                {'roots': [], 'root_frequency': {}},
                {'phrases': []},
                {'morphology': {}},
                {'keywords': []},
                {'themes': []}
            ]
        
        return {
            **basic_analysis,
            'hebrew_roots': results[0],
            'semantic_phrases': results[1],
            'morphological_analysis': results[2],
            'extracted_keywords': results[3],
            'content_themes': results[4],
            'analysis_metadata': {
                'processing_time_ms': 0,  # Will be calculated by caller
                'hebrew_ratio': basic_analysis.get('language_stats', {}).get('hebrew_ratio', 0.0),
                'complexity_score': self._calculate_complexity_score(tokens)
            }
        }
    
    async def _extract_hebrew_roots(self, tokens: List[Dict]) -> Dict[str, Any]:
        """Extract Hebrew roots (שורשים) from tokens."""
        roots_found = defaultdict(list)
        root_frequency = Counter()
        
        for token in tokens:
            # Safe access to token fields
            is_hebrew = token.get('is_hebrew', False)
            token_text = token.get('text', str(token) if isinstance(token, str) else '')
            
            if not is_hebrew or token_text in self.stop_words:
                continue
                
            # Try to find root using lemma
            lemma = token.get('lemma', token_text)
            if lemma and len(lemma) >= 2:
                # Check against known roots
                for root, variations in self.common_roots.items():
                    if lemma in variations or any(var in lemma for var in variations):
                        roots_found[root].append({
                            'token': token_text,
                            'lemma': lemma,
                            'pos': token.get('pos', 'UNKNOWN')
                        })
                        root_frequency[root] += 1
                        break
                else:
                    # Extract potential root using morphological patterns
                    potential_root = self._extract_root_pattern(lemma)
                    if potential_root and len(potential_root) >= 2:
                        roots_found[potential_root].append({
                            'token': token_text,
                            'lemma': lemma,
                            'pos': token.get('pos', 'UNKNOWN')
                        })
                        root_frequency[potential_root] += 1
        
        return {
            'roots_by_frequency': dict(root_frequency.most_common(20)),
            'root_details': dict(roots_found),
            'total_roots_found': len(roots_found),
            'dominant_roots': [root for root, count in root_frequency.most_common(5)]
        }
    
    def _extract_root_pattern(self, word: str) -> Optional[str]:
        """Extract potential Hebrew root using morphological patterns."""
        if len(word) < 3:
            return None
            
        # Remove common prefixes and suffixes
        prefixes = ['ה', 'ו', 'ב', 'כ', 'ל', 'מ', 'ש']
        suffixes = ['ים', 'ות', 'ה', 'ת', 'ן', 'ך', 'נו', 'כם', 'הן']
        
        cleaned = word
        
        # Remove prefixes
        for prefix in prefixes:
            if cleaned.startswith(prefix) and len(cleaned) > len(prefix) + 2:
                cleaned = cleaned[len(prefix):]
                break
        
        # Remove suffixes
        for suffix in suffixes:
            if cleaned.endswith(suffix) and len(cleaned) > len(suffix) + 2:
                cleaned = cleaned[:-len(suffix)]
                break
        
        # Return potential 3-letter root
        if len(cleaned) >= 3:
            return cleaned[:3]
        
        return None
    
    async def _extract_semantic_phrases(self, tokens: List[Dict]) -> Dict[str, Any]:
        """Extract meaningful Hebrew phrases and collocations."""
        phrases = []
        noun_phrases = []
        
        # Extract noun phrases (sequences of adjectives + nouns)
        current_phrase = []
        for i, token in enumerate(tokens):
            if not token['is_hebrew']:
                if current_phrase:
                    if len(current_phrase) >= 2:
                        phrase_text = ' '.join([t['text'] for t in current_phrase])
                        noun_phrases.append({
                            'text': phrase_text,
                            'tokens': current_phrase.copy(),
                            'length': len(current_phrase)
                        })
                    current_phrase = []
                continue
                
            pos = token['pos']
            if pos in ['NOUN', 'ADJ', 'PROPN']:
                current_phrase.append(token)
            else:
                if current_phrase and len(current_phrase) >= 2:
                    phrase_text = ' '.join([t['text'] for t in current_phrase])
                    noun_phrases.append({
                        'text': phrase_text,
                        'tokens': current_phrase.copy(),
                        'length': len(current_phrase)
                    })
                current_phrase = []
        
        # Extract verb phrases
        verb_phrases = []
        for i, token in enumerate(tokens):
            if token['pos'] == 'VERB' and token['is_hebrew']:
                # Look for surrounding context
                context_start = max(0, i - 2)
                context_end = min(len(tokens), i + 3)
                context_tokens = tokens[context_start:context_end]
                
                verb_phrase = ' '.join([t['text'] for t in context_tokens if t['is_hebrew']])
                if len(verb_phrase.split()) >= 2:
                    verb_phrases.append({
                        'text': verb_phrase,
                        'main_verb': token['text'],
                        'lemma': token['lemma']
                    })
        
        return {
            'noun_phrases': sorted(noun_phrases, key=lambda x: x['length'], reverse=True)[:15],
            'verb_phrases': verb_phrases[:10],
            'total_phrases': len(noun_phrases) + len(verb_phrases)
        }
    
    async def _analyze_morphology(self, tokens: List[Dict]) -> Dict[str, Any]:
        """Analyze Hebrew morphological patterns."""
        pos_distribution = Counter()
        morphological_features = defaultdict(Counter)
        
        for token in tokens:
            if not token['is_hebrew']:
                continue
                
            pos_distribution[token['pos']] += 1
            
            # Parse morphological features
            morph_data = token.get('morph', '')
if morph_data:
                morph_features = morph_data.split('|')
                for feature in morph_features:
                    if '=' in feature:
                        key, value = feature.split('=', 1)
                        morphological_features[key][value] += 1
        
        return {
            'pos_distribution': dict(pos_distribution),
            'morphological_features': {k: dict(v) for k, v in morphological_features.items()},
            'complexity_indicators': {
                'avg_word_length': np.mean([len(t['text']) for t in tokens if t['is_hebrew']]),
                'morphological_richness': len(morphological_features),
                'pos_diversity': len(pos_distribution)
            }
        }
    
    async def _extract_keywords(self, tokens: List[Dict]) -> Dict[str, Any]:
        """Extract important Hebrew keywords using frequency and linguistic features."""
        # Filter Hebrew content words
        content_words = []
        for token in tokens:
            # Safe access to token fields
            is_hebrew = token.get('is_hebrew', False)
            is_alpha = token.get('is_alpha', True)
            is_stop = token.get('is_stop', False)
            token_text = token.get('text', str(token) if isinstance(token, str) else '')
            pos = token.get('pos', 'UNKNOWN')
            
            if (is_hebrew and 
                is_alpha and 
                not is_stop and
                token_text not in self.stop_words and
                len(token_text) >= 2 and
                pos in ['NOUN', 'ADJ', 'VERB', 'PROPN']):
                content_words.append(token)
        
        # Calculate keyword scores
        word_freq = Counter([token.get('lemma', token.get('text', str(token) if isinstance(token, str) else '')) for token in content_words])
        pos_weights = {'NOUN': 1.5, 'PROPN': 1.8, 'ADJ': 1.2, 'VERB': 1.0}
        
        keyword_scores = {}
        for token in content_words:
            lemma = token.get('lemma', token.get('text', str(token) if isinstance(token, str) else ''))
            base_score = word_freq[lemma]
            pos_weight = pos_weights.get(token.get('pos', 'UNKNOWN'), 1.0)
            token_text = token.get('text', str(token) if isinstance(token, str) else '')
            length_bonus = min(len(token_text) / 10, 0.5)  # Longer words get slight bonus
            
            keyword_scores[lemma] = base_score * pos_weight + length_bonus
        
        # Sort by score
        top_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)[:20]
        
        return {
            'top_keywords': [{'keyword': k, 'score': round(v, 2)} for k, v in top_keywords],
            'keyword_density': len(content_words) / len(tokens) if tokens else 0,
            'unique_keywords': len(keyword_scores)
        }
    
    async def _analyze_content_themes(self, tokens: List[Dict]) -> Dict[str, Any]:
        """Identify main content themes using dynamic semantic clustering."""
        # Extract meaningful Hebrew lemmas
        hebrew_lemmas = []
        for token in tokens:
            if (token.get('is_hebrew', False) and 
                token.get('is_alpha', True) and
                not token.get('is_stop', False) and
                token.get('text', '') not in self.stop_words and
                len(token.get('text', '')) >= 3 and
                token.get('pos', '') in ['NOUN', 'ADJ', 'VERB', 'PROPN']):
                
                lemma = token.get('lemma', token.get('text', ''))
                hebrew_lemmas.append(lemma)
        
        if len(hebrew_lemmas) < 5:  # Not enough content for theme analysis
            return {
                'themes': [],
                'theme_clusters': {},
                'semantic_density': 0.0,
                'theme_coherence': 0.0
            }
        
        # Create text for TF-IDF analysis
        text_for_analysis = ' '.join(hebrew_lemmas)
        
        # Use TF-IDF to find important terms
        try:
            # Create word frequency distribution
            word_freq = Counter(hebrew_lemmas)
            total_words = len(hebrew_lemmas)
            
            # Calculate TF-IDF-like scores manually for Hebrew
            term_scores = {}
            for word, freq in word_freq.items():
                tf = freq / total_words
                # Simple IDF approximation based on word length and frequency
                idf = np.log(total_words / freq) + (len(word) / 10)
                term_scores[word] = tf * idf
            
            # Get top terms
            top_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)[:15]
            
            # Dynamic theme clustering based on co-occurrence
            theme_clusters = self._cluster_semantic_terms([term[0] for term in top_terms], hebrew_lemmas)
            
            # Calculate semantic metrics
            unique_lemmas = len(set(hebrew_lemmas))
            semantic_density = unique_lemmas / len(hebrew_lemmas) if hebrew_lemmas else 0
            
            # Theme coherence based on cluster quality
            theme_coherence = self._calculate_theme_coherence(theme_clusters, hebrew_lemmas)
            
            return {
                'themes': [{'theme': f'cluster_{i}', 'terms': cluster, 'strength': len(cluster)} 
                          for i, cluster in enumerate(theme_clusters) if len(cluster) >= 2],
                'top_semantic_terms': [{'term': term, 'score': round(score, 3)} for term, score in top_terms[:10]],
                'semantic_density': round(semantic_density, 3),
                'theme_coherence': round(theme_coherence, 3),
                'total_unique_concepts': unique_lemmas
            }
            
        except Exception as e:
            logger.warning(f"Theme analysis failed: {e}")
            return {
                'themes': [],
                'top_semantic_terms': [],
                'semantic_density': 0.0,
                'theme_coherence': 0.0,
                'total_unique_concepts': 0
            }
    
    def _cluster_semantic_terms(self, top_terms: List[str], all_lemmas: List[str]) -> List[List[str]]:
        """Cluster semantically related terms based on co-occurrence patterns."""
        if len(top_terms) < 3:
            return [top_terms] if top_terms else []
        
        # Create co-occurrence matrix
        co_occurrence = defaultdict(lambda: defaultdict(int))
        
        # Sliding window approach for co-occurrence
        window_size = 5
        for i in range(len(all_lemmas) - window_size + 1):
            window = all_lemmas[i:i + window_size]
            window_terms = [term for term in window if term in top_terms]
            
            # Count co-occurrences within window
            for j, term1 in enumerate(window_terms):
                for term2 in window_terms[j+1:]:
                    co_occurrence[term1][term2] += 1
                    co_occurrence[term2][term1] += 1
        
        # Simple clustering based on co-occurrence strength
        clusters = []
        used_terms = set()
        
        for term in top_terms:
            if term in used_terms:
                continue
                
            cluster = [term]
            used_terms.add(term)
            
            # Find strongly co-occurring terms
            if term in co_occurrence:
                related_terms = sorted(co_occurrence[term].items(), 
                                     key=lambda x: x[1], reverse=True)
                
                for related_term, strength in related_terms[:3]:  # Top 3 related
                    if related_term not in used_terms and strength >= 2:
                        cluster.append(related_term)
                        used_terms.add(related_term)
            
            if len(cluster) >= 1:  # Keep even single-term clusters
                clusters.append(cluster)
        
        return clusters
    
    def _calculate_theme_coherence(self, theme_clusters: List[List[str]], all_lemmas: List[str]) -> float:
        """Calculate how coherent the identified themes are."""
        if not theme_clusters or not all_lemmas:
            return 0.0
        
        # Calculate coherence based on cluster sizes and distribution
        total_clustered_terms = sum(len(cluster) for cluster in theme_clusters)
        unique_terms = len(set(all_lemmas))
        
        if unique_terms == 0:
            return 0.0
        
        # Coherence is higher when we have fewer, larger clusters
        cluster_quality = 0
        for cluster in theme_clusters:
            if len(cluster) > 1:
                cluster_quality += len(cluster) ** 1.5  # Reward larger clusters
        
        coherence = min(cluster_quality / (unique_terms * 2), 1.0)
        return coherence
    
    def _calculate_complexity_score(self, tokens: List[Dict]) -> float:
        """Calculate content complexity score (0-1)."""
        if not tokens:
            return 0.0
            
        hebrew_tokens = [t for t in tokens if t.get('is_hebrew', False)]
        if not hebrew_tokens:
            return 0.0
            
        # Factors for complexity
        avg_word_length = np.mean([len(t.get('text', '')) for t in hebrew_tokens])
        pos_diversity = len(set(t.get('pos', 'UNKNOWN') for t in hebrew_tokens))
        morphological_complexity = sum(1 for t in hebrew_tokens if t.get('morph', ''))
        
        # Normalize and combine
        length_score = min(avg_word_length / 8, 1.0)  # Normalize to 0-1
        pos_score = min(pos_diversity / 10, 1.0)  # Normalize to 0-1
        morph_score = morphological_complexity / max(len(hebrew_tokens), 1)  # Safe division
        
        complexity = (length_score + pos_score + morph_score) / 3
        return round(complexity, 3)


# Global analyzer instance
hebrew_analyzer = HebrewSemanticAnalyzer()
