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
            if not token['is_hebrew'] or token['text'] in self.stop_words:
                continue
                
            # Try to find root using lemma
            lemma = token['lemma']
            if lemma and len(lemma) >= 2:
                # Check against known roots
                for root, variations in self.common_roots.items():
                    if lemma in variations or any(var in lemma for var in variations):
                        roots_found[root].append({
                            'token': token['text'],
                            'lemma': lemma,
                            'pos': token['pos']
                        })
                        root_frequency[root] += 1
                        break
                else:
                    # Extract potential root using morphological patterns
                    potential_root = self._extract_root_pattern(lemma)
                    if potential_root and len(potential_root) >= 2:
                        roots_found[potential_root].append({
                            'token': token['text'],
                            'lemma': lemma,
                            'pos': token['pos']
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
            if token['morph']:
                morph_features = token['morph'].split('|')
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
        """Identify main content themes and topics."""
        # Group related words by semantic fields
        semantic_fields = {
            'technology': ['מחשב', 'טכנולוגיה', 'אינטרנט', 'תוכנה', 'מערכת', 'דיגיטלי'],
            'business': ['עסק', 'חברה', 'שיווק', 'מכירות', 'רווח', 'לקוח', 'שירות'],
            'education': ['לימוד', 'חינוך', 'בית ספר', 'מורה', 'תלמיד', 'קורס'],
            'health': ['בריאות', 'רפואה', 'רופא', 'חולה', 'טיפול', 'מחלה'],
            'home': ['בית', 'מטבח', 'חדר', 'עיצוב', 'ריהוט', 'דירה']
        }
        
        theme_scores = defaultdict(int)
        lemmas = [token.get('lemma', token.get('text', str(token) if isinstance(token, str) else '')).lower() for token in tokens if token.get('is_hebrew', False)]
        
        for theme, keywords in semantic_fields.items():
            for keyword in keywords:
                theme_scores[theme] += lemmas.count(keyword)
        
        # Find dominant themes
        total_theme_words = sum(theme_scores.values())
        theme_percentages = {
            theme: (score / total_theme_words * 100) if total_theme_words > 0 else 0
            for theme, score in theme_scores.items()
        }
        
        dominant_themes = sorted(theme_percentages.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'theme_distribution': dict(theme_percentages),
            'dominant_themes': [theme for theme, pct in dominant_themes[:3] if pct > 5],
            'theme_diversity': len([t for t in theme_percentages.values() if t > 0])
        }
    
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
        morph_score = morphological_complexity / len(hebrew_tokens)
        
        complexity = (length_score + pos_score + morph_score) / 3
        return round(complexity, 3)


# Global analyzer instance
hebrew_analyzer = HebrewSemanticAnalyzer()
