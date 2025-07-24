"""
Hebrew Keyword Expansion Service
Generates Hebrew keyword variations using morphological analysis,
root patterns, and semantic relationships for SEO optimization.
"""

import asyncio
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import defaultdict, Counter
from loguru import logger
import re

from models.hebspacy_loader import hebrew_loader


class HebrewKeywordExpander:
    """
    Advanced Hebrew keyword expansion engine.
    Generates morphological variations, semantic expansions, and related terms.
    """
    
    def __init__(self):
        self.morphological_patterns = self._load_morphological_patterns()
        self.semantic_relations = self._load_semantic_relations()
        self.common_prefixes = ['ה', 'ו', 'ב', 'כ', 'ל', 'מ', 'ש', 'ת']
        self.common_suffixes = ['ים', 'ות', 'ה', 'ת', 'ן', 'ך', 'נו', 'כם', 'הן', 'יה', 'ית']
        self.max_variations_per_type = 10  # New attribute
        
    def _load_morphological_patterns(self) -> Dict[str, List[str]]:
        """Load Hebrew morphological transformation patterns."""
        return {
            # Noun patterns
            'masculine_plural': ['ים', 'י'],
            'feminine_plural': ['ות'],
            'construct_state': ['ת', 'י'],
            'possessive': ['ו', 'ה', 'ם', 'ן', 'ך', 'נו', 'כם', 'הן'],
            
            # Verb patterns
            'past_tense': ['תי', 'ת', 'ה', 'נו', 'תם', 'תן', 'ו'],
            'present_tense': ['ת', 'ים', 'ות'],
            'future_tense': ['א', 'ת', 'י', 'נ'],
            'infinitive': ['ל'],
            
            # Adjective patterns
            'feminine_adj': ['ה', 'ת'],
            'plural_adj': ['ים', 'ות'],
            
            # Common prefixes for verbs
            'verb_prefixes': ['מ', 'נ', 'ה', 'ת'],
            
            # Participle patterns
            'participle': ['מ']
        }
    
    def _load_semantic_relations(self) -> Dict[str, Dict[str, List[str]]]:
        """Load semantic relationship mappings."""
        return {
            'synonyms': {
                'בית': ['דירה', 'מגורים', 'מעון', 'משכן'],
                'מטבח': ['מבשלה', 'בית מדרש לבישול'],
                'עיצוב': ['עיצוב פנים', 'דיזיין', 'תכנון'],
                'חדר': ['חלל', 'מרחב', 'אזור'],
                'מחיר': ['עלות', 'תמחור', 'מחירון'],
                'שירות': ['סרביס', 'טיפול', 'מתן שירות'],
                'איכות': ['רמה', 'סטנדרט', 'מעמד']
            },
            'related_terms': {
                'מטבח': ['בישול', 'אוכל', 'ארונות', 'משטח', 'כיור', 'תנור'],
                'עיצוב': ['אדריכלות', 'תכנון', 'סגנון', 'אסתטיקה'],
                'בית': ['נדלן', 'דיור', 'משכנתא', 'שכירות'],
                'טכנולוגיה': ['דיגיטל', 'חדשנות', 'אוטומציה', 'חכם'],
                'עסק': ['חברה', 'ארגון', 'מיזם', 'יזמות']
            },
            'commercial_intent': {
                'מחיר': ['זול', 'הנחה', 'מבצע', 'עלות', 'תקציב'],
                'קנייה': ['רכישה', 'הזמנה', 'מכירה', 'חנות'],
                'שירות': ['מומחה', 'מקצועי', 'איכותי', 'מהיר'],
                'איכות': ['טוב ביותר', 'מומלץ', 'מעולה', 'איכותי']
            }
        }
    
    async def expand_keywords(self, keywords: List[str], options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Expand Hebrew keywords with morphological and semantic variations.
        
        Args:
            keywords: List of Hebrew keywords to expand
            options: Expansion options
            
        Returns:
            Expanded keywords with variations and metadata
        """
        if not options:
            options = {}
            
        logger.info(f"Starting keyword expansion for {len(keywords)} keywords")
        
        expanded_results = {}
        
        # Process each keyword
        for keyword in keywords:
            keyword_variations = await self._expand_single_keyword(keyword, options)
            expanded_results[keyword] = keyword_variations
        
        # Generate combined variations
        combined_variations = self._generate_combined_variations(expanded_results, options)
        
        return {
            'keyword_expansions': expanded_results,
            'combined_variations': combined_variations,
            'expansion_metadata': {
                'original_keywords': len(keywords),
                'total_variations': sum(len(v['all_variations']) for v in expanded_results.values()),
                'expansion_methods': ['morphological', 'semantic', 'commercial', 'locational']
            }
        }
    
    async def _expand_single_keyword(self, keyword: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Expand a single keyword with strict limits and filtering."""
        if not keyword or not keyword.strip():
            return {'morphological': [], 'semantic': [], 'commercial': [], 'locational': [], 'question': [], 'all_variations': []}
        
        options = options or {}
        
        # Generate all variation types with limits
        morphological_task = self._generate_morphological_variations(keyword, options)
        semantic_task = self._generate_semantic_variations(keyword, options)
        commercial_task = self._generate_commercial_variations(keyword, options)
        locational_task = self._generate_locational_variations(keyword, options)
        question_task = self._generate_question_variations(keyword, options)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(
            morphological_task,
            semantic_task,
            commercial_task,
            locational_task,
            question_task,
            return_exceptions=True
        )
        
        # Handle any exceptions
        morphological = results[0] if not isinstance(results[0], Exception) else []
        semantic = results[1] if not isinstance(results[1], Exception) else []
        commercial = results[2] if not isinstance(results[2], Exception) else []
        locational = results[3] if not isinstance(results[3], Exception) else []
        question = results[4] if not isinstance(results[4], Exception) else []
        
        # Apply strict limits per category
        morphological = morphological[:6]  # Max 6 morphological
        semantic = semantic[:4]            # Max 4 semantic
        commercial = commercial[:12]       # Max 12 commercial
        locational = locational[:8]        # Max 8 locational
        question = question[:3]            # Max 3 questions
        
        # Combine all variations with final filtering
        all_variations = []
        all_variations.extend(morphological)
        all_variations.extend(semantic)
        all_variations.extend(commercial)
        all_variations.extend(locational)
        all_variations.extend(question)
        
        # Remove duplicates and apply final relevance filtering
        unique_variations = []
        seen = set()
        for variation in all_variations:
            if variation not in seen and variation != keyword:
                score = self._calculate_relevance_score(keyword, variation)
                if score > 0.3:  # Final relevance threshold
                    unique_variations.append(variation)
                    seen.add(variation)
        
        # Absolute maximum of 25 variations per keyword
        final_variations = unique_variations[:25]
        
        return {
            'morphological': morphological,
            'semantic': semantic,
            'commercial': commercial,
            'locational': locational,
            'question': question,
            'all_variations': final_variations
        }
    
    async def _generate_morphological_variations(self, keyword: str, options: Dict[str, Any] = None) -> List[str]:
        """Generate morphological variations based on Hebrew grammar."""
        variations = []
        analysis = await hebrew_loader.analyze_text(keyword)
        tokens = analysis.get('tokens', [])
        if tokens and len(tokens) > 0 and isinstance(tokens[0], dict):
            token_info = tokens[0]
        elif tokens and len(tokens) > 0 and isinstance(tokens[0], str):
            token_info = {
                'text': tokens[0],
                'lemma': tokens[0].lower(),
                'pos': 'UNKNOWN',
                'is_hebrew': any('\u0590' <= char <= '\u05FF' for char in tokens[0])
            }
        else:
            token_info = {
                'text': keyword,
                'lemma': keyword.lower(),
                'pos': 'UNKNOWN',
                'is_hebrew': any('\u0590' <= char <= '\u05FF' for char in keyword)
            }
        
        pos = token_info.get('pos', 'UNKNOWN')
        lemma = token_info.get('lemma', keyword)
        
        if pos == 'NOUN':
            variations.extend(self._generate_noun_variations(lemma))
        elif pos == 'VERB':
            variations.extend(self._generate_verb_variations(lemma))
        elif pos == 'ADJ':
            variations.extend(self._generate_adjective_variations(lemma))
        
        if not keyword.startswith('ה'):
            variations.append('ה' + keyword)
        
        for prep in ['ב', 'ל', 'של', 'עם', 'על']:
            variations.append(f"{prep} {keyword}")
            variations.append(f"{keyword} {prep}")
        
        return list(set(variations))
    
    def _generate_noun_variations(self, noun: str) -> List[str]:
        """Generate Hebrew noun variations."""
        variations = []
        
        if not noun.endswith('ים') and not noun.endswith('ות'):
            variations.extend([
                noun + 'ים',  # Masculine plural
                noun + 'ות',  # Feminine plural
            ])
        
        if noun.endswith('ה'):
            variations.append(noun[:-1] + 'ת')  # Feminine construct
        else:
            variations.append(noun + 'י')  # Masculine construct
        
        for suffix in ['ו', 'ה', 'ם', 'ן', 'נו', 'כם']:
            variations.append(noun + suffix)
        
        if len(noun) >= 3:
            variations.append(noun + 'ון')  # Diminutive
            variations.append(noun + 'ית')  # Feminine diminutive
        
        return variations
    
    def _generate_verb_variations(self, verb: str) -> List[str]:
        """Generate Hebrew verb variations."""
        variations = []
        
        root = self._extract_verb_root(verb)
        if not root:
            return variations
        
        patterns = [
            f"{root[0]}{root[1]}{root[2]}",
            f"{root[0]}ו{root[1]}{root[2]}",
            f"{root[0]}{root[1]}{root[2]}ת",
            f"{root[0]}{root[1]}{root[2]}ים",
            f"{root[0]}{root[1]}{root[2]}ות",
            
            f"{root[0]}{root[1]}{root[2]}תי",
            f"{root[0]}{root[1]}{root[2]}ת",
            f"{root[0]}{root[1]}{root[2]}ה",
            f"{root[0]}{root[1]}{root[2]}נו",
            f"{root[0]}{root[1]}{root[2]}ו",
            
            f"א{root[0]}{root[1]}{root[2]}",
            f"ת{root[0]}{root[1]}{root[2]}",
            f"י{root[0]}{root[1]}{root[2]}",
            f"נ{root[0]}{root[1]}{root[2]}",
            
            f"ל{root[0]}{root[1]}{root[2]}",
            
            f"מ{root[0]}{root[1]}{root[2]}",
            f"מ{root[0]}{root[1]}{root[2]}ת",
            f"מ{root[0]}{root[1]}{root[2]}ים",
            f"מ{root[0]}{root[1]}{root[2]}ות",
        ]
        
        variations.extend(patterns)
        return variations
    
    def _generate_adjective_variations(self, adjective: str) -> List[str]:
        """Generate Hebrew adjective variations."""
        variations = []
        
        if adjective.endswith('ה'):
            base = adjective[:-1]
            variations.extend([
                base,  # Masculine
                base + 'ים',  # Masculine plural
                adjective + 'ות'  # Feminine plural (rare but possible)
            ])
        else:
            variations.extend([
                adjective + 'ה',  # Feminine
                adjective + 'ים',  # Masculine plural
                adjective + 'ות'  # Feminine plural
            ])
        
        variations.extend([
            f"יותר {adjective}",  # More [adjective]
            f"הכי {adjective}",   # Most [adjective]
            f"{adjective} ביותר"  # [adjective] most
        ])
        
        return variations
    
    async def _generate_semantic_variations(self, keyword: str, options: Dict[str, Any] = None) -> List[str]:
        """Generate semantic variations with improved relevance filtering."""
        variations = []
        
        for word, synonyms in self.semantic_relations['synonyms'].items():
            if word == keyword:  # Exact match only
                variations.extend(synonyms[:3])  # Limit to top 3 synonyms
                break
        
        for word, related in self.semantic_relations['related_terms'].items():
            if word == keyword:  # Exact match only, not substring
                variations.extend(related[:2])  # Limit to top 2 related terms
                for term in related[:2]:
                    variations.append(f"{keyword} {term}")
                break
        
        return list(set(variations))[:self.max_variations_per_type]
    
    async def _generate_commercial_variations(self, keyword: str, options: Dict[str, Any] = None) -> List[str]:
        """Generate context-aware commercial variations with strict limits."""
        if not options.get('include_commercial', True):
            return []
        
        keyword_type = self._classify_keyword_type(keyword)
        
        if keyword_type == 'product':
            commercial_terms = ['מחיר', 'קנייה', 'מבצע', 'הנחה', 'זול']
        elif keyword_type == 'service':
            commercial_terms = ['עלות', 'שירות', 'מקצועי', 'איכותי', 'מומלץ']
        else:
            commercial_terms = ['מחיר', 'עלות', 'זול']  # Only 3 for general terms
        
        variations = []
        for term in commercial_terms:
            variations.extend([
                f"{keyword} {term}",
                f"{term} {keyword}",
                f"{term} ל{keyword}"
            ])
        
        filtered_variations = []
        for variation in variations:
            score = self._calculate_relevance_score(keyword, variation)
            if score > 0.4:  # Higher threshold for commercial
                filtered_variations.append(variation)
        
        return filtered_variations[:12]
    
    async def _generate_locational_variations(self, keyword: str, options: Dict[str, Any] = None) -> List[str]:
        """Generate selective locational variations only for relevant keywords."""
        if not options.get('include_locational', False):  # Default OFF
            return []
        
        if not self._is_location_relevant(keyword):
            return []
        
        major_cities = ['תל אביב', 'ירושלים', 'חיפה', 'באר שבע']
        
        variations = []
        for city in major_cities:
            variations.extend([
                f"{keyword} {city}",
                f"{keyword} ב{city}"
            ])
        
        filtered_variations = []
        for variation in variations:
            score = self._calculate_relevance_score(keyword, variation)
            if score > 0.3:
                filtered_variations.append(variation)
        
        return filtered_variations[:8]  # Maximum 8 locational variations
    
    def _is_location_relevant(self, keyword: str) -> bool:
        """Check if keyword is relevant for location-based variations."""
        location_indicators = [
            'שירות', 'חנות', 'מסעדה', 'בית קפה', 'רופא', 'עורך דין', 'מוסך',
            'ספר', 'מלון', 'צימר', 'אירוע', 'חתונה', 'קורס', 'לימודים',
            'עבודה', 'משרד', 'קליניקה', 'מרפאה', 'בנק', 'ביטוח'
        ]
        
        keyword_lower = keyword.lower()
        return any(indicator in keyword_lower for indicator in location_indicators)
    
    async def _generate_question_variations(self, keyword: str, options: Dict[str, Any] = None) -> List[str]:
        """Generate only essential question variations."""
        if not options.get('include_questions', True):
            return []
        
        # Only the 3 most important questions
        essential_questions = [
            f"מה זה {keyword}",
            f"איך {keyword}",
            f"כמה עולה {keyword}"
        ]
        
        return essential_questions
    
    async def _generate_combined_variations(self, expanded_results: Dict[str, Dict], options: Dict[str, Any] = None) -> List[str]:
        """Generate highly selective combined variations."""
        if len(expanded_results) < 2:
            return []
        
        options = options or {}
        max_combinations = options.get('max_combinations', 8)  # Reduced from unlimited
        
        keywords = list(expanded_results.keys())
        combined_variations = []
        
        # Only combine original keywords, not their variations
        for i, keyword1 in enumerate(keywords):
            for keyword2 in keywords[i+1:]:
                # Simple combinations only
                combinations = [
                    f"{keyword1} {keyword2}",
                    f"{keyword2} {keyword1}",
                    f"{keyword1} ו{keyword2}",
                    f"{keyword1} עם {keyword2}"
                ]
                
                # Filter by relevance
                for combo in combinations:
                    if len(combo.split()) <= 4:  # Limit length
                        combined_variations.append(combo)
                
                if len(combined_variations) >= max_combinations:
                    break
            
            if len(combined_variations) >= max_combinations:
                break
        
        return combined_variations[:max_combinations]
    
    def _extract_verb_root(self, verb: str) -> Optional[str]:
        """Extract Hebrew verb root (simplified approach)."""
        if len(verb) < 3:
            return None
            
        cleaned = verb
        for prefix in self.common_prefixes:
            if cleaned.startswith(prefix) and len(cleaned) > len(prefix) + 2:
                cleaned = cleaned[len(prefix):]
                break
        
        for suffix in self.common_suffixes:
            if cleaned.endswith(suffix) and len(cleaned) > len(suffix) + 2:
                cleaned = cleaned[:-len(suffix)]
                break
        
        if len(cleaned) >= 3:
            return cleaned[:3]
        
        return None
    
    def _calculate_expansion_score(self, original: str, variations: Set[str]) -> float:
        """Calculate expansion quality score."""
        if not variations:
            return 0.0
            
        variation_count = len(variations)
        diversity_score = min(variation_count / 20, 1.0)  # Normalize to 0-1
        
        quality_indicators = 0
        for variation in variations:
            if original in variation or variation in original:
                quality_indicators += 0.1  # Related variations
            if any(prep in variation for prep in ['ב', 'ל', 'של', 'עם']):
                quality_indicators += 0.05  # Grammatical variations
            if len(variation.split()) > 1:
                quality_indicators += 0.1  # Multi-word variations
        
        quality_score = min(quality_indicators, 1.0)
        
        return round((diversity_score + quality_score) / 2, 3)
    
    def _calculate_relevance_score(self, original: str, variation: str) -> float:
        """Calculate relevance score with much stricter filtering."""
        if not variation or not original:
            return 0.0
        
        original_words = set(original.split())
        variation_words = set(variation.split())
        
        intersection = len(original_words.intersection(variation_words))
        union = len(original_words.union(variation_words))
        jaccard = intersection / union if union > 0 else 0.0
        
        length_penalty = max(0, 1 - (len(variation.split()) - len(original.split())) / 5)
        
        connectors = ['ב', 'ל', 'של', 'עם', 'על', 'את', 'ה', 'ו']
        connector_count = sum(1 for word in variation.split() if word in connectors)
        connector_penalty = max(0, 1 - connector_count / 3)
        
        score = jaccard * length_penalty * connector_penalty
        
        return score if score > 0.4 else 0.0


# Global keyword expander instance
keyword_expander = HebrewKeywordExpander()
