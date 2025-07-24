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
    
    async def _expand_single_keyword(self, keyword: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Expand a single Hebrew keyword with all variation types."""
        
        # Get morphological analysis
        analysis = await hebrew_loader.analyze_text(keyword)
        
        # Get token info safely with fallback
        tokens = analysis.get('tokens', [])
        if tokens and len(tokens) > 0 and isinstance(tokens[0], dict):
            # Tokens are in correct Dict format
            token_info = tokens[0]
        elif tokens and len(tokens) > 0 and isinstance(tokens[0], str):
            # Tokens are in string format, create dict structure
            token_info = {
                'text': tokens[0],
                'lemma': tokens[0].lower(),
                'pos': 'UNKNOWN',
                'is_hebrew': any('\u0590' <= char <= '\u05FF' for char in tokens[0])
            }
        else:
            # No tokens available, create fallback
            token_info = {
                'text': keyword,
                'lemma': keyword.lower(),
                'pos': 'UNKNOWN',
                'is_hebrew': any('\u0590' <= char <= '\u05FF' for char in keyword)
            }
        
        # Generate different types of variations
        expansion_tasks = [
            self._generate_morphological_variations(keyword, token_info),
            self._generate_semantic_variations(keyword),
            self._generate_commercial_variations(keyword),
            self._generate_locational_variations(keyword),
            self._generate_question_variations(keyword)
        ]
        
        results = await asyncio.gather(*expansion_tasks)
        
        # Combine all variations
        all_variations = set([keyword])  # Include original
        variation_types = {}
        
        for i, (var_type, variations) in enumerate([
            ('morphological', results[0]),
            ('semantic', results[1]),
            ('commercial', results[2]),
            ('locational', results[3]),
            ('question', results[4])
        ]):
            variation_types[var_type] = variations
            all_variations.update(variations)
        
        return {
            'original': keyword,
            'morphological_info': {
                'lemma': token_info.get('lemma', keyword),
                'pos': token_info.get('pos', 'UNKNOWN'),
                'morph': token_info.get('morph', '')
            },
            'variations_by_type': variation_types,
            'all_variations': list(all_variations),
            'expansion_score': self._calculate_expansion_score(keyword, all_variations)
        }
    
    async def _generate_morphological_variations(self, keyword: str, token_info: Dict) -> List[str]:
        """Generate morphological variations based on Hebrew grammar."""
        variations = []
        pos = token_info.get('pos', 'UNKNOWN')
        lemma = token_info.get('lemma', keyword)
        
        if pos == 'NOUN':
            # Generate noun variations
            variations.extend(self._generate_noun_variations(lemma))
        elif pos == 'VERB':
            # Generate verb variations
            variations.extend(self._generate_verb_variations(lemma))
        elif pos == 'ADJ':
            # Generate adjective variations
            variations.extend(self._generate_adjective_variations(lemma))
        
        # Add definite article variations
        if not keyword.startswith('ה'):
            variations.append('ה' + keyword)
        
        # Add preposition combinations
        for prep in ['ב', 'ל', 'של', 'עם', 'על']:
            variations.append(f"{prep} {keyword}")
            variations.append(f"{keyword} {prep}")
        
        return list(set(variations))
    
    def _generate_noun_variations(self, noun: str) -> List[str]:
        """Generate Hebrew noun variations."""
        variations = []
        
        # Plural forms
        if not noun.endswith('ים') and not noun.endswith('ות'):
            variations.extend([
                noun + 'ים',  # Masculine plural
                noun + 'ות',  # Feminine plural
            ])
        
        # Construct state
        if noun.endswith('ה'):
            variations.append(noun[:-1] + 'ת')  # Feminine construct
        else:
            variations.append(noun + 'י')  # Masculine construct
        
        # Possessive forms
        for suffix in ['ו', 'ה', 'ם', 'ן', 'נו', 'כם']:
            variations.append(noun + suffix)
        
        # Diminutive forms
        if len(noun) >= 3:
            variations.append(noun + 'ון')  # Diminutive
            variations.append(noun + 'ית')  # Feminine diminutive
        
        return variations
    
    def _generate_verb_variations(self, verb: str) -> List[str]:
        """Generate Hebrew verb variations."""
        variations = []
        
        # Extract root (simplified)
        root = self._extract_verb_root(verb)
        if not root:
            return variations
        
        # Generate different tenses and forms
        patterns = [
            # Present tense patterns
            f"{root[0]}{root[1]}{root[2]}",
            f"{root[0]}ו{root[1]}{root[2]}",
            f"{root[0]}{root[1]}{root[2]}ת",
            f"{root[0]}{root[1]}{root[2]}ים",
            f"{root[0]}{root[1]}{root[2]}ות",
            
            # Past tense patterns
            f"{root[0]}{root[1]}{root[2]}תי",
            f"{root[0]}{root[1]}{root[2]}ת",
            f"{root[0]}{root[1]}{root[2]}ה",
            f"{root[0]}{root[1]}{root[2]}נו",
            f"{root[0]}{root[1]}{root[2]}ו",
            
            # Future tense patterns
            f"א{root[0]}{root[1]}{root[2]}",
            f"ת{root[0]}{root[1]}{root[2]}",
            f"י{root[0]}{root[1]}{root[2]}",
            f"נ{root[0]}{root[1]}{root[2]}",
            
            # Infinitive
            f"ל{root[0]}{root[1]}{root[2]}",
            
            # Participles
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
        
        # Gender and number variations
        if adjective.endswith('ה'):
            # Feminine adjective
            base = adjective[:-1]
            variations.extend([
                base,  # Masculine
                base + 'ים',  # Masculine plural
                adjective + 'ות'  # Feminine plural (rare but possible)
            ])
        else:
            # Masculine adjective
            variations.extend([
                adjective + 'ה',  # Feminine
                adjective + 'ים',  # Masculine plural
                adjective + 'ות'  # Feminine plural
            ])
        
        # Comparative and superlative (Hebrew doesn't have morphological forms,
        # but we can add common constructions)
        variations.extend([
            f"יותר {adjective}",  # More [adjective]
            f"הכי {adjective}",   # Most [adjective]
            f"{adjective} ביותר"  # [adjective] most
        ])
        
        return variations
    
    async def _generate_semantic_variations(self, keyword: str) -> List[str]:
        """Generate semantic variations with improved relevance filtering."""
        variations = []
        
        # Check synonyms with exact matching
        for word, synonyms in self.semantic_relations['synonyms'].items():
            if word == keyword:  # Exact match only
                variations.extend(synonyms[:3])  # Limit to top 3 synonyms
                break
        
        # Check related terms with stricter matching
        for word, related in self.semantic_relations['related_terms'].items():
            if word == keyword:  # Exact match only, not substring
                # Add only most relevant related terms
                variations.extend(related[:2])  # Limit to top 2 related terms
                # Add selective combinations
                for term in related[:2]:
                    variations.append(f"{keyword} {term}")
                break
        
        return list(set(variations))[:self.max_variations_per_type]
    
    async def _generate_commercial_variations(self, keyword: str) -> List[str]:
        """Generate commercial intent variations."""
        variations = []
        
        # Add commercial modifiers
        commercial_modifiers = [
            'מחיר', 'עלות', 'זול', 'יקר', 'הנחה', 'מבצע', 'קנייה', 'רכישה',
            'מכירה', 'חנות', 'אונליין', 'באינטרנט', 'משלוח', 'מהיר', 'איכותי',
            'מקצועי', 'מומחה', 'שירות', 'טוב ביותר', 'מומלץ', 'ביקורות'
        ]
        
        for modifier in commercial_modifiers:
            variations.extend([
                f"{keyword} {modifier}",
                f"{modifier} {keyword}",
                f"{modifier} ל{keyword}",
                f"{keyword} ב{modifier}"
            ])
        
        # Add question words for commercial intent
        question_words = ['איך', 'איפה', 'מה', 'כמה', 'מתי', 'למה']
        for q_word in question_words:
            variations.extend([
                f"{q_word} {keyword}",
                f"{q_word} לקנות {keyword}",
                f"{q_word} למצוא {keyword}"
            ])
        
        return variations
    
    async def _generate_locational_variations(self, keyword: str) -> List[str]:
        """Generate location-based variations."""
        variations = []
        
        # Israeli cities and regions
        locations = [
            'תל אביב', 'ירושלים', 'חיפה', 'באר שבע', 'נתניה', 'פתח תקווה',
            'אשדוד', 'ראשון לציון', 'אשקלון', 'רמת גן', 'בני ברק', 'הרצליה',
            'כפר סבא', 'רעננה', 'מודיעין', 'רחובות', 'גוש דן', 'השרון',
            'הגליל', 'הנגב', 'יהודה ושומרון', 'ישראל'
        ]
        
        for location in locations:
            variations.extend([
                f"{keyword} {location}",
                f"{keyword} ב{location}",
                f"{location} {keyword}",
                f"{keyword} באזור {location}",
                f"{keyword} קרוב ל{location}"
            ])
        
        # Add general location terms
        location_terms = ['קרוב', 'באזור', 'בסביבה', 'מקומי', 'אזורי']
        for term in location_terms:
            variations.append(f"{keyword} {term}")
        
        return variations
    
    async def _generate_question_variations(self, keyword: str) -> List[str]:
        """Generate question-based variations."""
        variations = []
        
        question_patterns = [
            f"מה זה {keyword}",
            f"איך {keyword}",
            f"למה {keyword}",
            f"מתי {keyword}",
            f"איפה {keyword}",
            f"כמה עולה {keyword}",
            f"איך לבחור {keyword}",
            f"מה הטוב ב{keyword}",
            f"איך עובד {keyword}",
            f"מה ההבדל ב{keyword}",
            f"למה צריך {keyword}",
            f"איך להשתמש ב{keyword}",
            f"מה היתרונות של {keyword}",
            f"איך למצוא {keyword} טוב"
        ]
        
        variations.extend(question_patterns)
        return variations
    
    def _generate_combined_variations(self, expanded_results: Dict[str, Any], options: Dict[str, Any]) -> List[str]:
        """Generate combined keyword variations from multiple base keywords."""
        combined = []
        
        keywords = list(expanded_results.keys())
        
        # Generate 2-word combinations
        for i in range(len(keywords)):
            for j in range(i + 1, len(keywords)):
                kw1, kw2 = keywords[i], keywords[j]
                combined.extend([
                    f"{kw1} {kw2}",
                    f"{kw2} {kw1}",
                    f"{kw1} ו{kw2}",
                    f"{kw1} עם {kw2}",
                    f"{kw1} ל{kw2}",
                    f"{kw2} ל{kw1}"
                ])
        
        # Add variations with common connecting words
        connectors = ['של', 'עם', 'ב', 'ל', 'על', 'תחת', 'ליד', 'בתוך']
        for i in range(len(keywords)):
            for j in range(i + 1, len(keywords)):
                for connector in connectors:
                    combined.append(f"{keywords[i]} {connector} {keywords[j]}")
        
        return list(set(combined))[:100]  # Limit results
    
    def _extract_verb_root(self, verb: str) -> Optional[str]:
        """Extract Hebrew verb root (simplified approach)."""
        if len(verb) < 3:
            return None
            
        # Remove common prefixes
        cleaned = verb
        for prefix in self.common_prefixes:
            if cleaned.startswith(prefix) and len(cleaned) > len(prefix) + 2:
                cleaned = cleaned[len(prefix):]
                break
        
        # Remove common suffixes
        for suffix in self.common_suffixes:
            if cleaned.endswith(suffix) and len(cleaned) > len(suffix) + 2:
                cleaned = cleaned[:-len(suffix)]
                break
        
        # Return first 3 characters as potential root
        if len(cleaned) >= 3:
            return cleaned[:3]
        
        return None
    
    def _calculate_expansion_score(self, original: str, variations: Set[str]) -> float:
        """Calculate expansion quality score."""
        if not variations:
            return 0.0
            
        # Factors: number of variations, diversity, relevance
        variation_count = len(variations)
        diversity_score = min(variation_count / 20, 1.0)  # Normalize to 0-1
        
        # Quality score based on variation types
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


# Global keyword expander instance
keyword_expander = HebrewKeywordExpander()
