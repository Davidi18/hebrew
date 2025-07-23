"""
HebSpacy Model Loader
Handles loading, caching, and validation of Hebrew NLP models.
"""

import os
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger
import spacy
from spacy.language import Language

from config.settings import settings


class HebSpacyLoader:
    """
    Manages HebSpacy model loading with caching and async support.
    Optimized for Hebrew morphological analysis and NER.
    """
    
    def __init__(self):
        self._model: Optional[Language] = None
        self._model_name = "he_ner_news_trf"
        self._cache_dir = Path(settings.hebspacy_cache_dir)
        self._is_loading = False
        self._load_lock = asyncio.Lock()
        
        # Ensure cache directory exists
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        
    async def load_model(self, force_reload: bool = False) -> Language:
        """
        Load HebSpacy model asynchronously with caching.
        
        Args:
            force_reload: Force reload even if model is cached
            
        Returns:
            Loaded HebSpacy language model
        """
        async with self._load_lock:
            if self._model is not None and not force_reload:
                return self._model
                
            if self._is_loading:
                # Wait for ongoing load to complete
                while self._is_loading:
                    await asyncio.sleep(0.1)
                return self._model
                
            self._is_loading = True
            
            try:
                logger.info(f"Loading HebSpacy model: {self._model_name}")
                
                # Load model in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                self._model = await loop.run_in_executor(
                    None, self._load_model_sync
                )
                
                # Validate model capabilities
                await self._validate_model()
                
                logger.info("HebSpacy model loaded successfully")
                return self._model
                
            except Exception as e:
                logger.error(f"Failed to load HebSpacy model: {e}")
                raise
            finally:
                self._is_loading = False
    
    def _load_model_sync(self) -> Language:
        """Synchronous model loading for thread executor."""
        try:
            # Load HebSpacy model using spaCy (correct API for hebspacy 0.1.7)
            import spacy
            model = spacy.load("he_ner_news_trf")
            logger.debug(f"Loaded HebSpacy model: he_ner_news_trf")
            return model
            
        except OSError as e:
            # Model not found - HebSpacy 0.1.7 downloads automatically on first import
            logger.error(f"HebSpacy model 'he_ner_news_trf' not found: {e}")
            logger.info("Make sure hebspacy is properly installed with: pip install hebspacy")
            
            # Try fallback to basic Hebrew spaCy model
            try:
                import spacy
                model = spacy.blank("he")
                logger.warning("Using blank Hebrew model as fallback")
                return model
            except Exception as fallback_error:
                logger.error(f"Failed to create fallback model: {fallback_error}")
                raise RuntimeError(f"Could not load Hebrew model: {e}")
    
    async def _validate_model(self) -> None:
        """Validate that the model has required Hebrew capabilities."""
        if not self._model:
            raise ValueError("Model not loaded")
            
        # Test Hebrew text processing
        test_text = "זהו טקסט בדיקה בעברית עם ניתוח מורפולוגי"
        
        try:
            doc = self._model(test_text)
            
            # Check basic tokenization
            if len(doc) == 0:
                raise ValueError("Model failed basic tokenization")
                
            # Check morphological analysis capabilities
            has_morphology = any(token.morph for token in doc)
            if not has_morphology:
                logger.warning("Model may not support Hebrew morphological analysis")
                
            # Check NER capabilities
            has_ner = len(self._model.pipe_names) > 0 and 'ner' in self._model.pipe_names
            if not has_ner:
                logger.warning("Model may not support Named Entity Recognition")
                
            logger.debug("Model validation completed successfully")
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            raise
    
    async def get_model(self) -> Language:
        """Get the loaded model, loading it if necessary."""
        if self._model is None:
            await self.load_model()
        return self._model
    
    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self._model is not None
    
    async def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze Hebrew text and return structured results.
        
        Args:
            text: Hebrew text to analyze
            
        Returns:
            Dictionary with analysis results
        """
        model = await self.get_model()
        
        if len(text) > settings.max_content_length:
            text = text[:settings.max_content_length]
            logger.warning(f"Text truncated to {settings.max_content_length} characters")
        
        doc = model(text)
        
        return {
            "tokens": [
                {
                    "text": token.text,
                    "lemma": token.lemma_,
                    "pos": token.pos_,
                    "tag": token.tag_,
                    "morph": str(token.morph) if token.morph else None,
                    "is_hebrew": self._is_hebrew_token(token.text),
                    "is_stop": token.is_stop,
                    "is_alpha": token.is_alpha
                }
                for token in doc
            ],
            "entities": [
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": getattr(ent, 'confidence', None)
                }
                for ent in doc.ents
            ],
            "sentences": [
                {
                    "text": sent.text,
                    "start": sent.start_char,
                    "end": sent.end_char
                }
                for sent in doc.sents
            ],
            "language_stats": self._get_language_stats(doc)
        }
    
    def _is_hebrew_token(self, text: str) -> bool:
        """Check if token contains Hebrew characters."""
        hebrew_chars = set(range(0x0590, 0x05FF))  # Hebrew Unicode block
        return any(ord(char) in hebrew_chars for char in text)
    
    def _get_language_stats(self, doc) -> Dict[str, Any]:
        """Get language distribution statistics."""
        total_tokens = len(doc)
        hebrew_tokens = sum(1 for token in doc if self._is_hebrew_token(token.text))
        
        return {
            "total_tokens": total_tokens,
            "hebrew_tokens": hebrew_tokens,
            "hebrew_ratio": hebrew_tokens / total_tokens if total_tokens > 0 else 0,
            "has_mixed_content": hebrew_tokens > 0 and hebrew_tokens < total_tokens
        }
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self._model:
            return {"loaded": False}
            
        return {
            "loaded": True,
            "model_name": self._model_name,
            "lang": self._model.lang,
            "pipeline": self._model.pipe_names,
            "vocab_size": len(self._model.vocab),
            "has_vectors": self._model.vocab.vectors_length > 0,
            "vectors_length": self._model.vocab.vectors_length
        }


# Global model loader instance
hebspacy_loader = HebSpacyLoader()
