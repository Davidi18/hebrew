"""
HebSpacy Model Loader
Handles loading, caching, and validation of Hebrew NLP models.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import spacy
from spacy.lang.he import Hebrew
from spacy.language import Language

# Transformers imports for advanced Hebrew NLP
try:
    from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForTokenClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from app.core.config import settings

logger = logging.getLogger(__name__)

class HebrewTransformersLoader:
    """Lightweight Hebrew NLP loader using heBERT for efficient Hebrew language understanding."""
    
    def __init__(self):
        self._model_name = "avichr/heBERT"  # Primary Hebrew BERT model
        self._ner_model_name = "avichr/heBERT_NER"  # Hebrew NER model
        # Removed AlephBERT to reduce resource usage
        self._cache_dir = Path(settings.hebspacy_cache_dir)
        self._is_loading = False
        self._load_lock = asyncio.Lock()
        
        # Model components
        self._tokenizer = None
        self._model = None
        self._ner_pipeline = None
        self._fallback_nlp = None
        
    async def get_model(self) -> Dict[str, Any]:
        """Get the Hebrew Transformers model components."""
        async with self._load_lock:
            if not self._is_model_loaded():
                await self._load_model_async()
            return self._get_model_components()
    
    def _is_model_loaded(self) -> bool:
        """Check if all model components are loaded."""
        return (self._tokenizer is not None and 
                self._model is not None and 
                self._ner_pipeline is not None)
    
    def _get_model_components(self) -> Dict[str, Any]:
        """Return dictionary of loaded model components."""
        return {
            'tokenizer': self._tokenizer,
            'model': self._model,
            'ner_pipeline': self._ner_pipeline,
            'fallback_nlp': self._fallback_nlp,
            'model_name': self._model_name,
            'capabilities': ['tokenization', 'ner', 'embeddings', 'contextual_analysis']
        }
    
    async def _load_model_async(self) -> None:
        """Load Hebrew Transformers models asynchronously."""
        if self._is_loading:
            return
            
        self._is_loading = True
        try:
            # Run the synchronous loading in a thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_model_sync)
        finally:
            self._is_loading = False
    
    def _load_model_sync(self) -> None:
        """Load Hebrew Transformers models synchronously."""
        logger.info("Loading Hebrew Transformers models...")
        
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available, falling back to spaCy")
            self._load_fallback_model()
            return
        
        try:
            # Load heBERT tokenizer and model
            logger.info(f"Loading heBERT model: {self._model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModel.from_pretrained(self._model_name)
            logger.info("heBERT model loaded successfully")
            
            # Load Hebrew NER pipeline
            logger.info(f"Loading Hebrew NER pipeline: {self._ner_model_name}")
            self._ner_pipeline = pipeline(
                'ner', 
                model=self._ner_model_name, 
                tokenizer=self._ner_model_name,
                aggregation_strategy='simple'
            )
            logger.info("Hebrew NER pipeline loaded successfully")
            
            # Test the models
            self._test_models()
            
        except Exception as e:
            logger.error(f"Failed to load Transformers Hebrew models: {e}")
            logger.info("Falling back to spaCy Hebrew model")
            self._load_fallback_model()
    
    def _test_models(self) -> None:
        """Test the loaded Hebrew models."""
        try:
            # Test tokenization
            test_text = "שלום, אני דוד מתל אביב"
            tokens = self._tokenizer.tokenize(test_text)
            logger.info(f"Hebrew tokenization test successful: {tokens}")
            
            # Test NER
            ner_results = self._ner_pipeline(test_text)
            logger.info(f"Hebrew NER test successful: {ner_results}")
            
            # Test embeddings
            inputs = self._tokenizer(test_text, return_tensors="pt")
            with torch.no_grad():
                outputs = self._model(**inputs)
            logger.info(f"Hebrew embeddings test successful, shape: {outputs.last_hidden_state.shape}")
            
        except Exception as e:
            logger.warning(f"Model testing failed: {e}")
    
    def _load_fallback_model(self) -> None:
        """Load fallback spaCy model for basic Hebrew processing."""
        try:
            logger.info("Loading fallback spaCy Hebrew model")
            self._fallback_nlp = spacy.blank("he")
            self._fallback_nlp.add_pipe("sentencizer")
            
            # Create mock components for compatibility
            self._tokenizer = self._create_mock_tokenizer()
            self._model = self._fallback_nlp
            self._ner_pipeline = self._create_mock_ner_pipeline()
            
            # Test fallback
            doc = self._fallback_nlp("שלום עולם")
            tokens = [token.text for token in doc]
            logger.info(f"Fallback spaCy test successful: {tokens}")
            
        except Exception as e:
            logger.error(f"Failed to load fallback model: {e}")
            raise RuntimeError("Unable to load any Hebrew NLP model")
    
    def _create_mock_tokenizer(self):
        """Create a mock tokenizer using spaCy for fallback."""
        class MockTokenizer:
            def __init__(self, nlp):
                self.nlp = nlp
            
            def tokenize(self, text: str) -> List[str]:
                doc = self.nlp(text)
                return [token.text for token in doc]
            
            def __call__(self, text: str, return_tensors=None):
                tokens = self.tokenize(text)
                return {'input_ids': tokens}
        
        return MockTokenizer(self._fallback_nlp)
    
    def _create_mock_ner_pipeline(self):
        """Create a mock NER pipeline for fallback."""
        def mock_ner(text: str) -> List[Dict[str, Any]]:
            # Basic Hebrew name detection (very simple)
            words = text.split()
            entities = []
            for i, word in enumerate(words):
                if word and word[0].isupper() and len(word) > 2:
                    entities.append({
                        'entity': 'PERSON',
                        'score': 0.5,
                        'word': word,
                        'start': text.find(word),
                        'end': text.find(word) + len(word)
                    })
            return entities
        
        return mock_ner
    
    # Analysis methods for Hebrew text
    async def analyze_text(self, text: str) -> Dict[str, Any]:
        """Comprehensive Hebrew text analysis."""
        model_components = await self.get_model()
        
        try:
            # Tokenization
            tokens = model_components['tokenizer'].tokenize(text)
            
            # NER
            entities = model_components['ner_pipeline'](text)
            
            # Basic analysis
            analysis = {
                'text': text,
                'tokens': tokens,
                'entities': entities,
                'token_count': len(tokens),
                'entity_count': len(entities) if isinstance(entities, list) else 0,
                'model_used': model_components['model_name'],
                'capabilities': model_components['capabilities']
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            return {
                'text': text,
                'error': str(e),
                'tokens': text.split(),
                'entities': [],
                'model_used': 'fallback'
            }
    
    # Backward compatibility methods for existing code
    async def load_model(self, force_reload: bool = False) -> Dict[str, Any]:
        """Backward compatibility: Load model (same as get_model)."""
        if force_reload:
            # Reset model components to force reload
            self._tokenizer = None
            self._model = None
            self._ner_pipeline = None
            self._fallback_nlp = None
        
        return await self.get_model()
    
    def is_loaded(self) -> bool:
        """Backward compatibility: Check if model is loaded."""
        return self._is_model_loaded()
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Backward compatibility: Get model information."""
        try:
            model_components = await self.get_model()
            return {
                "loaded": True,
                "model_name": model_components.get('model_name', 'unknown'),
                "type": "Hebrew Transformers (heBERT)",
                "capabilities": model_components.get('capabilities', []),
                "lang": "he",
                "pipeline": ["tokenizer", "ner", "embeddings"],
                "vocab_size": "N/A (Transformers)",
                "has_vectors": True,
                "vectors_length": 768  # Standard BERT embedding size
            }
        except Exception as e:
            return {
                "loaded": False,
                "error": str(e),
                "model_name": "unknown",
                "type": "Hebrew Transformers (heBERT)"
            }

# Global instance
hebrew_loader = HebrewTransformersLoader()
