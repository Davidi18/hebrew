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

from config.settings import settings

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
            # Temporary bypass for PyTorch security restriction
            # This will be resolved when we upgrade to PyTorch 2.6+
            import os
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            
            # Load heBERT tokenizer and model
            logger.info(f"Loading heBERT model: {self._model_name}")
            
            # Try with trust_remote_code for better compatibility
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
                trust_remote_code=True
            )
            self._model = AutoModel.from_pretrained(
                self._model_name,
                trust_remote_code=True
            )
            logger.info("heBERT model loaded successfully")
            
            # Load Hebrew NER pipeline
            logger.info(f"Loading Hebrew NER pipeline: {self._ner_model_name}")
            self._ner_pipeline = pipeline(
                'ner', 
                model=self._ner_model_name, 
                tokenizer=self._ner_model_name,
                aggregation_strategy='simple',
                trust_remote_code=True
            )
            logger.info("Hebrew NER pipeline loaded successfully")
            
            # Test the models
            self._test_models()
            
        except Exception as e:
            logger.error(f"Failed to load Transformers Hebrew models: {e}")
            
            # Check if it's specifically the PyTorch security issue
            if "CVE-2025-32434" in str(e) or "torch.load" in str(e):
                logger.warning("PyTorch security restriction detected. Consider upgrading to PyTorch 2.6+")
                logger.info("Attempting alternative loading method...")
                
                try:
                    # Alternative approach: use safetensors if available
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        self._model_name,
                        use_safetensors=True
                    )
                    self._model = AutoModel.from_pretrained(
                        self._model_name,
                        use_safetensors=True
                    )
                    logger.info("heBERT loaded successfully with safetensors")
                    return
                except Exception as safetensors_error:
                    logger.warning(f"Safetensors loading also failed: {safetensors_error}")
            
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
            
            # Try to load the Hebrew model first
            try:
                self._fallback_nlp = spacy.load("he_core_news_sm")
                logger.info("Loaded spaCy Hebrew model: he_core_news_sm")
            except OSError:
                # Fallback to blank Hebrew model if the trained model isn't available
                logger.warning("Hebrew model not found, using blank Hebrew model")
                self._fallback_nlp = spacy.blank("he")
                self._fallback_nlp.add_pipe("sentencizer")
            
            # Create mock components for compatibility
            self._tokenizer = self._create_mock_tokenizer()
            self._model = self._fallback_nlp
            self._ner_pipeline = self._create_mock_ner_pipeline()
            
            logger.info("Fallback spaCy model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load fallback model: {e}")
            # Create minimal mock components as last resort
            self._fallback_nlp = None
            self._tokenizer = self._create_mock_tokenizer()
            self._model = None
            self._ner_pipeline = self._create_mock_ner_pipeline()
    
    def _create_mock_tokenizer(self):
        """Create a mock tokenizer using spaCy for fallback."""
        class MockTokenizer:
            def __init__(self, nlp):
                self.nlp = nlp
            
            def tokenize(self, text: str) -> List[str]:
                if self.nlp is None:
                    return text.split()
                try:
                    doc = self.nlp(text)
                    return [token.text for token in doc]
                except Exception:
                    return text.split()
            
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
        logger.info(f"Starting analyze_text for: {text[:50]}...")
        
        try:
            model_components = await self.get_model()
            logger.info("Model components retrieved successfully")
            
            # Tokenization
            logger.info("Starting tokenization...")
            raw_tokens = model_components['tokenizer'].tokenize(text)
            logger.info(f"Tokenization completed: {len(raw_tokens)} tokens")
            
            # Convert tokens to expected format for hebrew_analyzer
            tokens = []
            for token_text in raw_tokens:
                # Check if token is Hebrew
                is_hebrew = any('\u0590' <= char <= '\u05FF' for char in token_text)
                
                tokens.append({
                    'text': token_text,
                    'lemma': token_text.lower(),  # Simple lemmatization fallback
                    'pos': 'UNKNOWN',  # Default POS tag
                    'is_hebrew': is_hebrew,
                    'is_alpha': token_text.isalpha(),
                    'is_stop': False  # Could be enhanced later
                })
            
            logger.info(f"Token formatting completed: {len(tokens)} structured tokens")
            
            # NER
            logger.info("Starting NER...")
            entities = model_components['ner_pipeline'](text)
            logger.info(f"NER completed: {len(entities) if isinstance(entities, list) else 0} entities")
            
            # Calculate Hebrew ratio for language stats
            logger.info("Calculating Hebrew ratio...")
            hebrew_chars = sum(1 for char in text if '\u0590' <= char <= '\u05FF')
            total_chars = len([c for c in text if c.isalpha()])
            hebrew_ratio = hebrew_chars / max(total_chars, 1)
            logger.info(f"Hebrew ratio calculated: {hebrew_ratio}")
            
            # Basic analysis
            logger.info("Building analysis response...")
            analysis = {
                'text': text,
                'tokens': tokens,
                'entities': entities,
                'token_count': len(tokens),
                'entity_count': len(entities) if isinstance(entities, list) else 0,
                'model_used': model_components['model_name'],
                'capabilities': model_components['capabilities'],
                'language_stats': {
                    'hebrew_ratio': hebrew_ratio,
                    'total_chars': len(text),
                    'hebrew_chars': hebrew_chars
                }
            }
            
            logger.info("Analysis completed successfully")
            return analysis
            
        except Exception as e:
            logger.error(f"Text analysis failed with error: {e}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error args: {e.args}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {
                'text': text,
                'error': str(e),
                'tokens': text.split(),
                'entities': [],
                'model_used': 'fallback',
                'language_stats': {
                    'hebrew_ratio': 0.0,
                    'total_chars': len(text),
                    'hebrew_chars': 0
                }
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
