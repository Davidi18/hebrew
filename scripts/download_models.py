#!/usr/bin/env python3
"""
Model Download Script for Hebrew Content Intelligence Service.
Downloads and verifies HebSpacy models for deployment.
"""

import sys
import os
import asyncio
from pathlib import Path
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from models.hebrew_loader import hebrew_loader


async def download_and_verify_models():
    """Download and verify HebSpacy models."""
    
    logger.info("Starting HebSpacy model download and verification")
    
    try:
        # Ensure model cache directory exists
        cache_dir = Path(settings.hebspacy_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Model cache directory: {cache_dir}")
        
        # Download and load model
        logger.info(f"Downloading HebSpacy model: {settings.hebspacy_model}")
        model = await hebrew_loader.load_model()
        
        # Verify model functionality
        logger.info("Verifying model functionality...")
        test_texts = [
            "זהו טקסט בדיקה בעברית",
            "עיצוב מטבח מודרני עם פתרונות אחסון חכמים",
            "בית יפה עם חדרים גדולים ונוף מרהיב",
            "טכנולוגיה מתקדמת לעסקים קטנים ובינוניים"
        ]
        
        for i, text in enumerate(test_texts, 1):
            logger.info(f"Testing text {i}/{len(test_texts)}: {text[:30]}...")
            
            analysis = await hebrew_loader.analyze_text(text)
            
            # Verify analysis results
            if not analysis or not analysis.get('tokens'):
                raise ValueError(f"Analysis failed for text {i}")
            
            tokens = analysis['tokens']
            hebrew_tokens = [t for t in tokens if t['is_hebrew']]
            
            logger.info(f"  - Total tokens: {len(tokens)}")
            logger.info(f"  - Hebrew tokens: {len(hebrew_tokens)}")
            logger.info(f"  - Entities found: {len(analysis.get('entities', []))}")
            logger.info(f"  - Hebrew ratio: {analysis['language_stats']['hebrew_ratio']:.2f}")
        
        # Get model information
        model_info = await hebrew_loader.get_model_info()
        logger.info("Model information:")
        for key, value in model_info.items():
            logger.info(f"  - {key}: {value}")
        
        logger.success("✅ HebSpacy model download and verification completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model download/verification failed: {e}")
        return False


def main():
    """Main function."""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO"
    )
    
    logger.info("Hebrew Content Intelligence Service - Model Download")
    logger.info("=" * 60)
    
    # Run async download
    success = asyncio.run(download_and_verify_models())
    
    if success:
        logger.success("Model download completed successfully!")
        sys.exit(0)
    else:
        logger.error("Model download failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
