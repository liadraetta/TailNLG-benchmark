"""
DeepL translator wrapper.
"""
import logging
import pandas as pd
from typing import Optional
import deepl


class DeepLTranslator:
    """Handles DeepL translation API."""
    
    # Language code mappings
    TARGET_CODES = {
        'en': 'EN-US',
        'es': 'ES',
        'it': 'IT'
    }
    
    def __init__(self, auth_key: str):
        """
        Initialize DeepL translator.
        
        Args:
            auth_key: DeepL API authentication key
        """
        self.auth_key = auth_key
        self.translator = None
        self.logger = logging.getLogger(__name__)
        
    def initialize(self) -> bool:
        """
        Initialize the DeepL translator.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.translator = deepl.Translator(self.auth_key)
            self.logger.info("DeepL translator initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize DeepL translator: {e}")
            return False
    
    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> Optional[str]:
        """
        Translate text using DeepL API.
        
        Args:
            text: Text to translate
            source_lang: Source language code (e.g., 'it', 'en', 'es')
            target_lang: Target language code (e.g., 'it', 'en', 'es')
            
        Returns:
            Translated text or None if translation fails
        """
        if self.translator is None:
            self.logger.error("DeepL translator not initialized. Call initialize() first.")
            return None
            
        # Check for empty text
        if not text or pd.isna(text) or str(text).strip() == '':
            self.logger.warning("Empty text provided for translation")
            return None
        
        try:
            # Get target code with fallback
            target_code = self.TARGET_CODES.get(target_lang, target_lang.upper())
            
            # Perform translation
            result = self.translator.translate_text(
                str(text),
                source_lang=source_lang.upper(),
                target_lang=target_code
            )
            
            return result.text
            
        except Exception as e:
            self.logger.error(f"DeepL translation error ({source_lang}->{target_lang}): {e}")
            return None
    
    def get_usage(self) -> Optional[dict]:
        """
        Get current API usage statistics.
        
        Returns:
            Dictionary with usage information or None if error
        """
        if self.translator is None:
            self.logger.error("DeepL translator not initialized")
            return None
            
        try:
            usage = self.translator.get_usage()
            return {
                "character_count": usage.character.count,
                "character_limit": usage.character.limit,
                "percentage_used": (usage.character.count / usage.character.limit * 100) if usage.character.limit > 0 else 0
            }
        except Exception as e:
            self.logger.error(f"Failed to get DeepL usage: {e}")
            return None
