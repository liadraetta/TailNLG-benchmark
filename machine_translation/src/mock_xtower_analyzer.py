"""
Mock xTower analyzer for testing without GPU.
"""
import logging
import random
from typing import Dict, Optional
from .base_xtower_analyzer import BaseXTowerAnalyzer


class MockXTowerAnalyzer(BaseXTowerAnalyzer):
    """Mock xTower analyzer that simulates model outputs for testing."""
    
    # Sample error explanations templates
    ERROR_TEMPLATES = [
        "The term \"{error_text}\" is a mistranslation that changes the meaning of the sentence.",
        "The phrase \"{error_text}\" introduces confusion about the context and should be corrected.",
        "The word \"{error_text}\" is not accurate in this context and affects comprehension.",
        "The expression \"{error_text}\" contains a grammatical error that impacts clarity.",
        "The segment \"{error_text}\" uses incorrect terminology for this domain."
    ]
    
    # Sample correction templates
    CORRECTION_TEMPLATES = [
        "corrected version",
        "proper translation",
        "accurate rendering",
        "better phrasing",
        "improved wording"
    ]
    
    def __init__(self, model_name: str = "sardinelab/xTower13B"):
        """
        Initialize mock xTower analyzer.
        
        Args:
            model_name: Name of the xTower model (ignored in mock)
        """
        super().__init__(model_name)
        self.logger = logging.getLogger(__name__)
        
    def load_model(self) -> bool:
        """
        Simulate loading the xTower model.
        
        Returns:
            True (always succeeds in mock mode)
        """
        self.logger.info(f"[MOCK] Loading xTower model: {self.model_name}...")
        self.logger.info("[MOCK] xTower model loaded successfully (mock mode)")
        return True
    
    def analyze_translation(
        self,
        source_text: str,
        translation_text: str,
        source_lang: str = "it",
        target_lang: str = "en",
        xcomet_data: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Generate mock error analysis and corrections.
        
        Args:
            source_text: Original text in source language
            translation_text: Translated text
            source_lang: Source language code
            target_lang: Target language code
            xcomet_data: Optional XCOMET score data
            
        Returns:
            Dictionary with mock error analysis and corrected translation
        """
        self.logger.info("[MOCK] Starting xTower analysis...")

        if not source_text or not translation_text:
            self.logger.warning("[MOCK] Empty source or translation text")
            return None
        
        # Extract quality info from XCOMET data
        quality_score = "moderate"
        quality_analysis = translation_text
        error_spans = []
        
        if xcomet_data:
            quality_score = xcomet_data.get('translation_quality_score', 'moderate')
            quality_analysis = xcomet_data.get('translation_quality_analysis', translation_text)
            error_spans = xcomet_data.get('error_spans', [])
        
        # Build prompt
        prompt = self._build_prompt(
            source_text, translation_text, quality_analysis,
            quality_score, source_lang, target_lang
        )

        self.logger.debug(f"[MOCK] xTower prompt: {prompt}")
        
        # Generate mock response
        response = self._generate_mock_response(translation_text, error_spans)
        
        # Build full output in the same format as real xTower
        prompt_formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        full_output = prompt_formatted + response
        
        # Parse the mock response
        result = self._parse_response(response, translation_text)

        self.logger.info("[MOCK] xTower analysis completed successfully")
        self.logger.debug(f"[MOCK] xTower result: {result}")
        
        # Add prompts and full output to result
        result["xtower_prompt"] = prompt
        result["xtower_prompt_formatted"] = prompt_formatted
        result["xtower_full_output"] = full_output
        
        self.logger.info(f"[MOCK] Generated xTower analysis with {len(result['errors'])} errors")
        
        return result
    
    def _generate_mock_response(self, translation_text: str, error_spans: list) -> str:
        """
        Generate mock xTower response.
        
        Example format:
        Explanation for error1: The term "Lawinenschilder" translates to "avalanche signs", 
        which is a significant mistranslation of "avalanche beacons"...
        Corrected translation: Alle trugen Lawinensuchgeräte.
        """
        if not error_spans:
            return f"The translation is accurate and requires no corrections.\nCorrected translation: {translation_text}"
        
        response_parts = []
        
        # Generate explanations for each error
        for i, error in enumerate(error_spans, 1):
            error_text = error.get('text', 'unknown')
            severity = error.get('severity', 'minor')
            template = random.choice(self.ERROR_TEMPLATES)
            explanation = template.format(error_text=error_text)
            
            response_parts.append(f"Explanation for error{i}: {explanation}")
        
        # Generate corrected translation (mock: just replace first error with a correction)
        corrected = translation_text
        if error_spans:
            first_error = error_spans[0]
            error_text = first_error.get('text', '')
            if error_text and error_text in corrected:
                correction = random.choice(self.CORRECTION_TEMPLATES)
                corrected = corrected.replace(error_text, correction, 1)
        
        response_parts.append(f"Translation correction: {corrected}")
        
        return "\n".join(response_parts)
    
    def cleanup(self):
        """Clean up (no-op for mock)."""
        self.logger.info("[MOCK] xTower model cleaned up")

