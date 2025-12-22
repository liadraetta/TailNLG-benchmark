"""
Mock XCOMET scorer for testing without GPU.
"""
import logging
import random
from typing import Dict, Optional, List
from .base_xcomet_scorer import BaseXCOMETScorer


class MockXCOMETScorer(BaseXCOMETScorer):
    """Mock XCOMET scorer that simulates model outputs for testing."""
    
    def __init__(self, model_name: str = "Unbabel/XCOMET-XL", use_gpu: bool = True):
        """
        Initialize mock XCOMET scorer.
        
        Args:
            model_name: Name of the XCOMET model (ignored in mock)
            use_gpu: Whether to use GPU (ignored in mock)
        """
        super().__init__(model_name, use_gpu)
        self.logger = logging.getLogger(__name__)
        
    def load_model(self) -> bool:
        """
        Simulate loading the XCOMET model.
        
        Returns:
            True (always succeeds in mock mode)
        """
        self.logger.info(f"[MOCK] Loading XCOMET model: {self.model_name}...")
        self.logger.info("[MOCK] XCOMET model loaded successfully (mock mode)")
        return True
    
    def score_translation(
        self,
        source_text: str,
        translation_text: str,
        source_lang: str = "it",
        target_lang: str = "en",
        reference_text: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Generate mock XCOMET scores and error spans.
        
        Args:
            source_text: Original text in source language
            translation_text: Translated text
            source_lang: Source language code
            target_lang: Target language code
            reference_text: Optional reference translation
            
        Returns:
            Dictionary with mock score, system_score, error_spans
        """
        if not source_text or not translation_text:
            self.logger.warning("[MOCK] Empty source or translation text")
            return None
        
        # Generate mock scores
        # System score example: 0.45751070976257324
        system_score = random.uniform(0.3, 0.9)
        score = random.uniform(0.5, 0.95)
        
        # Generate mock error spans
        # Example: [{'text': 'ist bei', 'confidence': 0.40954849123954773, 'severity': 'critical', 'start': 13, 'end': 21}, ...]
        error_spans = self._generate_mock_error_spans(translation_text)
        
        # Prepare prompt for logging
        ref_info = f"Reference: {reference_text}" if reference_text else "Reference: (none - reference-free mode)"
        xcomet_prompt = f"Source ({source_lang}): {source_text}\nTranslation ({target_lang}): {translation_text}\n{ref_info}"
        
        result = {
            "score": float(score),
            "system_score": float(system_score),
            "error_spans": error_spans,
            "translation_quality_score": self._interpret_score(score),
            "translation_quality_analysis": self._format_quality_analysis(translation_text, error_spans),
            "xcomet_prompt": xcomet_prompt
        }
        
        self.logger.info(f"[MOCK] Generated XCOMET score: {score:.4f}, system_score: {system_score:.4f}")
        
        return result
    
    def _generate_mock_error_spans(self, text: str) -> List[Dict]:
        """
        Generate mock error spans for a text.
        
        Args:
            text: The translation text
            
        Returns:
            List of error span dictionaries
        """
        if len(text) < 10:
            return []
        
        # Randomly decide number of errors (0-3)
        num_errors = random.randint(0, 3)
        
        if num_errors == 0:
            return []
        
        words = text.split()
        if len(words) < 2:
            return []
        
        error_spans = []
        severities = ['minor', 'major', 'critical']
        
        for _ in range(min(num_errors, len(words))):
            # Pick a random word or two
            word_idx = random.randint(0, len(words) - 1)
            error_text = words[word_idx]
            
            # Sometimes include adjacent word
            if random.random() > 0.5 and word_idx < len(words) - 1:
                error_text = f"{words[word_idx]} {words[word_idx + 1]}"
            
            # Find position in original text
            start = text.find(error_text)
            if start == -1:
                continue
                
            end = start + len(error_text)
            
            error_spans.append({
                'text': error_text,
                'confidence': random.uniform(0.2, 0.8),
                'severity': random.choice(severities),
                'start': start,
                'end': end
            })
        
        # Sort by start position
        error_spans.sort(key=lambda x: x['start'])
        
        return error_spans
    
    def cleanup(self):
        """Clean up (no-op for mock)."""
        self.logger.info("[MOCK] XCOMET model cleaned up")

