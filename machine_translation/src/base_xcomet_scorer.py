"""
Base class for XCOMET scorers.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional


class BaseXCOMETScorer(ABC):
    """Abstract base class for XCOMET scorers."""
    
    def __init__(self, model_name: str = "Unbabel/XCOMET-XL", use_gpu: bool = True):
        """
        Initialize XCOMET scorer.
        
        Args:
            model_name: Name of the XCOMET model to use
            use_gpu: Whether to use GPU for inference
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
    
    @abstractmethod
    def load_model(self) -> bool:
        """
        Load the XCOMET model.
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def score_translation(
        self,
        source_text: str,
        translation_text: str,
        source_lang: str = "it",
        target_lang: str = "en",
        reference_text: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Score a translation using XCOMET.
        
        Args:
            source_text: Original text in source language
            translation_text: Translated text
            source_lang: Source language code
            target_lang: Target language code
            reference_text: Optional reference translation
            
        Returns:
            Dictionary with score, system_score, error_spans, quality analysis
        """
        pass
    
    @abstractmethod
    def cleanup(self):
        """Clean up model and free resources."""
        pass
    
    @staticmethod
    def _interpret_score(score: Optional[float]) -> Optional[str]:
        """
        Interpret XCOMET score into quality categories.
        Scores based on https://aclanthology.org/2022.wmt-1.70.pdf (p.11)

        Args:
            score: XCOMET score (0-1)
            
        Returns:
            Quality category string
        """
        if score is None:
            return None
        elif score >= 0.80:
            return "excellent"
        elif score >= 0.60:
            return "good"
        elif score >= 0.40:
            return "moderate"
        else:
            return "weak"
    
    @staticmethod
    def _format_quality_analysis(translation_text: str, error_spans: list) -> str:
        """
        Format quality analysis with marked error spans.
        
        Args:
            translation_text: The translation text
            error_spans: List of error span dictionaries
            
        Returns:
            Formatted analysis string with XML-style error tags
        """
        if not error_spans:
            return translation_text
        
        # Sort error spans by start position (reverse order for insertion)
        sorted_spans = sorted(error_spans, key=lambda x: x['start'], reverse=True)
        
        result = translation_text
        for span in sorted_spans:
            severity = span.get('severity', 'unknown')
            start = span['start']
            end = span['end']
            error_text = span['text']
            
            # Insert XML-style tags
            result = result[:start] + f'<error severity="{severity}">{error_text}</error>' + result[end:]
        
        return result
