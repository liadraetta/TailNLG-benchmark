"""
Base class for xTower analyzers.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional


class BaseXTowerAnalyzer(ABC):
    """Abstract base class for xTower analyzers."""
    
    def __init__(self, model_name: str = "sardinelab/xTower13B"):
        """
        Initialize xTower analyzer.
        
        Args:
            model_name: Name of the xTower model to use
        """
        self.model_name = model_name
    
    @abstractmethod
    def load_model(self) -> bool:
        """
        Load the xTower model.
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def analyze_translation(
        self,
        source_text: str,
        translation_text: str,
        source_lang: str = "it",
        target_lang: str = "en",
        xcomet_data: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Analyze translation errors and generate corrections.
        
        Args:
            source_text: Original text in source language
            translation_text: Translated text
            source_lang: Source language code
            target_lang: Target language code
            xcomet_data: Optional XCOMET score data
            
        Returns:
            Dictionary with error analysis and corrected translation
        """
        pass
    
    @abstractmethod
    def cleanup(self):
        """Clean up model and free resources."""
        pass
    
    @staticmethod
    def _build_prompt(
        source_text: str,
        translation_text: str,
        quality_analysis: str,
        quality_score: str,
        source_lang: str,
        target_lang: str
    ) -> str:
        """
        Build xTower prompt.
        
        Args:
            source_text: Source text
            translation_text: Translation text
            quality_analysis: Quality analysis from XCOMET
            quality_score: Quality score category
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Formatted prompt string
        """
        lang_names = {
            'en': 'English',
            'es': 'Spanish',
            'it': 'Italian'
        }
        source_lang_name = lang_names.get(source_lang, source_lang.capitalize())
        target_lang_name = lang_names.get(target_lang, target_lang.capitalize())
        
        prompt = f"""You are provided with a Source, Translation, Translation quality analysis, and Translation quality score (weak, moderate, good, excellent). The Translation quality analysis contain a translation with marked error spans with different levels of severity (minor, major or critical). Additionally, we may provide a **reference translation**. Given this information, generate an explanation for each error and a fully correct translation.
        {source_lang_name} source: {source_text}
        {target_lang_name} translation: {translation_text}
        Translation quality analysis: {quality_analysis}
        Translation quality score: {quality_score}"""
        
        return prompt
    
    @staticmethod
    def _parse_response(response: str, original_translation: str) -> Dict:
        """
        Parse xTower response to extract errors and corrected translation.
        
        Args:
            response: Raw model response
            original_translation: Original translation text
            
        Returns:
            Dictionary with has_errors, errors list, and corrected_translation
        """
        import re
        
        errors = []
        corrected_translation = None
        explanations_text = None
        
        # Extract corrected translation
        correction_match = re.search(r'Translation correction:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if correction_match:
            corrected_translation = correction_match.group(1).strip()
        
        # Extract all explanations as a single block (from first "Explanation for error" to "Translation correction")
        explanations_match = re.search(
            r'(Explanation for error.+?)(?=Translation correction:|$)', 
            response, 
            re.DOTALL | re.IGNORECASE
        )
        if explanations_match:
            explanations_text = explanations_match.group(1).strip()
        
        # Extract individual error explanations for the errors list
        error_pattern = r'Explanation for error(\d+):\s*(.+?)(?=\nExplanation for error\d+:|Translation correction:|$)'
        error_matches = re.finditer(error_pattern, response, re.DOTALL | re.IGNORECASE)
        
        for match in error_matches:
            error_num = match.group(1)
            explanation = match.group(2).strip()
            
            errors.append({
                'error_number': int(error_num),
                'explanation': explanation,
                'severity': 'major' if 'significant' in explanation.lower() or 'critical' in explanation.lower() else 'minor'
            })
        
        has_errors = len(errors) > 0
        
        return {
            'has_errors': has_errors,
            'errors': errors,
            'explanations': explanations_text,
            'corrected_translation': corrected_translation or original_translation
        }
