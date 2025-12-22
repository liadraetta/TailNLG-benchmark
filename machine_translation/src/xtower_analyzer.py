"""
xTower error analyzer for translation post-editing.
"""
import logging
import re
from typing import Dict, Optional
from transformers import pipeline
from .base_xtower_analyzer import BaseXTowerAnalyzer


class XTowerAnalyzer(BaseXTowerAnalyzer):
    """Handles xTower model loading and error analysis."""
    
    def __init__(self, model_name: str = "sardinelab/xTower13B"):
        """
        Initialize xTower analyzer.
        
        Args:
            model_name: Name of the xTower model to use
        """
        super().__init__(model_name)
        self.pipeline = None
        self.logger = logging.getLogger(__name__)
        
    def load_model(self) -> bool:
        """
        Load the xTower model.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Loading xTower model: {self.model_name}...")
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device_map="auto"
            )
            self.logger.info("xTower model loaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load xTower model: {e}")
            return False
    
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
        self.logger.info("Starting xTower analysis...")

        if self.pipeline is None:
            self.logger.error("xTower model not loaded. Call load_model() first.")
            return None
            
        if not source_text or not translation_text:
            self.logger.warning("Empty source or translation text")
            return None
        
        # Extract quality info from XCOMET data
        quality_score = "moderate"
        quality_analysis = translation_text
        
        if xcomet_data:
            quality_score = xcomet_data.get('translation_quality_score', 'moderate')
            quality_analysis = xcomet_data.get('translation_quality_analysis', translation_text)
        
        # Build prompt
        prompt = self._build_prompt(
            source_text, translation_text, quality_analysis,
            quality_score, source_lang, target_lang
        )

        self.logger.debug(f"xTower prompt: {prompt}")
        
        try:
            # Format prompt with chat template
            messages = [{"role": "user", "content": prompt}]
            prompt_formatted = self.pipeline.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Generate response
            outputs = self.pipeline(prompt_formatted, max_new_tokens=1024, do_sample=False)
            full_output = outputs[0]['generated_text']
            
            # Extract assistant response
            assistant_marker = "<|im_start|>assistant"
            if assistant_marker in full_output:
                response = full_output.split(assistant_marker)[-1].strip()
            else:
                self.logger.warning("Assistant marker not found in xTower response")
                self.logger.warning(f"xTower response: {full_output}")
                response = full_output
            
            # Parse response
            result = self._parse_response(response, translation_text)

            self.logger.info("xTower analysis completed successfully")
            self.logger.debug(f"xTower result: {result}")

            
            # Add prompts and full output to result
            result["xtower_prompt"] = prompt
            result["xtower_prompt_formatted"] = prompt_formatted
            result["xtower_full_output"] = full_output
            
            return result
            
        except Exception as e:
            self.logger.error(f"xTower analysis failed: {e}")
            return {
                "has_errors": None,
                "errors": [],
                "corrected_translation": None,
                "xtower_prompt": prompt,
                "xtower_prompt_formatted": None,
                "xtower_full_output": None,
                "error_message": str(e)
            }
    
    @staticmethod
    def _build_prompt(
        source_text: str,
        translation_text: str,
        quality_analysis: str,
        quality_score: str,
        source_lang: str,
        target_lang: str
    ) -> str:
        """Build xTower prompt."""
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
    
    def cleanup(self):
        """Clean up model and free memory."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            
        self.logger.info("xTower model cleaned up")

