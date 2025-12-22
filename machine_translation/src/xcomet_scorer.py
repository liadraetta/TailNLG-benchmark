"""
XCOMET quality scorer for translation evaluation.
"""
import logging
import torch
from typing import Dict, Optional, List
from comet import download_model, load_from_checkpoint
from .base_xcomet_scorer import BaseXCOMETScorer


class XCOMETScorer(BaseXCOMETScorer):
    """Handles XCOMET model loading and scoring."""
    
    def __init__(self, model_name: str = "Unbabel/XCOMET-XL", use_gpu: bool = True):
        """
        Initialize XCOMET scorer.
        
        Args:
            model_name: Name of the XCOMET model to use
            use_gpu: Whether to use GPU for inference
        """
        super().__init__(model_name, use_gpu)
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.model = None
        self.logger = logging.getLogger(__name__)
        
    def load_model(self) -> bool:
        """
        Load the XCOMET model.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Loading XCOMET model: {self.model_name}...")
            model_path = download_model(self.model_name)
            self.model = load_from_checkpoint(model_path)
            
            if self.use_gpu:
                self.logger.info("Moving XCOMET model to GPU...")
                self.model = self.model.cuda()
                self.model.half()  # Use FP16 for memory efficiency
                
            self.model.eval()
            self.logger.info("XCOMET model loaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load XCOMET model: {e}")
            return False
    
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
        if self.model is None:
            self.logger.error("XCOMET model not loaded. Call load_model() first.")
            return None
            
        if not source_text or not translation_text:
            self.logger.warning("Empty source or translation text")
            return None
        
        try:
            data = [{
                "src": source_text,
                "mt": translation_text,
                "ref": reference_text or ""
            }]
            
            # Get prediction
            gpus = 1 if self.use_gpu else 0
            model_output = self.model.predict(data, batch_size=1, gpus=gpus)
            
            score = model_output.scores[0] if model_output.scores else None
            system_score = model_output.system_score if hasattr(model_output, 'system_score') else None
            error_spans = model_output.metadata.error_spans[0] if model_output.metadata.error_spans and model_output.metadata.error_spans[0] else []
            
            # Prepare prompt for logging
            ref_info = f"Reference: {reference_text}" if reference_text else "Reference: (none - reference-free mode)"
            xcomet_prompt = f"Source ({source_lang}): {source_text}\nTranslation ({target_lang}): {translation_text}\n{ref_info}"
            
            return {
                "score": float(score) if score is not None else None,
                "system_score": float(system_score) if system_score is not None else None,
                "error_spans": error_spans,
                "translation_quality_score": self._interpret_score(score),
                "translation_quality_analysis": self._format_quality_analysis(translation_text, error_spans),
                "xcomet_prompt": xcomet_prompt
            }
        except Exception as e:
            self.logger.error(f"XCOMET scoring failed: {e}")
            return {
                "score": None,
                "system_score": None,
                "error_spans": [],
                "translation_quality_score": None,
                "translation_quality_analysis": None,
                "xcomet_prompt": None,
                "error_message": str(e)
            }
    
    def cleanup(self):
        """Clean up model and free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
            
        if self.use_gpu:
            torch.cuda.empty_cache()
            
        self.logger.info("XCOMET model cleaned up")

