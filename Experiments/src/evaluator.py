"""
VerbalizationEvaluator: Standalone evaluation class for verbalization quality assessment.

This module provides a clean interface for evaluating generated verbalizations using
standard metrics: BERTScore, BLEU, chrF, and METEOR.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from evaluate import load
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
import gc
import re
import os
import xml.etree.ElementTree as ET
from scipy.stats import pearsonr, spearmanr



class VerbalizationEvaluator:
    """
    Evaluator for verbalization quality using multiple metrics.
    
    This class is independent of LLMHandler and can be used standalone
    to evaluate any JSON files containing predictions and references.
    """
    
    # Available metrics
    AVAILABLE_METRICS = ['bertscore', 'bertscore_rescaled', 'bleu', 'chrf', 'meteor', 'rouge1', 'rouge2', 'rougeL']
    
    def __init__(self, metrics: Optional[List[str]] = None):
        """
        Initialize the evaluator with specified metrics.
        
        Args:
            metrics: List of metrics to use. If None, uses all available metrics.
        """
        if metrics is None:
            metrics = self.AVAILABLE_METRICS
        
        # Validate metrics
        invalid_metrics = set(metrics) - set(self.AVAILABLE_METRICS)
        if invalid_metrics:
            raise ValueError(f"Invalid metrics: {invalid_metrics}. Available: {self.AVAILABLE_METRICS}")
        
        self.metrics = metrics
        self.metric_objects = self._load_metrics()
    
    @staticmethod
    def load_longtail_metadata_mapping(xml_path: Optional[Path] = None) -> Dict[str, Dict[str, str]]:
        """
        Load metadata mapping from longTailWebNLG XML file.
        
        Args:
            xml_path: Path to longTailWebNLG-v1.0.xml file.
                     If None, uses default path: experiments/datasets/longTail_webnlg/longTailWebNLG-v1.0.xml
        
        Returns:
            Dictionary mapping eid to metadata dict with keys: category, type, subtype
            Example: {"Id1": {"category": "Astronaut", "type": "long_tail", "subtype": "long_tail_en"}, ...}
        """
        if xml_path is None:
            # Default path relative to this file
            current_file = Path(__file__).resolve()
            xml_path = current_file.parent.parent / "datasets" / "longTail_webnlg" / "longTailWebNLG-v1.0.xml"
        
        if not xml_path.exists():
            raise FileNotFoundError(f"LongTail XML file not found: {xml_path}")
        
        mapping = {}
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        for entry in root.iter('entry'):
            eid = entry.get('eid')
            category = entry.get('category')
            type_ = entry.get('type')
            subtype = entry.get('subtype')
            
            mapping[eid] = {
                'category': category,
                'type': type_,
                'subtype': subtype
            }
        
        print(f"Loaded metadata for {len(mapping)} entries from {xml_path.name}")
        return mapping

    
    def _load_metrics(self) -> Dict[str, Any]:
        """Load metric objects from HuggingFace evaluate library."""
        metric_objects = {}
        
        print(f"Loading metrics: {self.metrics}")
        
        if 'bertscore' in self.metrics or 'bertscore_rescaled' in self.metrics:
            # Load BERTScore once (will be used with different params)
            metric_objects['bertscore'] = load("bertscore")
        if 'bleu' in self.metrics:
            metric_objects['bleu'] = load("bleu")
        if 'chrf' in self.metrics:
            metric_objects['chrf'] = load("chrf")
        if 'meteor' in self.metrics:
            metric_objects['meteor'] = load("meteor")
        if 'rouge1' in self.metrics or 'rouge2' in self.metrics or 'rougeL' in self.metrics:
            # Load ROUGE once (will be used with different params)
            metric_objects['rouge'] = load("rouge")
        
        return metric_objects
    
    @staticmethod
    def load_json_file(file_path: Path) -> List[Dict]:
        """
        Load a JSON file containing verbalization results.
        
        Args:
            file_path: Path to the JSON file
        
        Returns:
            List of dictionaries with verbalization data
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    @staticmethod
    def group_by_language(data: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group data by language.
        
        Args:
            data: List of dictionaries with 'language' field
        
        Returns:
            Dictionary mapping language codes to lists of entries
        """
        grouped = {}
        for entry in data:
            lang = entry.get('language', 'en')
            if lang not in grouped:
                grouped[lang] = []
            grouped[lang].append(entry)
        
        return grouped
    
    def evaluate_file(
        self,
        file_path: Path,
        prediction_key: str = 'prediction',
        reference_key: str = 'actual',
        save_individual_scores: bool = True,
        skip_existing: bool = True
    ) -> Dict[str, Any]:

        print(f"\nEvaluating: {file_path.name}")
        
        # Load data
        data = self.load_json_file(file_path)
        
        # Group by language
        data_by_language = self.group_by_language(data)
        
        # Results container
        results = {}
        
        # Track if we need to save (only if new scores were computed)
        need_to_save = False
        
        for language, lang_data in data_by_language.items():
            print(f"  Language: {language} ({len(lang_data)} samples)")
            
            # Initialize language results
            results[language] = {}
            
            # Compute metrics - BATCH EVALUATION for efficiency
            for metric_name in self.metrics:
                score_key = f'{metric_name}_score'
                
                # Check if metric already exists in ALL entries
                if skip_existing and all(score_key in entry for entry in lang_data):
                    print(f"    {metric_name}: already computed, skipping...", end=' ')
                    
                    # Extract existing scores
                    all_scores = [entry[score_key] for entry in lang_data]
                    
                    # Store results with statistics
                    results[language][metric_name] = {
                        'num_samples': len(all_scores),
                        'average': float(np.mean(all_scores)),
                        'std': float(np.std(all_scores)),
                        'min': float(np.min(all_scores)),
                        'max': float(np.max(all_scores))
                    }
                    
                    print(f"{results[language][metric_name]['average']:.4f} "
                          f"(±{results[language][metric_name]['std']:.4f})")
                else:
                    # Need to compute the metric
                    print(f"    Computing {metric_name}...", end=' ')
                    
                    # Extract all predictions and references for batch processing
                    predictions = [entry[prediction_key] for entry in lang_data]
                    references = [entry[reference_key] for entry in lang_data]
                    
                    # Compute metric for ALL predictions at once (much faster!)
                    all_scores = self._compute_batch_metric(
                        metric_name,
                        predictions,
                        references,
                        language
                    )
                    
                    # Save scores back to entries
                    if save_individual_scores:
                        for i, entry in enumerate(lang_data):
                            entry[score_key] = float(all_scores[i])
                        need_to_save = True
                        
                        # SAVE IMMEDIATELY after computing each metric to avoid memory issues
                        self._save_scores_to_file(file_path, data)
                        print(f"✓ saved", end=' ')
                    
                    # Store results with statistics across ALL generations
                    results[language][metric_name] = {
                        'num_samples': len(all_scores),
                        'average': float(np.mean(all_scores)),
                        'std': float(np.std(all_scores)),
                        'min': float(np.min(all_scores)),
                        'max': float(np.max(all_scores))
                    }
                    
                    print(f"{results[language][metric_name]['average']:.4f} "
                          f"(±{results[language][metric_name]['std']:.4f})")
        
        # Final message about saving
        if need_to_save:
            print(f"  All scores saved to {file_path.name}")
        elif save_individual_scores and not need_to_save:
            print(f"  No new scores to save")
        
        return results
    
    def _save_scores_to_file(self, file_path: Path, data: List[Dict]):
        """
        Save evaluation scores back to the original JSON file.
        
        Args:
            file_path: Path to the JSON file
            data: Updated data with scores
        """
        # Save updated data
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _compute_batch_metric(
        self,
        metric_name: str,
        predictions: List[str],
        references: List,  # Can be List[str] or List[List[str]]
        language: str
    ) -> List[float]:

        if language == 'it-PE':
            language = 'it'

        if 'bertscore' in metric_name:
            metric = self.metric_objects.get('bertscore')
        elif 'rouge' in metric_name:
            metric = self.metric_objects.get('rouge')
        else:
            metric = self.metric_objects.get(metric_name)
        
        # Normalize references to always be List[List[str]] for consistency
        normalized_references = []
        for ref in references:
            if isinstance(ref, list):
                # Already a list of references
                normalized_references.append(ref)
            else:
                # Single reference, wrap in list
                normalized_references.append([ref])
        
        if metric_name == 'bertscore':
            # BERTScore WITHOUT rescaling - batch mode
            # For multiple references, compute score against each and take the maximum
            scores = []
            for pred, refs in tqdm(zip(predictions, normalized_references), 
                                  total=len(predictions), 
                                  desc=f"Computing {metric_name}",
                                  leave=False):
                if len(refs) == 1:
                    # Single reference
                    result = metric.compute(
                        predictions=[pred],
                        references=refs,
                        model_type='bert-base-multilingual-cased',
                        lang=language,
                        rescale_with_baseline=False
                    )
                    scores.append(float(result['f1'][0]))
                else:
                    # Multiple references - compute against each and take max
                    max_score = 0.0
                    for ref in refs:
                        result = metric.compute(
                            predictions=[pred],
                            references=[ref],
                            model_type='bert-base-multilingual-cased',
                            lang=language,
                            rescale_with_baseline=False
                        )
                        max_score = max(max_score, float(result['f1'][0]))
                    scores.append(max_score)
            return scores
        
        elif metric_name == 'bertscore_rescaled':
            # BERTScore WITH rescaling - batch mode
            # For multiple references, compute score against each and take the maximum
            scores = []
            for pred, refs in tqdm(zip(predictions, normalized_references), 
                                  total=len(predictions), 
                                  desc=f"Computing {metric_name}",
                                  leave=False):
                if len(refs) == 1:
                    # Single reference
                    result = metric.compute(
                        predictions=[pred],
                        references=refs,
                        model_type='bert-base-multilingual-cased',
                        lang=language,
                        rescale_with_baseline=True
                    )
                    scores.append(float(result['f1'][0]))
                else:
                    # Multiple references - compute against each and take max
                    max_score = 0.0
                    for ref in refs:
                        result = metric.compute(
                            predictions=[pred],
                            references=[ref],
                            model_type='bert-base-multilingual-cased',
                            lang=language,
                            rescale_with_baseline=True
                        )
                        max_score = max(max_score, float(result['f1'][0]))
                    scores.append(max_score)
            return scores
        
        elif metric_name == 'bleu':
            # BLEU naturally supports multiple references but we need to score individually
            scores = []
            for pred, refs in tqdm(zip(predictions, normalized_references), 
                                  total=len(predictions), 
                                  desc=f"Computing {metric_name}",
                                  leave=False):
                if len(refs) == 1:
                    # Single reference
                    result = metric.compute(
                        predictions=[pred],
                        references=refs  # BLEU expects List[List[str]] format
                    )
                    scores.append(float(result['bleu']))
                else:
                    # Multiple references - compute against each and take max
                    max_score = 0.0
                    for ref in refs:
                        result = metric.compute(
                            predictions=[pred],
                            references=[ref]
                        )
                        max_score = max(max_score, float(result['bleu']))
                    scores.append(max_score)         
            return scores
        
        elif metric_name == 'chrf':
            # chrF++ - compute individually
            # For multiple references, compute against each and take the maximum
            scores = []
            for pred, refs in tqdm(zip(predictions, normalized_references), 
                                  total=len(predictions), 
                                  desc=f"Computing {metric_name}",
                                  leave=False):
                if len(refs) == 1:
                    # Single reference
                    result = metric.compute(
                        predictions=[pred],
                        references=refs,
                        word_order=2
                    )
                    scores.append(result['score'])
                else:
                    # Multiple references - compute against each and take max
                    max_score = 0.0
                    for ref in refs:
                        result = metric.compute(
                            predictions=[pred],
                            references=[ref],
                            word_order=2
                        )
                        max_score = max(max_score, result['score'])
                    scores.append(max_score)
            return scores
        
        elif metric_name == 'meteor':
            # METEOR - compute for each pair individually
            # For multiple references, compute against each and take the maximum
            scores = []
            for pred, refs in tqdm(zip(predictions, normalized_references), 
                                  total=len(predictions), 
                                  desc=f"Computing {metric_name}",
                                  leave=False):
                if len(refs) == 1:
                    # Single reference
                    result = metric.compute(
                        predictions=[pred],
                        references=refs
                    )
                    scores.append(result['meteor'])
                else:
                    # Multiple references - compute against each and take max
                    max_score = 0.0
                    for ref in refs:
                        result = metric.compute(
                            predictions=[pred],
                            references=[ref]
                        )
                        max_score = max(max_score, result['meteor'])
                    scores.append(max_score)
            return scores
        
        elif metric_name in ['rouge1', 'rouge2', 'rougeL']:
            # ROUGE - supports multiple references naturally
            rouge_type = 'rouge1' if metric_name == 'rouge1' else 'rouge2' if metric_name == 'rouge2' else 'rougeL'
            
            scores = []
            for pred, refs in tqdm(zip(predictions, normalized_references), 
                                  total=len(predictions), 
                                  desc=f"Computing {metric_name}",
                                  leave=False):
                if len(refs) == 1:
                    # Single reference
                    result = metric.compute(
                        predictions=[pred],
                        references=refs,
                        use_stemmer=True
                    )
                    scores.append(result[rouge_type])
                else:
                    # Multiple references - compute against each and take max
                    max_score = 0.0
                    for ref in refs:
                        result = metric.compute(
                            predictions=[pred],
                            references=[ref],
                            use_stemmer=True
                        )
                        max_score = max(max_score, result[rouge_type])
                    scores.append(max_score)
            return scores
        else:
            raise ValueError(f"Unknown metric: {metric_name}")
    
    def evaluate_multiple_files(
        self,
        file_paths: List[Path],
        prediction_key: str = 'prediction',
        reference_key: str = 'actual',
        save_individual_scores: bool = True,
        skip_existing: bool = True
    ) -> Dict[str, Dict[str, Any]]:

        all_results = {}
        
        for file_path in file_paths:
            method_name = file_path.stem  # Extract method name from filename
            results = self.evaluate_file(
                file_path, 
                prediction_key, 
                reference_key, 
                save_individual_scores,
                skip_existing
            )
            all_results[method_name] = results
        
        return all_results
    
    @staticmethod
    def save_results(results: Dict[str, Any], output_path: Path):

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_path}")
    
    @staticmethod
    def compare_methods(
        results: Dict[str, Dict[str, Any]],
        metric: str = 'bertscore',
        language: str = 'en'
    ) -> Dict[str, float]:

        comparison = {}
        
        for method_name, method_results in results.items():
            if language in method_results and metric in method_results[language]:
                comparison[method_name] = method_results[language][metric]['average']
        
        # Sort by score (descending)
        comparison = dict(sorted(comparison.items(), key=lambda x: x[1], reverse=True))
        
        return comparison
    
    @staticmethod
    def get_best_method(
        results: Dict[str, Dict[str, Any]],
        metric: str = 'bertscore',
        language: str = 'en'
    ) -> tuple[str, float]:

        comparison = VerbalizationEvaluator.compare_methods(results, metric, language)
        
        if not comparison:
            return None, 0.0
        
        best_method = max(comparison.items(), key=lambda x: x[1])
        return best_method
    
    @staticmethod
    def compute_metric_correlations(
        file_path: Path,
        metric1: str,
        metric2: str,
        group_by: Optional[str] = None
    ) -> Dict[str, Any]:


        print(f"\nComputing correlation between {metric1} and {metric2}")
        print(f"File: {file_path.name}")
        if group_by:
            print(f"Grouping by: {group_by}")
        
        # Load data
        data = VerbalizationEvaluator.load_json_file(file_path)
        
        # Check if both metrics exist
        metric1_key = f'{metric1}_score'
        metric2_key = f'{metric2}_score'
        
        # Filter entries that have both metrics
        valid_entries = [
            entry for entry in data 
            if metric1_key in entry and metric2_key in entry
            and entry[metric1_key] is not None and entry[metric2_key] is not None
        ]
        
        if not valid_entries:
            raise ValueError(f"No entries found with both {metric1_key} and {metric2_key}")
        
        print(f"  Valid entries with both metrics: {len(valid_entries)}/{len(data)}")
        
        def calculate_correlation(scores1: List[float], scores2: List[float]) -> Dict[str, Any]:
            """Helper function to calculate correlations."""
            if len(scores1) < 3:
                return {
                    'n': len(scores1),
                    'pearson': {'r': None, 'p_value': None, 'error': 'Insufficient samples (need >= 3)'},
                    'spearman': {'rho': None, 'p_value': None, 'error': 'Insufficient samples (need >= 3)'}
                }
            
            try:
                # Remove any NaN or Inf values
                valid_pairs = [
                    (s1, s2) for s1, s2 in zip(scores1, scores2)
                    if not (np.isnan(s1) or np.isnan(s2) or np.isinf(s1) or np.isinf(s2))
                ]
                
                if len(valid_pairs) < 3:
                    return {
                        'n': len(valid_pairs),
                        'pearson': {'r': None, 'p_value': None, 'error': 'Insufficient valid samples after NaN/Inf removal'},
                        'spearman': {'rho': None, 'p_value': None, 'error': 'Insufficient valid samples after NaN/Inf removal'}
                    }
                
                clean_scores1, clean_scores2 = zip(*valid_pairs)
                
                # Pearson correlation
                pearson_r, pearson_p = pearsonr(clean_scores1, clean_scores2)
                
                # Spearman correlation
                spearman_rho, spearman_p = spearmanr(clean_scores1, clean_scores2)
                
                return {
                    'n': len(clean_scores1),
                    'pearson': {
                        'r': float(pearson_r),
                        'p_value': float(pearson_p),
                        'interpretation': 'strong' if abs(pearson_r) >= 0.7 else 'moderate' if abs(pearson_r) >= 0.4 else 'weak'
                    },
                    'spearman': {
                        'rho': float(spearman_rho),
                        'p_value': float(spearman_p),
                        'interpretation': 'strong' if abs(spearman_rho) >= 0.7 else 'moderate' if abs(spearman_rho) >= 0.4 else 'weak'
                    }
                }
            except Exception as e:
                return {
                    'n': len(scores1),
                    'pearson': {'r': None, 'p_value': None, 'error': str(e)},
                    'spearman': {'rho': None, 'p_value': None, 'error': str(e)}
                }
        
        if group_by is None:
            # Compute overall correlation
            scores1 = [entry[metric1_key] for entry in valid_entries]
            scores2 = [entry[metric2_key] for entry in valid_entries]
            
            result = calculate_correlation(scores1, scores2)
            
            if result['pearson']['r'] is not None:
                print(f"\n  Overall Correlation (n={result['n']}):")
                print(f"    Pearson r:  {result['pearson']['r']:>6.3f} (p={result['pearson']['p_value']:.4f}) [{result['pearson']['interpretation']}]")
                print(f"    Spearman ρ: {result['spearman']['rho']:>6.3f} (p={result['spearman']['p_value']:.4f}) [{result['spearman']['interpretation']}]")
            
            return result
        
        else:
            # Compute correlations by group
            groups = {}
            for entry in valid_entries:
                group_value = entry.get(group_by, 'unknown')
                if group_value not in groups:
                    groups[group_value] = []
                groups[group_value].append(entry)
            
            results = {}
            
            # Compute correlation for each group
            for group_value, group_entries in sorted(groups.items()):
                scores1 = [entry[metric1_key] for entry in group_entries]
                scores2 = [entry[metric2_key] for entry in group_entries]
                
                results[group_value] = calculate_correlation(scores1, scores2)
            
            # Compute overall correlation
            overall_scores1 = [entry[metric1_key] for entry in valid_entries]
            overall_scores2 = [entry[metric2_key] for entry in valid_entries]
            results['overall'] = calculate_correlation(overall_scores1, overall_scores2)
            
            # Print results
            print(f"\n  Correlations by {group_by}:")
            for group_value in sorted([k for k in results.keys() if k != 'overall']):
                result = results[group_value]
                if result['pearson']['r'] is not None:
                    print(f"\n    {group_value} (n={result['n']}):")
                    print(f"      Pearson r:  {result['pearson']['r']:>6.3f} (p={result['pearson']['p_value']:.4f}) [{result['pearson']['interpretation']}]")
                    print(f"      Spearman ρ: {result['spearman']['rho']:>6.3f} (p={result['spearman']['p_value']:.4f}) [{result['spearman']['interpretation']}]")
                else:
                    print(f"\n    {group_value} (n={result['n']}): Error - {result['pearson'].get('error', 'Unknown error')}")
            
            if results['overall']['pearson']['r'] is not None:
                print(f"\n    Overall (n={results['overall']['n']}):")
                print(f"      Pearson r:  {results['overall']['pearson']['r']:>6.3f} (p={results['overall']['pearson']['p_value']:.4f}) [{results['overall']['pearson']['interpretation']}]")
                print(f"      Spearman ρ: {results['overall']['spearman']['rho']:>6.3f} (p={results['overall']['spearman']['p_value']:.4f}) [{results['overall']['spearman']['interpretation']}]")
            
            return results
    

    # ===========================
    # Statistics
    # ===========================
    @staticmethod
    def compute_statistics(
        file_path: Path,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:

        print(f"\nComputing statistics for: {file_path.name}")
        
        # Load data
        data = VerbalizationEvaluator.load_json_file(file_path)
        
        # Auto-detect available metrics from first entry if not specified
        if metrics is None:
            metrics = []
            for key in data[0].keys():
                if key.endswith('_score'):
                    metrics.append(key.replace('_score', ''))
        
        print(f"  Metrics found: {metrics}")
        
        # Initialize containers
        overall_scores = {metric: [] for metric in metrics}
        by_language = {}
        
        # Collect scores
        for entry in data:
            lang = entry.get('language', 'unknown')
            
            # Initialize language if needed
            if lang not in by_language:
                by_language[lang] = {metric: [] for metric in metrics}
            
            # Collect scores for each metric
            for metric in metrics:
                score_key = f'{metric}_score'
                if score_key in entry and entry[score_key] is not None:
                    score = entry[score_key]
                    overall_scores[metric].append(score)
                    by_language[lang][metric].append(score)
        
        # Compute overall statistics
        result = {
            'overall': {
                'n': len(data)
            }
        }
        
        for metric in metrics:
            if overall_scores[metric]:
                result['overall'][metric] = {
                    'mean': float(np.mean(overall_scores[metric])),
                    'std': float(np.std(overall_scores[metric])),
                    'min': float(np.min(overall_scores[metric])),
                    'max': float(np.max(overall_scores[metric]))
                }
        
        # Compute by-language statistics
        result['by_language'] = {}
        
        for lang in sorted(by_language.keys()):
            result['by_language'][lang] = {
                'n': len([entry for entry in data if entry.get('language') == lang])
            }
            
            for metric in metrics:
                if by_language[lang][metric]:
                    result['by_language'][lang][metric] = {
                        'mean': float(np.mean(by_language[lang][metric])),
                        'std': float(np.std(by_language[lang][metric])),
                        'min': float(np.min(by_language[lang][metric])),
                        'max': float(np.max(by_language[lang][metric]))
                    }
        
        # Print summary
        print(f"\n  Overall statistics (n={result['overall']['n']}):")
        for metric in metrics:
            if metric in result['overall']:
                stats = result['overall'][metric]
                print(f"    {metric}: {stats['mean']:.4f} (±{stats['std']:.4f})")
        
        print(f"\n  By language:")
        for lang in sorted(result['by_language'].keys()):
            print(f"    {lang} (n={result['by_language'][lang]['n']}):")
            for metric in metrics:
                if metric in result['by_language'][lang]:
                    stats = result['by_language'][lang][metric]
                    print(f"      {metric}: {stats['mean']:.4f} (±{stats['std']:.4f})")
        
        return result
    
    @staticmethod
    def compute_statistics_longtail_only(
        file_path: Path,
        metrics: Optional[List[str]] = None,
        xml_path: Optional[Path] = None
    ) -> Dict[str, Any]:

        print(f"\nComputing statistics for LONG_TAIL only: {file_path.name}")
        
        # Load metadata mapping from XML
        metadata_mapping = VerbalizationEvaluator.load_longtail_metadata_mapping(xml_path)
        
        # Load data
        data = VerbalizationEvaluator.load_json_file(file_path)
        
        # Filter only long_tail entries
        filtered_data = []
        for entry in data:
            eid = entry.get('eid')
            if eid in metadata_mapping:
                if metadata_mapping[eid]['type'] == 'long_tail':
                    filtered_data.append(entry)
        
        print(f"  Filtered: {len(filtered_data)}/{len(data)} long_tail entries")
        
        if not filtered_data:
            print("  Warning: No long_tail entries found!")
            return {'overall': {}, 'by_language': {}}
        
        # Auto-detect available metrics from first entry if not specified
        if metrics is None:
            metrics = []
            for key in filtered_data[0].keys():
                if key.endswith('_score'):
                    metrics.append(key.replace('_score', ''))
        
        print(f"  Metrics found: {metrics}")
        
        # Initialize containers
        overall_scores = {metric: [] for metric in metrics}
        by_language = {}
        
        # Collect scores
        for entry in filtered_data:
            lang = entry.get('language', 'unknown')
            
            # Initialize language if needed
            if lang not in by_language:
                by_language[lang] = {metric: [] for metric in metrics}
            
            # Collect scores for each metric
            for metric in metrics:
                score_key = f'{metric}_score'
                if score_key in entry and entry[score_key] is not None:
                    score = entry[score_key]
                    overall_scores[metric].append(score)
                    by_language[lang][metric].append(score)
        
        # Compute overall statistics
        result = {
            'overall': {
                'n': len(filtered_data)
            }
        }
        
        for metric in metrics:
            if overall_scores[metric]:
                result['overall'][metric] = {
                    'mean': float(np.mean(overall_scores[metric])),
                    'std': float(np.std(overall_scores[metric])),
                    'min': float(np.min(overall_scores[metric])),
                    'max': float(np.max(overall_scores[metric]))
                }
        
        # Compute by-language statistics
        result['by_language'] = {}
        
        for lang in sorted(by_language.keys()):
            result['by_language'][lang] = {
                'n': len([entry for entry in filtered_data if entry.get('language') == lang])
            }
            
            for metric in metrics:
                if by_language[lang][metric]:
                    result['by_language'][lang][metric] = {
                        'mean': float(np.mean(by_language[lang][metric])),
                        'std': float(np.std(by_language[lang][metric])),
                        'min': float(np.min(by_language[lang][metric])),
                        'max': float(np.max(by_language[lang][metric]))
                    }
        
        # Print summary
        print(f"\n  Overall statistics (n={result['overall']['n']}):")
        for metric in metrics:
            if metric in result['overall']:
                stats = result['overall'][metric]
                print(f"    {metric}: {stats['mean']:.4f} (±{stats['std']:.4f})")
        
        print(f"\n  By language:")
        for lang in sorted(result['by_language'].keys()):
            print(f"    {lang} (n={result['by_language'][lang]['n']}):")
            for metric in metrics:
                if metric in result['by_language'][lang]:
                    stats = result['by_language'][lang][metric]
                    print(f"      {metric}: {stats['mean']:.4f} (±{stats['std']:.4f})")
        
        return result
    
    @staticmethod
    def format_aggregate_statistics_table(
        stats: Dict[str, Any],
        metrics: Optional[List[str]] = None,
        output_path: Optional[Path] = None
    ) -> str:

        lines = []
        lines.append("=" * 120)
        lines.append("AGGREGATE STATISTICS")
        lines.append("=" * 120)
        
        # Auto-detect metrics if not provided
        if metrics is None:
            if 'overall' in stats and stats['overall']:
                metrics = [k for k in stats['overall'].keys() if k != 'n']
            else:
                metrics = []
        
        # Overall statistics
        lines.append("\n" + "=" * 120)
        lines.append("OVERALL STATISTICS")
        lines.append("=" * 120)
        
        if 'overall' in stats and stats['overall']:
            n = stats['overall'].get('n', 0)
            lines.append(f"\nTotal samples: {n}")
            lines.append("")
            
            # Create table header
            header = f"{'Metric':20s} | {'Mean':>10s} | {'Std':>10s} | {'Min':>10s} | {'Max':>10s}"
            lines.append(header)
            lines.append("-" * 120)
            
            # Add rows
            for metric in metrics:
                if metric in stats['overall']:
                    s = stats['overall'][metric]
                    row = f"{metric:20s} | {s['mean']:10.4f} | {s['std']:10.4f} | {s['min']:10.4f} | {s['max']:10.4f}"
                    lines.append(row)
        else:
            lines.append("\nNo overall statistics available")
        
        # By language statistics
        lines.append("\n" + "=" * 120)
        lines.append("STATISTICS BY LANGUAGE")
        lines.append("=" * 120)
        
        if 'by_language' in stats and stats['by_language']:
            for lang in sorted(stats['by_language'].keys()):
                lang_stats = stats['by_language'][lang]
                n = lang_stats.get('n', 0)
                
                lines.append(f"\n{'-' * 120}")
                lines.append(f"Language: {lang.upper()} (n={n})")
                lines.append('-' * 120)
                
                # Create table header
                header = f"{'Metric':20s} | {'Mean':>10s} | {'Std':>10s} | {'Min':>10s} | {'Max':>10s}"
                lines.append(header)
                lines.append("-" * 120)
                
                # Add rows
                for metric in metrics:
                    if metric in lang_stats:
                        s = lang_stats[metric]
                        row = f"{metric:20s} | {s['mean']:10.4f} | {s['std']:10.4f} | {s['min']:10.4f} | {s['max']:10.4f}"
                        lines.append(row)
        else:
            lines.append("\nNo language-specific statistics available")
        
        lines.append("\n" + "=" * 120)
        
        result = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"\nAggregate statistics table saved to: {output_path}")
        
        return result
    
    @staticmethod
    def compute_statistics_by_quality(
        file_path: Path,
        metrics: Optional[List[str]] = None,
        xml_path: Optional[Path] = None,
        output_json_path: Optional[Path] = None,
        output_md_path: Optional[Path] = None
    ) -> Dict[str, Any]:

        print(f"\nComputing quality statistics for: {file_path.name}")
        
        # Load metadata mapping from XML (includes quality field)
        if xml_path is None:
            # Default path relative to this file
            current_file = Path(__file__).resolve()
            xml_path = current_file.parent.parent / "datasets" / "longTail_webnlg" / "longTailWebNLG-v1.0.xml"
        
        if not xml_path.exists():
            raise FileNotFoundError(f"LongTail XML file not found: {xml_path}")
        
        # Parse XML to extract quality information
        # Quality is stored in <lex> elements, not in <entry> attributes
        print(f"  Loading quality metadata from: {xml_path.name}")
        import xml.etree.ElementTree as ET
        
        quality_mapping = {}
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        for entry in root.iter('entry'):
            eid = entry.get('eid')
            category = entry.get('category')
            type_ = entry.get('type')
            subtype = entry.get('subtype')
            
            # Extract quality from <lex> elements (quality varies by language)
            # Create a mapping: (eid, language) -> quality
            for lex in entry.findall('lex'):
                quality = lex.get('quality', 'unknown')
                lang = lex.get('lang', 'unknown')
                
                # Create composite key: eid|lang
                key = f"{eid}|{lang}"
                
                quality_mapping[key] = {
                    'quality': quality,
                    'category': category,
                    'type': type_,
                    'subtype': subtype
                }
        
            print(f"  Loaded quality metadata for {len(quality_mapping)} eid-language combinations")
            type_ = entry.get('type')
            subtype = entry.get('subtype')

            if quality is None:
                raise ValueError(f"Entry with eid={eid} missing 'quality' attribute in XML")
            
            quality_mapping[eid] = {
                'quality': quality,
                'category': category,
                'type': type_,
                'subtype': subtype
            }
        
        print(f"  Loaded quality metadata for {len(quality_mapping)} entries")
        
        # Load data
        data = VerbalizationEvaluator.load_json_file(file_path)
        
        # Auto-detect available metrics from first entry if not specified
        if metrics is None:
            metrics = []
            for key in data[0].keys():
                if key.endswith('_score'):
                    metrics.append(key.replace('_score', ''))
        
        print(f"  Metrics found: {metrics}")
        
        # Initialize containers with lists to store individual scores
        stats = {
            'by_quality': {},
            'by_quality_language': {},
            'by_quality_type': {},
            'by_quality_subtype': {},
        }
        
        # Track entries without quality
        missing_quality_count = 0
        missing_eid_count = 0
        quality_distribution = {}
        
        # Group data by different criteria
        for entry in data:
            eid = entry.get('eid')
            lang = entry.get('language', 'unknown')
            
            # Get quality from XML mapping using eid AND language
            if eid is None:
                missing_eid_count += 1
                continue
            
            # Create composite key: eid|lang
            key = f"{eid}|{lang}"
            
            if key not in quality_mapping:
                missing_quality_count += 1
                continue
            
            # Get metadata from XML
            metadata = quality_mapping[key]
            quality = metadata['quality']
            category = metadata['category']
            type_ = metadata['type']
            subtype = metadata['subtype']
            
            # Track quality distribution
            quality_distribution[quality] = quality_distribution.get(quality, 0) + 1
            
            # By quality
            if quality not in stats['by_quality']:
                stats['by_quality'][quality] = {m: [] for m in metrics}
            
            # By quality-language
            quality_lang_key = f"{quality}|{lang}"
            if quality_lang_key not in stats['by_quality_language']:
                stats['by_quality_language'][quality_lang_key] = {m: [] for m in metrics}
            
            # By quality-type
            quality_type_key = f"{quality}|{type_}"
            if quality_type_key not in stats['by_quality_type']:
                stats['by_quality_type'][quality_type_key] = {m: [] for m in metrics}
            
            # By quality-subtype
            quality_subtype_key = f"{quality}|{subtype}"
            if quality_subtype_key not in stats['by_quality_subtype']:
                stats['by_quality_subtype'][quality_subtype_key] = {m: [] for m in metrics}
        
            
            # Collect scores
            for metric in metrics:
                score_key = f'{metric}_score'
                if score_key in entry:
                    score = entry[score_key]
                    stats['by_quality'][quality][metric].append(score)
                    stats['by_quality_language'][quality_lang_key][metric].append(score)
                    stats['by_quality_type'][quality_type_key][metric].append(score)
                    stats['by_quality_subtype'][quality_subtype_key][metric].append(score)
        
        if missing_eid_count > 0:
            print(f"  Warning: {missing_eid_count} entries skipped due to missing eid field")
        if missing_quality_count > 0:
            print(f"  Warning: {missing_quality_count} entries skipped due to eid not found in XML")
        
        print(f"  Quality distribution: {quality_distribution}")
        
        # Compute summary statistics (including individual scores for t-test)
        result = {
            'file': file_path.name,
            'quality_distribution': quality_distribution,
            'statistics': {}
        }
        
        for group_name in ['by_quality', 'by_quality_language', 'by_quality_type', 
                           'by_quality_subtype']:
            result['statistics'][group_name] = {}
            
            for group_key, metric_scores in stats[group_name].items():
                result['statistics'][group_name][group_key] = {}
                
                for metric, scores in metric_scores.items():
                    if scores:  # Only compute if we have scores
                        result['statistics'][group_name][group_key][metric] = {
                            'n': len(scores),
                            'mean': float(np.mean(scores)),
                            'std': float(np.std(scores)),
                            'min': float(np.min(scores)),
                            'max': float(np.max(scores)),
                            'scores': scores  # Keep individual scores for t-test
                        }
        
        # Save to JSON if requested
        if output_json_path:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n  JSON results saved to: {output_json_path}")
        
        # Generate and save Markdown tables if requested
        if output_md_path:
            md_content = VerbalizationEvaluator.format_quality_tables(
                result['statistics'], 
                metrics
            )
            with open(output_md_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
            print(f"  Markdown tables saved to: {output_md_path}")
        
        return result
    
    @staticmethod
    def format_quality_tables(
        stats: Dict[str, Any],
        metrics: Optional[List[str]] = None
    ) -> str:

        lines = []
        lines.append("# Verbalization Quality Analysis: Silver vs Gold\n")
        lines.append("=" * 100)
        lines.append("")
        
        # Determine metrics to display
        if metrics is None:
            # Get all metrics from first group
            first_group = list(stats.values())[0]
            first_key = list(first_group.keys())[0]
            metrics = list(first_group[first_key].keys())
        
        # Remove 'scores' from display metrics if present
        display_metrics = [m for m in metrics if m != 'scores']
        
        # 1. TABLE BY QUALITY (Main comparison)
        lines.append("\n## 1. Overall Quality Comparison\n")
        
        if 'by_quality' in stats:
            lines.append("| Quality | n | " + " | ".join([f"{m}" for m in display_metrics]) + " |")
            lines.append("|---------|---|" + "|".join(["-------" for _ in display_metrics]) + "|")
            
            for quality in sorted(stats['by_quality'].keys()):
                quality_stats = stats['by_quality'][quality]
                
                # Get sample count
                n = 0
                for metric in display_metrics:
                    if metric in quality_stats:
                        n = quality_stats[metric]['n']
                        break
                
                row = f"| **{quality}** | {n} |"
                
                for metric in display_metrics:
                    if metric in quality_stats:
                        mean = quality_stats[metric]['mean']
                        std = quality_stats[metric]['std']
                        row += f" {mean:.2f}±{std:.2f} |"
                    else:
                        row += " N/A |"
                
                lines.append(row)
            
            lines.append("")
            lines.append("*Values shown as mean±std*\n")
        
        # 2. TABLE BY QUALITY AND LANGUAGE
        if 'by_quality_language' in stats:
            lines.append("\n## 2. Quality Comparison by Language\n")
            
            # Group by quality for better organization
            qualities = {}
            for quality_lang_key in stats['by_quality_language'].keys():
                quality, lang = quality_lang_key.split('|')
                if quality not in qualities:
                    qualities[quality] = []
                qualities[quality].append((quality_lang_key, lang))
            
            for quality in sorted(qualities.keys()):
                lines.append(f"\n### Quality: {quality}\n")
                
                lines.append("| Language | n | " + " | ".join([f"{m}" for m in display_metrics]) + " |")
                lines.append("|----------|---|" + "|".join(["-------" for _ in display_metrics]) + "|")
                
                for quality_lang_key, lang in sorted(qualities[quality]):
                    quality_lang_stats = stats['by_quality_language'][quality_lang_key]
                    
                    # Get sample count
                    n = 0
                    for metric in display_metrics:
                        if metric in quality_lang_stats:
                            n = quality_lang_stats[metric]['n']
                            break
                    
                    row = f"| {lang} | {n} |"
                    
                    for metric in display_metrics:
                        if metric in quality_lang_stats:
                            mean = quality_lang_stats[metric]['mean']
                            std = quality_lang_stats[metric]['std']
                            row += f" {mean:.2f}±{std:.2f} |"
                        else:
                            row += " N/A |"
                    
                    lines.append(row)
                
                lines.append("")
            
            lines.append("*Values shown as mean±std*\n")
        
        # 3. TABLE BY QUALITY AND TYPE
        if 'by_quality_type' in stats:
            lines.append("\n## 3. Quality Comparison by Type\n")
            
            # Group by quality
            qualities = {}
            for quality_type_key in stats['by_quality_type'].keys():
                quality, type_ = quality_type_key.split('|')
                if quality not in qualities:
                    qualities[quality] = []
                qualities[quality].append((quality_type_key, type_))
            
            for quality in sorted(qualities.keys()):
                lines.append(f"\n### Quality: {quality}\n")
                
                lines.append("| Type | n | " + " | ".join([f"{m}" for m in display_metrics]) + " |")
                lines.append("|------|---|" + "|".join(["-------" for _ in display_metrics]) + "|")
                
                for quality_type_key, type_ in sorted(qualities[quality]):
                    quality_type_stats = stats['by_quality_type'][quality_type_key]
                    
                    # Get sample count
                    n = 0
                    for metric in display_metrics:
                        if metric in quality_type_stats:
                            n = quality_type_stats[metric]['n']
                            break
                    
                    row = f"| {type_} | {n} |"
                    
                    for metric in display_metrics:
                        if metric in quality_type_stats:
                            mean = quality_type_stats[metric]['mean']
                            std = quality_type_stats[metric]['std']
                            row += f" {mean:.2f}±{std:.2f} |"
                        else:
                            row += " N/A |"
                    
                    lines.append(row)
                
                lines.append("")
            
            lines.append("*Values shown as mean±std*\n")
        
        return "\n".join(lines)
    
    @staticmethod
    def compute_statistics_by_metadata(
        file_path: Path,
        metrics: Optional[List[str]] = None,
        xml_path: Optional[Path] = None
    ) -> Dict[str, Any]:

        print(f"\nComputing metadata statistics for: {file_path.name}")
        
        # Load metadata mapping from XML
        metadata_mapping = VerbalizationEvaluator.load_longtail_metadata_mapping(xml_path)
        
        # Load data
        data = VerbalizationEvaluator.load_json_file(file_path)
        
        # Auto-detect available metrics from first entry if not specified
        if metrics is None:
            metrics = []
            for key in data[0].keys():
                if key.endswith('_score'):
                    metrics.append(key.replace('_score', ''))
        
        # Separate quality metrics from perplexity for better reporting
        quality_metrics = [m for m in metrics if m != 'perplexity']
        perplexity_metrics = [m for m in metrics if m == 'perplexity']
        
        if quality_metrics and perplexity_metrics:
            print(f"  Quality metrics found: {quality_metrics}")
            print(f"  Perplexity metric found: {perplexity_metrics}")
        else:
            print(f"  Metrics found: {metrics}")
        
        # Initialize containers
        stats = {
            'by_language': {},
            'by_type': {},
            'by_subtype': {},
            'by_category': {},
            'by_category_type_subtype': {},
            'by_type_language': {},
            'by_type_subtype_language': {},
            'by_num_triples': {},
            'by_type_num_triples': {}
        }
        
        # Group data by different criteria
        for entry in data:
            eid = entry.get('eid')
            lang = entry.get('language')
            
            # Get num_triples (convert to string for consistency)
            num_triples = str(entry.get('num_triples', 'unknown'))
            
            # Get metadata from XML mapping
            if eid in metadata_mapping:
                metadata = metadata_mapping[eid]
                type_ = metadata['type']
                subtype = metadata['subtype']
                category = metadata['category']
            else:
                raise ValueError(f"Entry with eid={eid} not found in XML metadata mapping")
            
            # By language
            if lang not in stats['by_language']:
                stats['by_language'][lang] = {m: [] for m in metrics}
            
            # By type
            if type_ not in stats['by_type']:
                stats['by_type'][type_] = {m: [] for m in metrics}
            
            # By subtype
            if subtype not in stats['by_subtype']:
                stats['by_subtype'][subtype] = {m: [] for m in metrics}
            
            # By category
            if category not in stats['by_category']:
                stats['by_category'][category] = {m: [] for m in metrics}
            
            # By category-type-subtype
            cat_key = f"{category}|{type_}|{subtype}"
            if cat_key not in stats['by_category_type_subtype']:
                stats['by_category_type_subtype'][cat_key] = {m: [] for m in metrics}
            
            # By type-language
            type_lang_key = f"{type_}|{lang}"
            if type_lang_key not in stats['by_type_language']:
                stats['by_type_language'][type_lang_key] = {m: [] for m in metrics}
            
            # By type-subtype-language
            type_subtype_lang_key = f"{type_}|{subtype}|{lang}"
            if type_subtype_lang_key not in stats['by_type_subtype_language']:
                stats['by_type_subtype_language'][type_subtype_lang_key] = {m: [] for m in metrics}
            
            # By num_triples
            if num_triples not in stats['by_num_triples']:
                stats['by_num_triples'][num_triples] = {m: [] for m in metrics}
            
            # By type-num_triples
            type_numtriples_key = f"{type_}|{num_triples}"
            if type_numtriples_key not in stats['by_type_num_triples']:
                stats['by_type_num_triples'][type_numtriples_key] = {m: [] for m in metrics}
            
            # Collect scores
            for metric in metrics:
                score_key = f'{metric}_score'
                if score_key in entry:
                    score = entry[score_key]
                    stats['by_language'][lang][metric].append(score)
                    stats['by_type'][type_][metric].append(score)
                    stats['by_subtype'][subtype][metric].append(score)
                    stats['by_category'][category][metric].append(score)
                    stats['by_category_type_subtype'][cat_key][metric].append(score)
                    stats['by_type_language'][type_lang_key][metric].append(score)
                    stats['by_type_subtype_language'][type_subtype_lang_key][metric].append(score)
                    stats['by_num_triples'][num_triples][metric].append(score)
                    stats['by_type_num_triples'][type_numtriples_key][metric].append(score)

        
        # Compute summary statistics
        result = {}
        
        for group_name in ['by_language', 'by_type', 'by_subtype', 'by_category', 'by_category_type_subtype', 
                           'by_type_language', 'by_type_subtype_language', 'by_num_triples', 'by_type_num_triples']:
            result[group_name] = {}
            
            for group_key, metric_scores in stats[group_name].items():
                result[group_name][group_key] = {}
                
                for metric, scores in metric_scores.items():
                    if scores:  # Only compute if we have scores
                        result[group_name][group_key][metric] = {
                            'n': len(scores),
                            'mean': float(np.mean(scores)),
                            'std': float(np.std(scores)),
                            'min': float(np.min(scores)),
                            'max': float(np.max(scores))
                        }
        
        return result
    
    @staticmethod
    def format_statistics_tables(
        stats: Dict[str, Any],
        metrics: Optional[List[str]] = None,
        output_path: Optional[Path] = None
    ) -> str:

        lines = []
        lines.append("=" * 120)
        lines.append("STATISTICS BY METADATA - COMPARISON TABLES")
        lines.append("=" * 120)
        
        # Determine metrics to display
        if metrics is None:
            # Get all metrics from first group
            first_group = list(stats.values())[0]
            first_key = list(first_group.keys())[0]
            metrics = list(first_group[first_key].keys())
        
        # Helper function to create comparison table
        def create_comparison_table(group_data: Dict[str, Dict[str, Dict]], title: str, group_label: str):
            table_lines = []
            table_lines.append("\n" + "=" * 120)
            table_lines.append(f"TABLE: {title}")
            table_lines.append("=" * 120)
            
            if not group_data:
                table_lines.append("No data available")
                return table_lines
            
            # Get all groups (languages, types, etc.)
            groups = sorted(group_data.keys())
            
            # Create header
            header = f"{group_label:20s} | {'n':>6s} |"
            for metric in metrics:
                header += f" {metric:>12s} |"
            table_lines.append(header)
            table_lines.append("-" * 120)
            
            # Add data rows
            for group in groups:
                if group not in group_data:
                    continue
                    
                group_stats = group_data[group]
                
                # Get sample count from first metric
                n = 0
                for metric in metrics:
                    if metric in group_stats:
                        n = group_stats[metric]['n']
                        break
                
                row = f"{group:20s} | {n:>6d} |"
                
                for metric in metrics:
                    if metric in group_stats:
                        mean = group_stats[metric]['mean']
                        std = group_stats[metric]['std']
                        row += f" {mean:>5.2f}±{std:<.2f} |"
                    else:
                        row += f" {'N/A':>12s} |"
                
                table_lines.append(row)
            
            table_lines.append("=" * 120)
            
            # Add mean±std explanation
            table_lines.append(f"Note: Values shown as mean±std")
            
            return table_lines
        
        # 1. TABLE BY LANGUAGE
        if 'by_language' in stats:
            lines.extend(create_comparison_table(
                stats['by_language'],
                "COMPARISON BY LANGUAGE",
                "Language"
            ))
        
        # 2. TABLE BY TYPE AND LANGUAGE
        if 'by_type_language' in stats:
            lines.append("\n" + "=" * 120)
            lines.append("TABLE: COMPARISON BY TYPE AND LANGUAGE")
            lines.append("=" * 120)
            
            # Group by type for better organization
            types = {}
            for type_lang_key in stats['by_type_language'].keys():
                type_, lang = type_lang_key.split('|')
                if type_ not in types:
                    types[type_] = []
                types[type_].append((type_lang_key, lang))
            
            for type_ in sorted(types.keys()):
                lines.append(f"\nType: {type_}")
                lines.append("-" * 120)
                
                header = f"{'Language':20s} | {'n':>6s} |"
                for metric in metrics:
                    header += f" {metric:>12s} |"
                lines.append(header)
                lines.append("-" * 120)
                
                for type_lang_key, lang in sorted(types[type_]):
                    type_lang_stats = stats['by_type_language'][type_lang_key]
                    
                    # Get sample count
                    n = 0
                    for metric in metrics:
                        if metric in type_lang_stats:
                            n = type_lang_stats[metric]['n']
                            break
                    
                    row = f"{lang:20s} | {n:>6d} |"
                    
                    for metric in metrics:
                        if metric in type_lang_stats:
                            mean = type_lang_stats[metric]['mean']
                            std = type_lang_stats[metric]['std']
                            row += f" {mean:>5.2f}±{std:<.2f} |"
                        else:
                            row += f" {'N/A':>12s} |"
                    
                    lines.append(row)
            
            lines.append("=" * 120)
            lines.append("Note: Values shown as mean±std")
        
        # 3. TABLE BY TYPE, SUBTYPE AND LANGUAGE
        if 'by_type_subtype_language' in stats:
            lines.append("\n" + "=" * 120)
            lines.append("TABLE: COMPARISON BY TYPE, SUBTYPE AND LANGUAGE")
            lines.append("=" * 120)
            
            # Group by type and subtype for better organization
            type_subtypes = {}
            for type_subtype_lang_key in stats['by_type_subtype_language'].keys():
                type_, subtype, lang = type_subtype_lang_key.split('|')
                type_subtype = f"{type_}|{subtype}"
                if type_subtype not in type_subtypes:
                    type_subtypes[type_subtype] = []
                type_subtypes[type_subtype].append((type_subtype_lang_key, lang))
            
            for type_subtype in sorted(type_subtypes.keys()):
                type_, subtype = type_subtype.split('|')
                lines.append(f"\nType: {type_} | Subtype: {subtype}")
                lines.append("-" * 120)
                
                header = f"{'Language':20s} | {'n':>6s} |"
                for metric in metrics:
                    header += f" {metric:>12s} |"
                lines.append(header)
                lines.append("-" * 120)
                
                for type_subtype_lang_key, lang in sorted(type_subtypes[type_subtype]):
                    type_subtype_lang_stats = stats['by_type_subtype_language'][type_subtype_lang_key]
                    
                    # Get sample count
                    n = 0
                    for metric in metrics:
                        if metric in type_subtype_lang_stats:
                            n = type_subtype_lang_stats[metric]['n']
                            break
                    
                    row = f"{lang:20s} | {n:>6d} |"
                    
                    for metric in metrics:
                        if metric in type_subtype_lang_stats:
                            mean = type_subtype_lang_stats[metric]['mean']
                            std = type_subtype_lang_stats[metric]['std']
                            row += f" {mean:>5.2f}±{std:<.2f} |"
                        else:
                            row += f" {'N/A':>12s} |"
                    
                    lines.append(row)
            
            lines.append("=" * 120)
            lines.append("Note: Values shown as mean±std")
        
        # 4. TABLE BY NUM_TRIPLES
        if 'by_num_triples' in stats:
            lines.append("\n" + "=" * 120)
            lines.append("TABLE: COMPARISON BY NUMBER OF TRIPLES")
            lines.append("=" * 120)
            
            # Sort by number of triples (convert to int for sorting)
            sorted_keys = sorted(stats['by_num_triples'].keys(), 
                               key=lambda x: int(x) if x.isdigit() else 999)
            
            header = f"{'Num Triples':20s} | {'n':>6s} |"
            for metric in metrics:
                header += f" {metric:>12s} |"
            lines.append(header)
            lines.append("-" * 120)
            
            for num_triples in sorted_keys:
                num_triples_stats = stats['by_num_triples'][num_triples]
                
                # Get sample count
                n = 0
                for metric in metrics:
                    if metric in num_triples_stats:
                        n = num_triples_stats[metric]['n']
                        break
                
                row = f"{num_triples:20s} | {n:>6d} |"
                
                for metric in metrics:
                    if metric in num_triples_stats:
                        mean = num_triples_stats[metric]['mean']
                        std = num_triples_stats[metric]['std']
                        row += f" {mean:>5.2f}±{std:<.2f} |"
                    else:
                        row += f" {'N/A':>12s} |"
                
                lines.append(row)
            
            lines.append("=" * 120)
            lines.append("Note: Values shown as mean±std")
        
        # 5. TABLE BY TYPE AND NUM_TRIPLES
        if 'by_type_num_triples' in stats:
            lines.append("\n" + "=" * 120)
            lines.append("TABLE: COMPARISON BY TYPE AND NUMBER OF TRIPLES")
            lines.append("=" * 120)
            
            # Group by type for better organization
            types = {}
            for type_numtriples_key in stats['by_type_num_triples'].keys():
                type_, num_triples = type_numtriples_key.split('|')
                if type_ not in types:
                    types[type_] = []
                types[type_].append((type_numtriples_key, num_triples))
            
            for type_ in sorted(types.keys()):
                lines.append(f"\nType: {type_}")
                lines.append("-" * 120)
                
                header = f"{'Num Triples':20s} | {'n':>6s} |"
                for metric in metrics:
                    header += f" {metric:>12s} |"
                lines.append(header)
                lines.append("-" * 120)
                
                # Sort by num_triples
                sorted_entries = sorted(types[type_], 
                                      key=lambda x: int(x[1]) if x[1].isdigit() else 999)
                
                for type_numtriples_key, num_triples in sorted_entries:
                    type_numtriples_stats = stats['by_type_num_triples'][type_numtriples_key]
                    
                    # Get sample count
                    n = 0
                    for metric in metrics:
                        if metric in type_numtriples_stats:
                            n = type_numtriples_stats[metric]['n']
                            break
                    
                    row = f"{num_triples:20s} | {n:>6d} |"
                    
                    for metric in metrics:
                        if metric in type_numtriples_stats:
                            mean = type_numtriples_stats[metric]['mean']
                            std = type_numtriples_stats[metric]['std']
                            row += f" {mean:>5.2f}±{std:<.2f} |"
                        else:
                            row += f" {'N/A':>12s} |"
                    
                    lines.append(row)
            
            lines.append("=" * 120)
            lines.append("Note: Values shown as mean±std")


        # 6. TABLE BY TYPE
        if 'by_type' in stats:
            lines.extend(create_comparison_table(
                stats['by_type'],
                "COMPARISON BY TYPE",
                "Type"
            ))
        
        # 7. TABLE BY SUBTYPE
        if 'by_subtype' in stats:
            lines.extend(create_comparison_table(
                stats['by_subtype'],
                "COMPARISON BY SUBTYPE",
                "Subtype"
            ))
        
        # 4. TABLE BY CATEGORY
        if 'by_category' in stats:
            lines.extend(create_comparison_table(
                stats['by_category'],
                "COMPARISON BY CATEGORY",
                "Category"
            ))
        
        # 8. TABLE BY CATEGORY-TYPE-SUBTYPE (detailed breakdown)
            lines.append("\n" + "=" * 120)
            lines.append("TABLE: COMPARISON BY CATEGORY (with Type and Subtype)")
            lines.append("=" * 120)
            
            # Group by category for better organization
            categories = {}
            for cat_key in stats['by_category_type_subtype'].keys():
                category, type_, subtype = cat_key.split('|')
                if category not in categories:
                    categories[category] = []
                categories[category].append((cat_key, type_, subtype))
            
            for category in sorted(categories.keys()):
                lines.append(f"\nCategory: {category}")
                lines.append("-" * 120)
                
                header = f"{'Type':15s} | {'Subtype':20s} | {'n':>6s} |"
                for metric in metrics:
                    header += f" {metric:>12s} |"
                lines.append(header)
                lines.append("-" * 120)
                
                for cat_key, type_, subtype in sorted(categories[category]):
                    cat_stats = stats['by_category_type_subtype'][cat_key]
                    
                    # Get sample count
                    n = 0
                    for metric in metrics:
                        if metric in cat_stats:
                            n = cat_stats[metric]['n']
                            break
                    
                    row = f"{type_:15s} | {subtype:20s} | {n:>6d} |"
                    
                    for metric in metrics:
                        if metric in cat_stats:
                            mean = cat_stats[metric]['mean']
                            std = cat_stats[metric]['std']
                            row += f" {mean:>5.2f}±{std:<.2f} |"
                        else:
                            row += f" {'N/A':>12s} |"
                    
                    lines.append(row)
            
            lines.append("=" * 120)
        
        result = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"\nStatistics tables saved to: {output_path}")
        
        return result
    
    @staticmethod
    def summarize_results(
        results: Dict[str, Dict[str, Any]],
        output_path: Optional[Path] = None
    ) -> str:

        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append("EVALUATION SUMMARY")
        summary_lines.append("=" * 80)
        
        for method_name, method_results in results.items():
            summary_lines.append(f"\n{method_name.upper()}")
            summary_lines.append("-" * 80)
            
            for language, lang_results in method_results.items():
                summary_lines.append(f"\n  Language: {language}")
                
                for metric, metric_results in lang_results.items():
                    avg = metric_results['average']
                    std = metric_results['std']
                    summary_lines.append(f"    {metric:12s}: {avg:.4f} (±{std:.4f})")
        
        summary_lines.append("\n" + "=" * 80)
        summary_lines.append("BEST METHODS PER METRIC (English)")
        summary_lines.append("=" * 80)
        
        for metric in ['bertscore', 'bertscore_rescaled', 'bleu', 'chrf', 'meteor', 'rouge1', 'rouge2', 'rougeL']:
            try:
                best_method, best_score = VerbalizationEvaluator.get_best_method(
                    results, metric, 'en'
                )
                summary_lines.append(f"{metric:12s}: {best_method} ({best_score:.4f})")
            except:
                pass
        
        summary_lines.append("=" * 80)
        
        summary = "\n".join(summary_lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            print(f"Summary saved to: {output_path}")
        
        return summary
    

    # ===========================
    # Perplexity
    # ===========================

    @staticmethod
    def extract_model_name_from_filename(file_path: Path) -> Optional[str]:

        filename = file_path.stem
        
        # Try to match common patterns
        # Pattern 1: *_<org>_<model>.json
        match = re.search(r'_([a-zA-Z0-9-]+)_([a-zA-Z0-9.-]+)$', filename)
        if match:
            org = match.group(1)
            model = match.group(2)
            return f"{org}/{model}"
        
        return None
    
    @staticmethod
    def calculate_perplexity(
        texts: List[str],
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str = "cuda"
    ) -> List[float]:

        batch_size = 1
        model.eval()
        perplexities = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Calculate loss for each item in batch
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                
            perplexity = torch.exp(loss)
            perplexities.append(perplexity.item())

        return perplexities

    def calculate_perplexity_for_file(
        self,
        file_path: Path,
        prediction_key: str = 'prediction',
        model_name: Optional[str] = None,
        device: str = "cuda",
        batch_size: int = 8,
        save_individual_scores: bool = True,
        skip_existing: bool = True
    ) -> Dict[str, Any]:

        print(f"\nCalculating perplexity for: {file_path.name}")
        
        # Load data
        data = self.load_json_file(file_path)
        
        # Check if perplexity already exists
        if skip_existing and all('perplexity_score' in entry for entry in data):
            print(f"  Perplexity already computed, skipping...")
            
            # Group by language and compute statistics
            data_by_language = self.group_by_language(data)
            results = {}
            
            for language, lang_data in data_by_language.items():
                perplexities = [entry['perplexity_score'] for entry in lang_data]
                results[language] = {
                    'num_samples': len(perplexities),
                    'average': float(np.mean(perplexities)),
                    'std': float(np.std(perplexities)),
                    'min': float(np.min(perplexities)),
                    'max': float(np.max(perplexities))
                }
                print(f"  Language: {language} - Perplexity: {results[language]['average']:.4f} "
                      f"(±{results[language]['std']:.4f})")
            
            return results
        
        # Extract model name from filename if not provided
        if model_name is None:
            model_name = self.extract_model_name_from_filename(file_path)
            if model_name is None:
                raise ValueError(f"Could not extract model name from filename: {file_path.name}. "
                               f"Please provide model_name parameter.")
        
        print(f"  Loading model: {model_name}")
        
        # Check device availability
        if device == "cuda" and not torch.cuda.is_available():
            print(f"  CUDA not available, switching to CPU")
            device = "cpu"
        
        # Load model and tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
            
            model_kwargs = {
                'low_cpu_mem_usage': True,
                'return_dict': True,
                'torch_dtype': torch.bfloat16,
                'device_map': "auto"
            }
            
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            
            if device == "cpu":
                model = model.to(device)
            
            print(f"  Model loaded successfully on {device}")
            
        except Exception as e:
            print(f"  Error loading model: {e}")
            raise
        
        # Group data by language
        data_by_language = self.group_by_language(data)
        results = {}
        
        # Calculate perplexity for each language
        for language, lang_data in data_by_language.items():
            print(f"  Language: {language} ({len(lang_data)} samples)")
            print(f"    Calculating perplexity...")
            
            # Extract predictions
            predictions = [entry[prediction_key] for entry in lang_data]
            
            # Calculate perplexity
            perplexities = self.calculate_perplexity(
                predictions,
                model,
                tokenizer,
                #batch_size=batch_size,
                device=device
            )
            
            # Save scores back to entries
            if save_individual_scores:
                for i, entry in enumerate(lang_data):
                    entry['perplexity_score'] = float(perplexities[i])
            
            # Store statistics
            results[language] = {
                'num_samples': len(perplexities),
                'average': float(np.mean(perplexities)),
                'std': float(np.std(perplexities)),
                'min': float(np.min(perplexities)),
                'max': float(np.max(perplexities))
            }
            
            print(f"    Perplexity: {results[language]['average']:.4f} "
                  f"(±{results[language]['std']:.4f})")
        
        # Save scores back to JSON if requested
        if save_individual_scores:
            self._save_scores_to_file(file_path, data)
        
        # Clean up
        del model
        del tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        return results
    
    def calculate_perplexity_for_multiple_files(
        self,
        file_paths: List[Path],
        prediction_key: str = 'prediction',
        device: str = "cuda",
        batch_size: int = 8,
        save_individual_scores: bool = True,
        skip_existing: bool = True
    ) -> Dict[str, Dict[str, Any]]:

        all_results = {}
        
        for file_path in file_paths:
            method_name = file_path.stem
            
            try:
                results = self.calculate_perplexity_for_file(
                    file_path,
                    prediction_key=prediction_key,
                    device=device,
                    batch_size=batch_size,
                    save_individual_scores=save_individual_scores,
                    skip_existing=skip_existing
                )
                all_results[method_name] = results
                
            except Exception as e:
                print(f"  Error processing {file_path.name}: {e}")
                continue
        
        return all_results
