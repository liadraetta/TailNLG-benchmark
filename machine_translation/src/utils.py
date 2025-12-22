"""
Utility functions for the translation and post-editing pipeline.
"""
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict


class PipelineUtils:
    """Static utility methods for the translation pipeline."""
    
    @staticmethod
    def save_incremental(record: Dict, output_dir: Path) -> bool:
        """
        Save or update a single record incrementally to the unified corpus.
        
        Args:
            record: Single record to save
            output_dir: Output directory path
            
        Returns:
            True if successful, False otherwise
        """
        logger = logging.getLogger(__name__)
        json_path = output_dir / "unified_corpus.json"
        
        try:
            # Load existing records
            existing_records = []
            if json_path.exists():
                with open(json_path, 'r', encoding='utf-8') as f:
                    existing_records = json.load(f)
            
            # Find and update or append
            unique_id = record.get('unique_id')
            updated = False
            
            for i, existing_rec in enumerate(existing_records):
                if existing_rec.get('unique_id') == unique_id:
                    existing_records[i] = record
                    updated = True
                    break
            
            if not updated:
                existing_records.append(record)
            
            # Save back
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(existing_records, f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save incremental record {record.get('unique_id')}: {e}")
            return False
    
    @staticmethod
    def load_existing_records(output_dir: Path) -> Dict[str, Dict]:
        """
        Load existing records from unified corpus as a dictionary keyed by unique_id.
        
        Args:
            output_dir: Output directory path
            
        Returns:
            Dictionary mapping unique_id to record
        """
        logger = logging.getLogger(__name__)
        json_path = output_dir / "unified_corpus.json"
        
        if not json_path.exists():
            return {}
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                records = json.load(f)
            
            return {rec.get('unique_id'): rec for rec in records if rec.get('unique_id')}
            
        except Exception as e:
            logger.error(f"Failed to load existing records: {e}")
            return {}
    
    @staticmethod
    def save_intermediate_results(results: List[Dict], lang_code: str, output_dir: Path):
        """
        Save intermediate results to JSON file.
        
        Args:
            results: List of processed records
            lang_code: Language code for filename
            output_dir: Output directory path
        """
        logger = logging.getLogger(__name__)
        output_path = output_dir / f"intermediate_{lang_code}.json"
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Intermediate results saved: {len(results)} records to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save intermediate results: {e}")
    
    @staticmethod
    def save_final_corpus(all_records: List[Dict], output_dir: Path) -> tuple:
        """
        Save final unified corpus to JSON and CSV.
        
        Args:
            all_records: All processed records
            output_dir: Output directory path
            
        Returns:
            Tuple of (csv_path, json_path)
        """
        logger = logging.getLogger(__name__)
        
        # Save complete JSON (without timestamp - directory already has it)
        json_path = output_dir / "unified_corpus.json"
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(all_records, f, ensure_ascii=False, indent=2)
            logger.info(f"Complete corpus saved to: {json_path}")
        except Exception as e:
            logger.error(f"Failed to save JSON corpus: {e}")
            json_path = None
        
        # Create CSV version
        csv_path = None
        try:
            df = pd.DataFrame(all_records)
            
            # Reorder columns for better readability
            column_order = [
                'unique_id', 'QID', 'configuration', 'category', 'type', 'triples_number',
                'source_language', 'source_file',
                'triples_en', 'triples_es', 'triples_it',
                'annotation_en', 'annotation_en_is_gold', 
                'annotation_en_xcomet_sentence_scores', 'annotation_en_xcomet_system_score', 'annotation_en_xcomet_error_spans',
                'annotation_en_xcomet_minor_errors', 'annotation_en_xcomet_major_errors', 'annotation_en_xcomet_critical_errors',
                'annotation_en_xtower_prompt', 'annotation_en_xtower_explanations', 'annotation_en_xtower_corrected_translation', 'annotation_en_xtower_output',
                'annotation_es', 'annotation_es_is_gold',
                'annotation_es_xcomet_sentence_scores', 'annotation_es_xcomet_system_score', 'annotation_es_xcomet_error_spans',
                'annotation_es_xcomet_minor_errors', 'annotation_es_xcomet_major_errors', 'annotation_es_xcomet_critical_errors',
                'annotation_es_xtower_prompt', 'annotation_es_xtower_explanations', 'annotation_es_xtower_corrected_translation', 'annotation_es_xtower_output',
                'annotation_it', 'annotation_it_is_gold',
                'annotation_it_xcomet_sentence_scores', 'annotation_it_xcomet_system_score', 'annotation_it_xcomet_error_spans',
                'annotation_it_xcomet_minor_errors', 'annotation_it_xcomet_major_errors', 'annotation_it_xcomet_critical_errors',
                'annotation_it_xtower_prompt', 'annotation_it_xtower_explanations', 'annotation_it_xtower_corrected_translation', 'annotation_it_xtower_output',
                'notes', 'Order preserved'
            ]
            
            # Only include columns that exist
            column_order = [col for col in column_order if col in df.columns]
            df = df[column_order]
            
            csv_path = output_dir / "unified_corpus.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8')
            logger.info(f"CSV corpus saved to: {csv_path}")
        except Exception as e:
            logger.error(f"Failed to save CSV corpus: {e}")
        
        return csv_path, json_path
    
    @staticmethod
    def create_summary_report(all_records: List[Dict], output_dir: Path, timestamp: str):
        """
        Create comprehensive summary statistics report.
        
        Args:
            all_records: All processed records
            output_dir: Output directory path
            timestamp: Timestamp string for documentation
        """
        logger = logging.getLogger(__name__)
        report_path = output_dir / "summary_report.txt"
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("TRANSLATION AND POST-EDITING SUMMARY REPORT\n")
                f.write("=" * 80 + "\n")
                f.write(f"Generated: {timestamp}\n")
                f.write(f"Total records processed: {len(all_records)}\n")
                f.write("=" * 80 + "\n\n")
                
                # ===== SECTION 1: DATASET COMPOSITION =====
                f.write("1. DATASET COMPOSITION\n")
                f.write("-" * 80 + "\n\n")
                
                # Count by source language
                source_lang_counts = {}
                for rec in all_records:
                    lang = rec.get('source_language', 'unknown')
                    source_lang_counts[lang] = source_lang_counts.get(lang, 0) + 1
                
                f.write("Records by source language:\n")
                for lang, count in sorted(source_lang_counts.items()):
                    percentage = (count / len(all_records) * 100) if all_records else 0
                    f.write(f"  {lang.upper()}: {count:4d} ({percentage:5.2f}%)\n")
                f.write("\n")
                
                # Count by category
                category_counts = {}
                for rec in all_records:
                    cat = rec.get('category', 'unknown')
                    category_counts[cat] = category_counts.get(cat, 0) + 1
                
                f.write("Records by category:\n")
                for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / len(all_records) * 100) if all_records else 0
                    f.write(f"  {cat:25s}: {count:4d} ({percentage:5.2f}%)\n")
                f.write("\n")
                
                # Count by type (head/long_tail)
                type_counts = {}
                for rec in all_records:
                    type_val = rec.get('type', 'unknown')
                    type_counts[type_val] = type_counts.get(type_val, 0) + 1
                
                f.write("Records by type:\n")
                for type_val, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / len(all_records) * 100) if all_records else 0
                    f.write(f"  {type_val:25s}: {count:4d} ({percentage:5.2f}%)\n")
                f.write("\n")
                
                # Count by configuration
                config_counts = {}
                for rec in all_records:
                    config = rec.get('configuration', 'unknown')
                    config_counts[config] = config_counts.get(config, 0) + 1
                
                f.write("Records by configuration:\n")
                for config, count in sorted(config_counts.items()):
                    percentage = (count / len(all_records) * 100) if all_records else 0
                    f.write(f"  {config:15s}: {count:4d} ({percentage:5.2f}%)\n")
                f.write("\n")
                
                # Triples number statistics
                triples_numbers = [rec.get('triples_number', 0) for rec in all_records]
                if triples_numbers:
                    f.write("Triples per record:\n")
                    f.write(f"  Min:     {min(triples_numbers)}\n")
                    f.write(f"  Max:     {max(triples_numbers)}\n")
                    f.write(f"  Average: {sum(triples_numbers) / len(triples_numbers):.2f}\n")
                f.write("\n\n")
                
                # ===== SECTION 2: TRANSLATION STATISTICS =====
                f.write("2. TRANSLATION STATISTICS\n")
                f.write("-" * 80 + "\n\n")
                
                for lang in ['en', 'es', 'it']:
                    PipelineUtils._write_language_translation_stats(f, all_records, lang)
                
                # ===== SECTION 3: XCOMET QUALITY SCORES =====
                f.write("\n3. XCOMET QUALITY ASSESSMENT\n")
                f.write("-" * 80 + "\n\n")
                
                for lang in ['en', 'es', 'it']:
                    PipelineUtils._write_xcomet_stats(f, all_records, lang)
                
                # ===== SECTION 4: XTOWER ERROR ANALYSIS =====
                f.write("\n4. XTOWER ERROR ANALYSIS\n")
                f.write("-" * 80 + "\n\n")
                
                for lang in ['en', 'es', 'it']:
                    PipelineUtils._write_xtower_stats(f, all_records, lang)
            
            logger.info(f"Summary report saved to: {report_path}")
        except Exception as e:
            logger.error(f"Failed to create summary report: {e}")
    
    @staticmethod
    def _write_language_translation_stats(f, all_records: List[Dict], lang: str):
        """Write translation statistics for a specific language."""
        import json
        
        f.write(f"{lang.upper()} Translations:\n")
        
        total = len(all_records)
        gold_count = sum(1 for rec in all_records if rec.get(f'annotation_{lang}_is_gold', False))
        translated_count = sum(1 for rec in all_records if not rec.get(f'annotation_{lang}_is_gold', False) and rec.get(f'annotation_{lang}'))
        
        f.write(f"  Total records:       {total:4d}\n")
        f.write(f"  Gold annotations:    {gold_count:4d} ({gold_count/total*100:5.2f}%)\n")
        f.write(f"  Translated:          {translated_count:4d} ({translated_count/total*100:5.2f}%)\n")
        f.write("\n")
    
    @staticmethod
    def _write_xcomet_stats(f, all_records: List[Dict], lang: str):
        """Write XCOMET statistics for a specific language."""
        import json
        
        f.write(f"{lang.upper()} - XCOMET Scores:\n")
        
        # Collect scores
        sentence_scores = []
        system_scores = []
        quality_categories = {'excellent': 0, 'good': 0, 'moderate': 0, 'weak': 0}
        
        # Error span statistics
        total_errors = 0
        error_severities = {'minor': 0, 'major': 0, 'critical': 0}
        
        for rec in all_records:
            # Skip gold annotations
            if rec.get(f'annotation_{lang}_is_gold', False):
                continue
            
            # Sentence scores
            score = rec.get(f'annotation_{lang}_xcomet_sentence_scores')
            if score is not None:
                sentence_scores.append(score)
                
                # Categorize score
                if score >= 0.80:
                    quality_categories['excellent'] += 1
                elif score >= 0.60:
                    quality_categories['good'] += 1
                elif score >= 0.40:
                    quality_categories['moderate'] += 1
                else:
                    quality_categories['weak'] += 1
            
            # System scores
            sys_score = rec.get(f'annotation_{lang}_xcomet_system_score')
            if sys_score is not None:
                system_scores.append(sys_score)
            
            # Error spans
            error_spans_str = rec.get(f'annotation_{lang}_xcomet_error_spans')
            if error_spans_str:
                try:
                    error_spans = json.loads(error_spans_str) if isinstance(error_spans_str, str) else error_spans_str
                    total_errors += len(error_spans)
                    for error in error_spans:
                        severity = error.get('severity', 'unknown')
                        error_severities[severity] = error_severities.get(severity, 0) + 1
                except:
                    pass
        
        if sentence_scores:
            f.write(f"  Sentence-level scores ({len(sentence_scores)} translations):\n")
            f.write(f"    Min:     {min(sentence_scores):.4f}\n")
            f.write(f"    Max:     {max(sentence_scores):.4f}\n")
            f.write(f"    Average: {sum(sentence_scores)/len(sentence_scores):.4f}\n")
            f.write(f"    Median:  {sorted(sentence_scores)[len(sentence_scores)//2]:.4f}\n")
            f.write("\n")
            
            f.write(f"  Quality distribution:\n")
            for category in ['excellent', 'good', 'moderate', 'weak']:
                count = quality_categories[category]
                percentage = (count / len(sentence_scores) * 100) if sentence_scores else 0
                f.write(f"    {category.capitalize():12s}: {count:4d} ({percentage:5.2f}%)\n")
            f.write("\n")
        
        if system_scores:
            f.write(f"  System-level scores ({len(system_scores)} translations):\n")
            f.write(f"    Average: {sum(system_scores)/len(system_scores):.4f}\n")
            f.write("\n")
        
        if total_errors > 0:
            f.write(f"  Error spans detected: {total_errors}\n")
            f.write(f"    Average per translation: {total_errors/len(sentence_scores):.2f}\n")
            f.write(f"  Error severity breakdown:\n")
            for severity in ['minor', 'major', 'critical']:
                count = error_severities.get(severity, 0)
                percentage = (count / total_errors * 100) if total_errors > 0 else 0
                f.write(f"    {severity.capitalize():12s}: {count:4d} ({percentage:5.2f}%)\n")
        else:
            f.write(f"  No error spans detected\n")
        
        f.write("\n")
    
    @staticmethod
    def _write_xtower_stats(f, all_records: List[Dict], lang: str):
        """Write xTower statistics for a specific language."""
        
        f.write(f"{lang.upper()} - xTower Error Analysis:\n")
        
        total_analyzed = 0
        translations_with_corrections = 0
        translations_with_output = 0
        
        for rec in all_records:
            # Skip gold annotations
            if rec.get(f'annotation_{lang}_is_gold', False):
                continue
            
            # Check if xTower was run
            if rec.get(f'annotation_{lang}_xtower_prompt') is not None:
                total_analyzed += 1
                
                # Check if corrected translation exists
                if rec.get(f'annotation_{lang}_xtower_corrected_translation'):
                    translations_with_corrections += 1
                
                # Check if full output exists
                if rec.get(f'annotation_{lang}_xtower_output'):
                    translations_with_output += 1
        
        if total_analyzed > 0:
            f.write(f"  Translations analyzed: {total_analyzed}\n")
            f.write(f"  Translations with corrections: {translations_with_corrections} ({translations_with_corrections/total_analyzed*100:5.2f}%)\n")
            f.write(f"  Translations with full output: {translations_with_output} ({translations_with_output/total_analyzed*100:5.2f}%)\n")
        else:
            f.write(f"  No translations analyzed by xTower\n")
        
        f.write("\n")
    
    @staticmethod
    def load_data_file(file_path: Path) -> pd.DataFrame:
        """
        Load data from CSV or Excel file.
        
        Args:
            file_path: Path to data file
            
        Returns:
            DataFrame with loaded data
        """
        logger = logging.getLogger(__name__)
        
        try:
            if file_path.suffix.lower() == '.csv':
                return pd.read_csv(file_path, encoding='utf-8')
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                return pd.read_excel(file_path)
            else:
                logger.error(f"Unsupported file format: {file_path.suffix}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return pd.DataFrame()
