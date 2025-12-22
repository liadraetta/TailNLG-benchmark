"""
Translation pipeline processor.
"""
import json
import logging
import time
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from .deepl_translator import DeepLTranslator
from .xcomet_scorer import XCOMETScorer
from .xtower_analyzer import XTowerAnalyzer
from .base_xcomet_scorer import BaseXCOMETScorer
from .utils import PipelineUtils


class TranslationPipeline:
    """Main pipeline for processing translations with quality assessment."""
    
    # Language configurations
    LANG_CONFIGS = {
        'it': {'deepl_code': 'IT', 'folder': 'it', 'targets': ['en', 'es']},
        'en': {'deepl_code': 'EN', 'folder': 'en', 'targets': ['it', 'es']},
        'es': {'deepl_code': 'ES', 'folder': 'sp', 'targets': ['it', 'en']}
    }
    
    def __init__(
        self,
        deepl_translator: DeepLTranslator,
        xcomet_scorer: Optional[XCOMETScorer] = None,
        xtower_analyzer: Optional[XTowerAnalyzer] = None,
        test_mode: bool = False,
        test_records_limit: Optional[int] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize translation pipeline.
        
        Args:
            deepl_translator: DeepL translator instance
            xcomet_scorer: Optional XCOMET scorer instance
            xtower_analyzer: Optional xTower analyzer instance
            test_mode: Whether to use test mode (mock models)
            test_records_limit: Maximum number of records to process in test mode
            output_dir: Output directory for incremental saves
        """
        self.deepl = deepl_translator
        self.xcomet = xcomet_scorer
        self.xtower = xtower_analyzer
        self.test_mode = test_mode
        self.test_records_limit = test_records_limit
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
    def process_record(
        self,
        row: pd.Series,
        source_lang: str,
        file_source: str
    ) -> Dict:
        """
        Process a single record and generate translations.
        
        Args:
            row: DataFrame row with record data
            source_lang: Source language code
            file_source: Source filename
            
        Returns:
            Dictionary with processed record data
        """
        unique_id = row.get('unique_id', f'record_{row.name}')
        self.logger.info(f"  Processing record: {unique_id}")
        
        # Base record data
        record = {
            'unique_id': unique_id,
            'QID': row.get('QID', ''),
            'configuration': row.get('configuration', ''),
            'category': row.get('category', ''),
            'type': row.get('type', ''),
            'triples_number': row.get('triples_number', 0),
            'source_file': file_source,
            'source_language': source_lang
        }
        
        # Add triples in all languages
        record['triples_en'] = row.get('triples_en', '')
        record['triples_es'] = row.get('triples_es', '')
        record['triples_it'] = row.get('triples_it', '')
        
        # Get the gold annotation
        gold_annotation = row.get('annotation', '')
        
        # Check if annotation is NA
        is_na_annotation = (
            not gold_annotation or 
            pd.isna(gold_annotation) or 
            str(gold_annotation).strip().upper() == 'NA' or
            str(gold_annotation).strip() == ''
        )
        
        # Initialize annotation columns
        annotation_cols = self._initialize_annotation_columns()
        
        if is_na_annotation:
            self.logger.info(f"    Annotation is NA - skipping translation")
            annotation_cols['annotation_en'] = 'NA'
            annotation_cols['annotation_es'] = 'NA'
            annotation_cols['annotation_it'] = 'NA'
            annotation_cols[f'annotation_{source_lang}_is_gold'] = True
        else:
            # Set gold annotation
            annotation_cols[f'annotation_{source_lang}'] = gold_annotation
            annotation_cols[f'annotation_{source_lang}_is_gold'] = True
            
            # Translate to target languages
            target_langs = self.LANG_CONFIGS[source_lang]['targets']
            
            for target_lang in target_langs:
                self._process_translation(
                    gold_annotation,
                    source_lang,
                    target_lang,
                    annotation_cols
                )
        
        # Merge annotation data into record
        record.update(annotation_cols)
        
        # Add notes and order preserved
        record['notes'] = row.get('notes', '')
        record['notes'] = '' if pd.isna(record['notes']) else record['notes']
        record['Order preserved'] = row.get('Order preserved', '')
        
        # Save incrementally after translation
        if self.output_dir:
            PipelineUtils.save_incremental(record, self.output_dir)
        
        return record
    
    def _initialize_annotation_columns(self) -> Dict:
        """Initialize empty annotation columns."""
        return {
            'annotation_en': None,
            'annotation_es': None,
            'annotation_it': None,
            'annotation_en_is_gold': False,
            'annotation_es_is_gold': False,
            'annotation_it_is_gold': False,
            'annotation_en_xcomet_sentence_scores': None,
            'annotation_es_xcomet_sentence_scores': None,
            'annotation_it_xcomet_sentence_scores': None,
            'annotation_en_xcomet_system_score': None,
            'annotation_es_xcomet_system_score': None,
            'annotation_it_xcomet_system_score': None,
            'annotation_en_xcomet_error_spans': None,
            'annotation_es_xcomet_error_spans': None,
            'annotation_it_xcomet_error_spans': None,
            'annotation_en_xcomet_minor_errors': None,
            'annotation_es_xcomet_minor_errors': None,
            'annotation_it_xcomet_minor_errors': None,
            'annotation_en_xcomet_major_errors': None,
            'annotation_es_xcomet_major_errors': None,
            'annotation_it_xcomet_major_errors': None,
            'annotation_en_xcomet_critical_errors': None,
            'annotation_es_xcomet_critical_errors': None,
            'annotation_it_xcomet_critical_errors': None,
            'annotation_en_xtower_prompt': None,
            'annotation_es_xtower_prompt': None,
            'annotation_it_xtower_prompt': None,
            'annotation_en_xtower_explanations': None,
            'annotation_es_xtower_explanations': None,
            'annotation_it_xtower_explanations': None,
            'annotation_en_xtower_corrected_translation': None,
            'annotation_es_xtower_corrected_translation': None,
            'annotation_it_xtower_corrected_translation': None,
            'annotation_en_xtower_output': None,
            'annotation_es_xtower_output': None,
            'annotation_it_xtower_output': None,
        }
    
    def _process_translation(
        self,
        gold_annotation: str,
        source_lang: str,
        target_lang: str,
        annotation_cols: Dict
    ):
        """Process translation for a target language."""
        self.logger.info(f"    Translating {source_lang} -> {target_lang}...")
        
        # Translate
        translation = self.deepl.translate(gold_annotation, source_lang, target_lang)
        annotation_cols[f'annotation_{target_lang}'] = translation
        annotation_cols[f'annotation_{target_lang}_is_gold'] = False
        
        if not translation:
            annotation_cols[f'annotation_{target_lang}_errors'] = json.dumps(
                {"error": "Translation failed"}
            )
            return
        
        # Quality analysis will be done in batch later
        self.logger.info(f"    DeepL Translation completed")
    
    def _perform_quality_analysis(
        self,
        gold_annotation: str,
        translation: str,
        source_lang: str,
        target_lang: str,
        annotation_cols: Dict
    ):
        """Perform XCOMET and xTower quality analysis."""
        # Get XCOMET score
        self.logger.info(f"    Getting XCOMET score for {target_lang}...")
        xcomet_score = None
        
        if self.xcomet:
            xcomet_score = self.xcomet.score_translation(
                gold_annotation, translation, source_lang, target_lang
            )
        
        # Save XCOMET data
        annotation_cols[f'annotation_{target_lang}_xcomet_data'] = (
            json.dumps(xcomet_score, ensure_ascii=False) if xcomet_score else None
        )
        annotation_cols[f'annotation_{target_lang}_xcomet_sentence_scores'] = (
            xcomet_score.get('score') if xcomet_score else None
        )
        
        # Check if XCOMET score is valid
        xcomet_score_valid = xcomet_score and xcomet_score.get('score') is not None
        
        if not xcomet_score_valid:
            self.logger.warning(f"    XCOMET score is null - skipping xTower analysis")
            return
        
        self.logger.info(
            f"    XCOMET score: {xcomet_score['score']:.4f} "
            f"({xcomet_score['translation_quality_score']})"
        )
        
        # Analyze with xTower
        if self.xtower:
            self.logger.info(f"    Analyzing {target_lang} translation errors with xTower...")
            errors = self.xtower.analyze_translation(
                gold_annotation, translation, source_lang, target_lang, xcomet_score
            )
            
            if errors:
                annotation_cols[f'annotation_{target_lang}_xtower_prompt'] = (
                    errors.get('xtower_prompt')
                )
                annotation_cols[f'annotation_{target_lang}_errors'] = (
                    json.dumps(errors, ensure_ascii=False)
                )
                
                corrected = errors.get('corrected_translation')
                if corrected:
                    self.logger.info(f"    Corrected translation available")
    
    def batch_process_xcomet(self, all_records: List[Dict]) -> List[Dict]:
        """
        Process all records with XCOMET in batch.
        
        Args:
            all_records: List of records with translations
            
        Returns:
            Updated list of records with XCOMET scores
        """
        if not self.xcomet:
            self.logger.warning("XCOMET scorer not available")
            return all_records
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("BATCH PROCESSING WITH XCOMET")
        self.logger.info("=" * 80)
        
        processed_count = 0
        skipped_count = 0
        
        for record in all_records:
            source_lang = record.get('source_language')
            if not source_lang:
                continue
            
            gold_annotation = record.get(f'annotation_{source_lang}')
            if not gold_annotation or gold_annotation == 'NA':
                continue
            
            target_langs = self.LANG_CONFIGS[source_lang]['targets']
            record_updated = False
            
            for target_lang in target_langs:
                translation = record.get(f'annotation_{target_lang}')
                is_gold = record.get(f'annotation_{target_lang}_is_gold', False)
                
                # Skip if NA, gold, or no translation
                if not translation or translation == 'NA' or is_gold:
                    continue
                
                # Skip if already processed (has xcomet score)
                if record.get(f'annotation_{target_lang}_xcomet_sentence_scores') is not None:
                    skipped_count += 1
                    self.logger.info(
                        f"  Skipping {record['unique_id']} ({source_lang}->{target_lang}): "
                        f"already has XCOMET score"
                    )
                    continue
                
                # Score with XCOMET
                xcomet_score = self.xcomet.score_translation(
                    gold_annotation, translation, source_lang, target_lang
                )
                
                # Save XCOMET data in separate fields
                if xcomet_score:
                    record[f'annotation_{target_lang}_xcomet_sentence_scores'] = xcomet_score.get('score')
                    record[f'annotation_{target_lang}_xcomet_system_score'] = xcomet_score.get('system_score')
                    record[f'annotation_{target_lang}_xcomet_error_spans'] = (
                        json.dumps(xcomet_score.get('error_spans', []), ensure_ascii=False)
                    )
                    
                    # Count errors by severity
                    error_spans = xcomet_score.get('error_spans', [])
                    minor_count = sum(1 for e in error_spans if e.get('severity') == 'minor')
                    major_count = sum(1 for e in error_spans if e.get('severity') == 'major')
                    critical_count = sum(1 for e in error_spans if e.get('severity') == 'critical')
                    
                    record[f'annotation_{target_lang}_xcomet_minor_errors'] = minor_count
                    record[f'annotation_{target_lang}_xcomet_major_errors'] = major_count
                    record[f'annotation_{target_lang}_xcomet_critical_errors'] = critical_count
                else:
                    record[f'annotation_{target_lang}_xcomet_sentence_scores'] = None
                    record[f'annotation_{target_lang}_xcomet_system_score'] = None
                    record[f'annotation_{target_lang}_xcomet_error_spans'] = None
                    record[f'annotation_{target_lang}_xcomet_minor_errors'] = None
                    record[f'annotation_{target_lang}_xcomet_major_errors'] = None
                    record[f'annotation_{target_lang}_xcomet_critical_errors'] = None
                
                processed_count += 1
                record_updated = True
                
                if xcomet_score and xcomet_score.get('score') is not None:
                    self.logger.info(
                        f"  {record['unique_id']} ({source_lang}->{target_lang}): "
                        f"{xcomet_score['score']:.4f} ({xcomet_score['translation_quality_score']})"
                    )
            
            # Save incrementally after processing each record
            if record_updated and self.output_dir:
                PipelineUtils.save_incremental(record, self.output_dir)
                # Delay between records
                #self.logger.info("  Waiting 15 seconds before next record...")
                #time.sleep(15)
        
        self.logger.info(f"\nXCOMET processing completed: {processed_count} translations scored, {skipped_count} skipped")
        return all_records
    
    def batch_process_xtower(self, all_records: List[Dict]) -> List[Dict]:
        """
        Process all records with xTower in batch.
        
        Args:
            all_records: List of records with XCOMET scores
            
        Returns:
            Updated list of records with xTower analysis
        """
        if not self.xtower:
            self.logger.warning("xTower analyzer not available")
            return all_records
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("BATCH PROCESSING WITH XTOWER")
        self.logger.info("=" * 80)
        
        processed_count = 0
        skipped_count = 0
        
        for record in all_records:
            source_lang = record.get('source_language')
            if not source_lang:
                continue
            
            gold_annotation = record.get(f'annotation_{source_lang}')
            if not gold_annotation or gold_annotation == 'NA':
                continue
            
            target_langs = self.LANG_CONFIGS[source_lang]['targets']
            record_updated = False
            
            for target_lang in target_langs:
                translation = record.get(f'annotation_{target_lang}')
                is_gold = record.get(f'annotation_{target_lang}_is_gold', False)
                
                # Skip if NA, gold, or no translation
                if not translation or translation == 'NA' or is_gold:
                    continue
                
                # Skip if already processed (has xtower output)
                if record.get(f'annotation_{target_lang}_xtower_output') is not None:
                    skipped_count += 1
                    self.logger.info(
                        f"  Skipping {record['unique_id']} ({source_lang}->{target_lang}): "
                        f"already has xTower output"
                    )
                    continue
                
                # Reconstruct XCOMET data from separate fields
                sentence_score = record.get(f'annotation_{target_lang}_xcomet_sentence_scores')
                system_score = record.get(f'annotation_{target_lang}_xcomet_system_score')
                error_spans_str = record.get(f'annotation_{target_lang}_xcomet_error_spans')
                
                # Skip if no XCOMET score
                if sentence_score is None:
                    self.logger.debug(
                        f"  Skipping {record['unique_id']} ({source_lang}->{target_lang}): "
                        f"No XCOMET score available"
                    )
                    continue
                
                # Parse error spans
                error_spans = []
                if error_spans_str:
                    try:
                        error_spans = json.loads(error_spans_str)
                    except Exception as e:
                        self.logger.warning(f"Failed to parse error_spans: {e}")
                        error_spans = []
                
                # Reconstruct XCOMET data dictionary
                xcomet_data = {
                    'score': sentence_score,
                    'system_score': system_score,
                    'error_spans': error_spans,
                    'translation_quality_score': BaseXCOMETScorer._interpret_score(sentence_score),
                    'translation_quality_analysis': BaseXCOMETScorer._format_quality_analysis(translation, error_spans)
                }
                
                self.logger.debug(
                    f"  Processing {record['unique_id']} ({source_lang}->{target_lang}) "
                    f"with XCOMET score: {sentence_score:.4f}"
                )
                
                # Analyze with xTower
                errors = self.xtower.analyze_translation(
                    gold_annotation, translation, source_lang, target_lang, xcomet_data
                )
                
                if errors:
                    # Save the prompt
                    record[f'annotation_{target_lang}_xtower_prompt'] = (
                        errors.get('xtower_prompt')
                    )
                    
                    # Save the explanations in readable format
                    record[f'annotation_{target_lang}_xtower_explanations'] = (
                        errors.get('explanations')
                    )
                    
                    # Save the corrected translation
                    record[f'annotation_{target_lang}_xtower_corrected_translation'] = (
                        errors.get('corrected_translation')
                    )
                    
                    # Save the full output
                    record[f'annotation_{target_lang}_xtower_output'] = (
                        errors.get('xtower_full_output')
                    )
                    
                    processed_count += 1
                    record_updated = True
                    has_errors = errors.get('has_errors', False)
                    num_errors = len(errors.get('errors', []))
                    self.logger.info(
                        f"  {record['unique_id']} ({source_lang}->{target_lang}): "
                        f"{num_errors} errors detected, has_correction={bool(errors.get('corrected_translation'))}"
                    )
            
            # Save incrementally after processing each record
            if record_updated and self.output_dir:
                PipelineUtils.save_incremental(record, self.output_dir)
                # Delay between records
                #self.logger.info("  Waiting 15 seconds before next record...")
                #time.sleep(15)
        
        self.logger.info(f"\nxTower processing completed: {processed_count} translations analyzed, {skipped_count} skipped")
        return all_records
    
    def process_language_folder(
        self,
        lang_code: str,
        datasets_dir: Path,
        output_dir: Path,
        save_interval: int = 10
    ) -> List[Dict]:
        """
        Process all data files in a language folder.
        
        Args:
            lang_code: Language code ('it', 'en', 'es')
            datasets_dir: Path to datasets directory
            output_dir: Path to output directory
            save_interval: Save intermediate results every N records
            
        Returns:
            List of processed records
        """
        folder_path = datasets_dir / self.LANG_CONFIGS[lang_code]['folder']
        
        if not folder_path.exists():
            self.logger.warning(f"Folder not found: {folder_path}")
            return []
        
        # Find all data files
        csv_files = list(folder_path.glob("*.csv"))
        xlsx_files = list(folder_path.glob("*.xlsx"))
        all_files = csv_files + xlsx_files
        
        if not all_files:
            self.logger.warning(f"No CSV or Excel files found in {folder_path}")
            return []
        
        self.logger.info(
            f"\nProcessing {lang_code.upper()} folder: {len(all_files)} file(s) found"
        )
        self.logger.info(f"  CSV files: {len(csv_files)}, Excel files: {len(xlsx_files)}")
        
        all_records = []
        
        for data_file in all_files:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"File: {data_file.name}")
            self.logger.info(f"{'='*60}")
            
            df = PipelineUtils.load_data_file(data_file)
            if df.empty:
                continue
                
            self.logger.info(f"Loaded {len(df)} records")
            
            # Apply test mode limit if enabled
            records_to_process = len(df)
            if self.test_mode and self.test_records_limit:
                records_to_process = min(records_to_process, self.test_records_limit - len(all_records))
                if records_to_process <= 0:
                    self.logger.info(f"Test records limit ({self.test_records_limit}) reached. Skipping remaining files.")
                    break
                self.logger.info(f"[TEST MODE] Processing only {records_to_process} records from this file")
            
            for idx, row in df.iterrows():
                # Check test mode limit
                if self.test_mode and self.test_records_limit and len(all_records) >= self.test_records_limit:
                    self.logger.info(f"[TEST MODE] Reached limit of {self.test_records_limit} records. Stopping.")
                    break
                    
                try:
                    record = self.process_record(row, lang_code, data_file.name)
                    all_records.append(record)
                    
                    # Save intermediate results
                    if (idx + 1) % save_interval == 0:
                        PipelineUtils.save_intermediate_results(
                            all_records, lang_code, output_dir
                        )
                        
                except Exception as e:
                    self.logger.error(f"Error processing record {idx}: {e}")
                    continue
            
            # Break outer loop if test limit reached
            if self.test_mode and self.test_records_limit and len(all_records) >= self.test_records_limit:
                break
        
        return all_records
