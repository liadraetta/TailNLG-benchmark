"""
Main script for running the translation and post-editing pipeline.
"""
import os
import sys
import json
import logging
import argparse
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from src import (
    DeepLTranslator,
    XCOMETScorer,
    XTowerAnalyzer,
    MockXCOMETScorer,
    MockXTowerAnalyzer,
    TranslationPipeline,
    PipelineUtils
)


def setup_logging(output_dir: Path) -> None:
    """
    Setup logging configuration.
    
    Args:
        output_dir: Directory for log file (the run-specific directory)
    """
    log_filename = output_dir / "pipeline.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True
    )


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Run translation and post-editing pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Run complete pipeline from start
            python main.py
            
            # Resume from XCOMET step using previous run
            python main.py --start-from xcomet --resume-from run_20241105_143000
            
            # Resume from xTower step
            python main.py --start-from xtower --resume-from run_20241105_143000
            
            # Only generate summary from existing run
            python main.py --start-from summary --resume-from run_20241105_143000
        """
    )
    
    parser.add_argument(
        '--start-from',
        type=str,
        choices=['translation', 'xcomet', 'xtower', 'summary'],
        default='translation',
        help='Step to start from (default: translation)'
    )
    
    parser.add_argument(
        '--resume-from',
        type=str,
        help='Previous run directory to resume from (e.g., run_20241105_143000). Required when starting from xcomet, xtower, or summary.'
    )
    
    parser.add_argument(
        '--test-mode',
        action='store_true',
        default=True,
        help='Use mock models for testing (default: True)'
    )
    
    parser.add_argument(
        '--no-test-mode',
        action='store_true',
        help='Use real models (GPU required)'
    )
    
    parser.add_argument(
        '--test-limit',
        type=int,
        default=1,
        help='Number of records to process in test mode (default: 1)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.start_from in ['xcomet', 'xtower', 'summary'] and not args.resume_from:
        parser.error(f"--resume-from is required when starting from '{args.start_from}'")
    
    # Handle test mode
    if args.no_test_mode:
        args.test_mode = False
    
    return args


def load_previous_data(resume_dir: Path, logger) -> list:
    """
    Load data from a previous run.
    
    Args:
        resume_dir: Directory of the previous run
        logger: Logger instance
        
    Returns:
        List of records from previous run
    """
    json_path = resume_dir / "unified_corpus.json"
    
    if not json_path.exists():
        logger.error(f"Cannot find unified_corpus.json in {resume_dir}")
        sys.exit(1)
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            all_records = json.load(f)
        logger.info(f"Loaded {len(all_records)} records from {resume_dir}")
        return all_records
    except Exception as e:
        logger.error(f"Failed to load data from {resume_dir}: {e}")
        sys.exit(1)


def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Load environment variables
    load_dotenv()
    
    # Configuration
    SCRIPT_DIR = Path(__file__).parent.absolute()
    DEEPL_AUTH_KEY = os.getenv("DEEPL_AUTH_KEY")
    DATASETS_DIR = SCRIPT_DIR / "datasets"
    BASE_OUTPUT_DIR = SCRIPT_DIR / "output"
    
    # Create run-specific output directory with timestamp (only if not resuming)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.resume_from:
        # Reuse existing directory
        RUN_OUTPUT_DIR = BASE_OUTPUT_DIR / args.resume_from
        if not RUN_OUTPUT_DIR.exists():
            print(f"ERROR: Resume directory does not exist: {RUN_OUTPUT_DIR}")
            return
    else:
        # Create new run directory
        RUN_OUTPUT_DIR = BASE_OUTPUT_DIR / f"run_{timestamp}"
        os.makedirs(RUN_OUTPUT_DIR, exist_ok=True)
    
    # Feature flags
    TEST_MODE = args.test_mode
    TEST_RECORDS_LIMIT = args.test_limit if TEST_MODE else None
    LANG_CODES = ['it', 'en', 'es']
 
    # Setup logging in run directory
    setup_logging(RUN_OUTPUT_DIR)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("MULTILINGUAL TRANSLATION AND POST-EDITING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Run output directory: {RUN_OUTPUT_DIR}")
    logger.info(f"Starting from step: {args.start_from}")
    
    # Log resume information
    if args.resume_from:
        resume_dir = BASE_OUTPUT_DIR / args.resume_from
        if not resume_dir.exists():
            logger.error(f"Resume directory not found: {resume_dir}")
            sys.exit(1)
        logger.info(f"Resuming from run: {args.resume_from}")
        logger.info(f"Resume directory: {resume_dir}")
    
    logger.info(f"Test Mode: {TEST_MODE}")
    if TEST_MODE:
        logger.info(f"Test Records Limit: {TEST_RECORDS_LIMIT}")
    logger.info("")
    
    # Initialize variables
    all_records = []
    deepl_translator = None
    pipeline = None
    
    # Determine which output directory to use
    if args.resume_from:
        # Resume in the same directory
        ACTUAL_OUTPUT_DIR = BASE_OUTPUT_DIR / args.resume_from
        logger.info(f"Resuming in directory: {ACTUAL_OUTPUT_DIR}")
    else:
        # Use new run directory
        ACTUAL_OUTPUT_DIR = RUN_OUTPUT_DIR
    
    # PHASE 1: Process translations (or load previous data)
    if args.start_from == 'translation':
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 1: PROCESSING TRANSLATIONS")
        logger.info("=" * 80)
        
        # Initialize DeepL translator
        logger.info("Initializing DeepL translator...")
        deepl_translator = DeepLTranslator(DEEPL_AUTH_KEY)
        if not deepl_translator.initialize():
            logger.error("Failed to initialize DeepL translator. Exiting.")
            return
        
        # Initialize pipeline
        pipeline = TranslationPipeline(
            deepl_translator=deepl_translator,
            xcomet_scorer=None,
            xtower_analyzer=None,
            test_mode=TEST_MODE,
            test_records_limit=TEST_RECORDS_LIMIT,
            output_dir=ACTUAL_OUTPUT_DIR
        )
        
        for lang_code in LANG_CODES:
            try:
                logger.info(f"\n{'='*80}")
                logger.info(f"Processing {lang_code.upper()} folder")
                logger.info(f"{'='*80}")
                
                records = pipeline.process_language_folder(
                    lang_code=lang_code,
                    datasets_dir=DATASETS_DIR,
                    output_dir=ACTUAL_OUTPUT_DIR,
                    save_interval=10
                )
                
                all_records.extend(records)
                logger.info(f"\n{lang_code.upper()} processing complete: {len(records)} records")
                
            except Exception as e:
                logger.error(f"Error processing {lang_code} folder: {e}", exc_info=True)
                continue
        
        if not all_records:
            logger.warning("No records processed. Check input folders and files.")
            return
        
        # Show DeepL usage after translations
        if deepl_translator:
            usage = deepl_translator.get_usage()
            if usage:
                logger.info(f"\nDeepL API Usage (after translations):")
                logger.info(f"  Characters used: {usage['character_count']:,} / {usage['character_limit']:,}")
                logger.info(f"  Percentage: {usage['percentage_used']:.2f}%")
    else:
        # Load data from previous run
        logger.info("\n" + "=" * 80)
        logger.info(f"LOADING DATA FROM PREVIOUS RUN: {args.resume_from}")
        logger.info("=" * 80)
        
        resume_dir = BASE_OUTPUT_DIR / args.resume_from
        all_records = load_previous_data(resume_dir, logger)
        
        # Initialize pipeline without translator (not needed)
        pipeline = TranslationPipeline(
            deepl_translator=None,
            xcomet_scorer=None,
            xtower_analyzer=None,
            test_mode=TEST_MODE,
            test_records_limit=TEST_RECORDS_LIMIT,
            output_dir=ACTUAL_OUTPUT_DIR
        )
    

    # PHASE 2A: XCOMET scoring
    if args.start_from in ['translation', 'xcomet']:
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2A: LOADING XCOMET AND SCORING TRANSLATIONS")
        logger.info("=" * 80)
        
        # Use mock or real model based on TEST_MODE
        if TEST_MODE:
            logger.info("Using MockXCOMETScorer (test mode)")
            xcomet_scorer = MockXCOMETScorer(model_name="Unbabel/XCOMET-XL", use_gpu=False)
        else:
            logger.info("Using real XCOMETScorer")
            xcomet_scorer = XCOMETScorer(model_name="Unbabel/XCOMET-XL", use_gpu=True)
        
        if xcomet_scorer.load_model():
            pipeline.xcomet = xcomet_scorer
            all_records = pipeline.batch_process_xcomet(all_records)
            
            # Cleanup XCOMET
            logger.info("\nCleaning up XCOMET model from memory...")
            xcomet_scorer.cleanup()
            pipeline.xcomet = None
            del xcomet_scorer
        else:
            logger.warning("Failed to load XCOMET model. Skipping quality scoring.")
    else:
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2A: SKIPPING XCOMET (already computed in previous run)")
        logger.info("=" * 80)

    # PHASE 2B: xTower analysis
    if args.start_from in ['translation', 'xcomet', 'xtower']:
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2B: LOADING XTOWER AND ANALYZING ERRORS")
        logger.info("=" * 80)
        
        # Use mock or real model based on TEST_MODE
        if TEST_MODE:
            logger.info("Using MockXTowerAnalyzer (test mode)")
            xtower_analyzer = MockXTowerAnalyzer(model_name="sardinelab/xTower13B")
        else:
            logger.info("Using real XTowerAnalyzer")
            xtower_analyzer = XTowerAnalyzer(model_name="sardinelab/xTower13B")
        
        if xtower_analyzer.load_model():
            pipeline.xtower = xtower_analyzer
            all_records = pipeline.batch_process_xtower(all_records)
            
            # Cleanup xTower
            logger.info("\nCleaning up xTower model from memory...")
            xtower_analyzer.cleanup()
            pipeline.xtower = None
            del xtower_analyzer
        else:
            logger.warning("Failed to load xTower model. Skipping error analysis.")
    else:
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2B: SKIPPING XTOWER (already computed in previous run)")
        logger.info("=" * 80)
    
    # PHASE 3: Save final results
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3: SAVING FINAL CORPUS")
    logger.info("=" * 80)
    
    csv_path, json_path = PipelineUtils.save_final_corpus(all_records, ACTUAL_OUTPUT_DIR)
    
    # Create summary report (reuse the same timestamp from directory name)
    run_timestamp = ACTUAL_OUTPUT_DIR.name.replace('run_', '') if args.resume_from else timestamp
    PipelineUtils.create_summary_report(all_records, ACTUAL_OUTPUT_DIR, run_timestamp)
    
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"Total records: {len(all_records)}")
    logger.info(f"Output directory: {RUN_OUTPUT_DIR}")
    logger.info(f"CSV output: {csv_path}")
    logger.info(f"JSON output: {json_path}")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    logger.info("\nPipeline execution completed.")


if __name__ == "__main__":
    main()
