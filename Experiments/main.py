from src.webnlg_loader import WebNLGLoader
from src.config import (
    MODELS, USE_QUANTIZATION, OUTPUT_DIR, 
    NUM_GENERATIONS_PER_TRIPLE, GENERATION_TEMPERATURE,
    ONE_SHOT_EXAMPLES, FEW_SHOT_EXAMPLES, NEGATIVE_FEW_SHOT_EXAMPLES,
    INFERENCE_BATCH_SIZE
)
from src.llm_handler import LLMHandler
from src.evaluator import VerbalizationEvaluator
from pathlib import Path
import pandas as pd
import json
import re

def mainTestWebNLGLoader():
    print("Loading merged dataset for all languages...")
    merged_data = WebNLGLoader.load_webnlg_dataset(method="merged")

    print(f"Merged - Train samples: {len(merged_data['train'])}")
    print(f"Merged - Dev samples: {len(merged_data['dev'])}")
    print(f"Merged - Test samples: {len(merged_data['test'])}")
    print()
    
    # Esempio 2: Carica dataset separati per lingua
    print("Loading separated datasets...")
    separated_data = WebNLGLoader.load_webnlg_dataset(languages=['en', 'es', 'it'], method="separated")
    for lang in separated_data:
        print(f"{lang.upper()} - Train: {len(separated_data[lang]['train'])}, "
              f"Dev: {len(separated_data[lang]['dev'])}, "
              f"Test: {len(separated_data[lang]['test'])}")
    print()
    
    # Esempio 3: Carica solo una lingua
    print("Loading only English dataset...")
    en_data = WebNLGLoader.load_webnlg_dataset(languages=['en'], method="merged")
    print(f"English - Train samples: {len(en_data['train'])}")
    
    # Mostra un esempio di dato
    if en_data['test']:
        print("\nSample entry:")
        sample = en_data['test'][1000]
        for key, value in sample.items():
            print(f"  {key}: {value}")

    longtail_webnlg_dataset = WebNLGLoader.load_longtail_webnlg_dataset(languages=['en', 'es', 'it'])
    print(f"LongTail - Total samples: {len(longtail_webnlg_dataset)}")

def generateTailNLG():
    # Load merged WebNLG dataset for fine-tuning
    print("Loading merged WebNLG dataset for all languages...")
    merged_webnlg_dataset = WebNLGLoader.load_webnlg_dataset(languages=['en', 'es', 'it'], method="merged")
    #merged_webnlg_dataset = {
    #    'train': merged_webnlg_dataset['train'][:100],
    #    'dev': merged_webnlg_dataset['dev'][:50],
    #    'test': merged_webnlg_dataset['test'][:50]
    #}

    print(f"Merged - Train samples: {len(merged_webnlg_dataset['train'])}")
    print(f"Merged - Dev samples: {len(merged_webnlg_dataset['dev'])}")
    print(f"Merged - Test samples: {len(merged_webnlg_dataset['test'])}")

    # Load LongTail WebNLG dataset for zero/one/few-shot
    longtail_webnlg_dataset = WebNLGLoader.load_longtail_webnlg_dataset(languages=['en', 'es', 'it'])
    print(f"LongTail - Total samples: {len(longtail_webnlg_dataset)}")
    print("\nSample entry:")
    sample = longtail_webnlg_dataset[1000]
    for key, value in sample.items():
        print(f"  {key}: {value}")

    #longtail_webnlg_dataset = longtail_webnlg_dataset[:15]  # Use a subset for testing

    for model_name in MODELS:
        print(f"\nTesting model: {model_name}")
        
        handler = LLMHandler(
            model_name=model_name,
            merged_webnlg_dataset=merged_webnlg_dataset,
            longtail_webnlg_dataset=longtail_webnlg_dataset,
            use_quantization=USE_QUANTIZATION,
            output_dir=OUTPUT_DIR,
            num_generations=NUM_GENERATIONS_PER_TRIPLE,
            temperature=GENERATION_TEMPERATURE,
            inference_batch_size=INFERENCE_BATCH_SIZE,
            test_set='TailNLG'
        )
        
        # Run zero-shot verbalization
        print("\nRunning zero-shot verbalization...")
        handler.zero_shot()
        print("Zero-shot verbalization completed.")
        
        # run one-shot verbalization
        print("\nRunning one-shot verbalization...")
        handler.one_shot(ONE_SHOT_EXAMPLES)
        print("One-shot verbalization completed.")

        # run few-shot verbalization
        print("\nRunning few-shot verbalization...")
        handler.few_shot(FEW_SHOT_EXAMPLES)
        print("Few-shot verbalization completed.")

        # run contrastive few-shot verbalization
        print("\nRunning contrastive few-shot verbalization...")
        handler.contrastive_few_shot(FEW_SHOT_EXAMPLES, NEGATIVE_FEW_SHOT_EXAMPLES)
        print("Contrastive few-shot verbalization completed.")

        # run fine-tuning
        print("\nRunning fine-tuning...")
        output_path = handler.fine_tuning(
            train_split='train',
            val_split='dev',
        )

        print(f"Fine-tuned model saved to: {output_path}")
        
        # Destroy the training model before loading the fine-tuned version
        print("\nUnloading training model...")
        handler.destroy_model()
        
        # Reload handler with the fine-tuned model
        print("Loading fine-tuned model for inference...")
        handler = LLMHandler(
            model_name=model_name,
            merged_webnlg_dataset=merged_webnlg_dataset,
            longtail_webnlg_dataset=longtail_webnlg_dataset,
            use_quantization=USE_QUANTIZATION,
            output_dir=OUTPUT_DIR,
            num_generations=NUM_GENERATIONS_PER_TRIPLE,
            temperature=GENERATION_TEMPERATURE,
            inference_batch_size=INFERENCE_BATCH_SIZE,
            test_set='TailNLG'
        )
        handler.load_fine_tuned_model(output_path)
        
        print("\nRunning zero-shot with fine-tuned model...")
        handler.zero_shot()
        print("Fine-tuned model zero-shot verbalization completed.")
    
        # Clean up
        handler.destroy_model()

def generateWebNLG():
    # Load merged WebNLG dataset for fine-tuning
    print("Loading merged WebNLG dataset for all languages...")
    merged_webnlg_dataset = WebNLGLoader.load_webnlg_dataset(languages=['en', 'es', 'it'], method="merged")

    print(f"Merged - Train samples: {len(merged_webnlg_dataset['train'])}")
    print(f"Merged - Dev samples: {len(merged_webnlg_dataset['dev'])}")
    print(f"Merged - Test samples: {len(merged_webnlg_dataset['test'])}")


    for model_name in MODELS:
        print(f"\nTesting model: {model_name}")
        
        handler = LLMHandler(
            model_name=model_name,
            merged_webnlg_dataset=merged_webnlg_dataset,
            use_quantization=USE_QUANTIZATION,
            output_dir=OUTPUT_DIR,
            num_generations=NUM_GENERATIONS_PER_TRIPLE,
            temperature=GENERATION_TEMPERATURE,
            inference_batch_size=INFERENCE_BATCH_SIZE,
            test_set="WebNLG"
        )

        # Run zero-shot verbalization
        print("\nRunning zero-shot verbalization...")
        handler.zero_shot()
        print("Zero-shot verbalization completed.")
        
        # Destroy the training model before loading the fine-tuned version
        print("\nUnloading training model...")
        handler.destroy_model()

def postediting(test_set='TailNLG'):
    """
    Post-edit verbalizations to clean common formatting issues.
    
    Creates a new 'prediction_pe' (post-edited) field in each JSON entry
    while preserving the original 'prediction' field.
    
    Cleaning rules:
    1. If verbalization starts with '[', extract only content within square brackets
    2. If verbalization contains '\\n\\n', take only content after the last occurrence
    3. Remove prefix phrases: "La verbalizzazione finale è:", "La verbalización final es:", "The final verbalization is:"
    4. Remove surrounding brackets if text starts with '[' and ends with ']'
    
    Args:
        test_set: 'TailNLG' or 'WebNLG' - determines which files to process
    """    
    # Determine file prefix based on test set
    test_set_prefix = "tail_" if test_set == "TailNLG" else "web_"
    
    def clean_verbalization(text: str) -> str:
        """
        Clean a single verbalization according to the rules.
        
        Args:
            text: Original verbalization text
            
        Returns:
            Cleaned verbalization text
        """
        if not isinstance(text, str):
            return text
        
        original = text
        changed = False
        
        # Rule 1: If starts with '[', extract content within square brackets
        if text.startswith('['):
            # Find matching closing bracket
            match = re.match(r'^\[(.*?)\]', text, re.DOTALL)
            if match:
                text = match.group(1).strip()
                changed = True
                print(f"    [Brackets] Cleaned: '{original[:50]}...' -> '{text[:50]}...'")
        
        # Rule 2: If contains '\n\n', take content after last occurrence
        if '\n\n' in text:
            parts = text.split('\n\n')
            text = parts[-1].strip()
            changed = True
            print(f"    [Newlines] Cleaned: '{original[:50]}...' -> '{text[:50]}...'")
        
        # Rule 3: Remove prefix phrases in multiple languages
        prefix_patterns = [
            r'^La verbalizzazione finale è:\s*',  # Italian
            r'^La verbalización final es:\s*',     # Spanish
            r'^The final verbalization is:\s*'     # English
        ]
        
        for pattern in prefix_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                old_text = text
                text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
                changed = True
                print(f"    [Prefix] Cleaned: '{old_text[:50]}...' -> '{text[:50]}...'")
                break  # Only match one pattern
        
        # rimuovi [ e ]
        if text.startswith('[') and text.endswith(']'):
            old_text = text
            text = text[1:-1].strip()
            changed = True
            print(f"    [Brackets Removal] Cleaned: '{old_text[:50]}...' -> '{text[:50]}...'")
        return text
    
    for model_name in MODELS:
        print(f"\n{'='*100}")
        print(f"Post-editing verbalizations for model: {model_name}")
        print(f"Test set: {test_set}")
        print('='*100)
        
        output_dir = Path(OUTPUT_DIR) / model_name.replace('/', '_')
        model_safe_name = model_name.replace('/', '_')
        
        # Find all relevant files
        base_methods = ['zero_shot', 'one_shot', 'few_shot', 'contrastive_few_shot', 'fine_tuned_zero_shot']
        all_methods = base_methods + ['fine_tuned_zero_shot']
        
        for method in all_methods:
            file_path = output_dir / f"{test_set_prefix}{method}_{model_safe_name}.json"
            
            if not file_path.exists():
                print(f"\nSkipping {method}: file not found")
                continue
            
            print(f"\n{'-'*100}")
            print(f"Processing: {file_path.name}")
            print('-'*100)
            
            try:
                # Load data
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Clean predictions and create prediction_pe field
                cleaned_count = 0
                bracket_count = 0
                newline_count = 0
                prefix_count = 0
                bracket_removal_count = 0
                
                for entry in data:
                    if 'prediction' in entry:
                        original = entry['prediction']
                        cleaned = clean_verbalization(original)
                        
                        # Always create prediction_pe field (even if unchanged)
                        entry['prediction_pe'] = cleaned
                        
                        # Count changes
                        if cleaned != original:
                            cleaned_count += 1
                            
                            if original.startswith('[') and ']' in original[:100]:  # Bracket extraction
                                bracket_count += 1
                            if '\n\n' in original:
                                newline_count += 1
                            # Check for prefix patterns
                            if (re.match(r'^La verbalizzazione finale è:\s*', original, re.IGNORECASE) or
                                re.match(r'^La verbalización final es:\s*', original, re.IGNORECASE) or
                                re.match(r'^The final verbalization is:\s*', original, re.IGNORECASE)):
                                prefix_count += 1
                            # Check for surrounding brackets removal
                            if original.startswith('[') and original.endswith(']') and cleaned == original[1:-1].strip():
                                bracket_removal_count += 1
                
                # Save data with new prediction_pe field
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                print(f"\n  ✓ Post-editing completed:")
                print(f"    - Total entries: {len(data)}")
                print(f"    - Entries with changes: {cleaned_count}")
                print(f"    - Bracket extraction: {bracket_count}")
                print(f"    - Newline fixes: {newline_count}")
                print(f"    - Prefix removal: {prefix_count}")
                print(f"    - Bracket removal: {bracket_removal_count}")
                print(f"    - New field 'prediction_pe' added to all entries")
                print(f"    - File updated: {file_path.name}")
                
            except Exception as e:
                print(f"  ✗ Error processing {file_path.name}: {e}")
                continue
        
        print(f"\n{'='*100}")
        print(f"Post-editing completed for {model_name}")
        print('='*100)


def evaluate(test_set='TailNLG', metrics=None):
    """
    Evaluate generated verbalizations.
    
    Args:
        test_set: 'TailNLG' or 'WebNLG' - determines which files to evaluate
    """
    # Determine file prefix based on test set
    test_set_prefix = "tail_" if test_set == "TailNLG" else "web_"
    
    for model_name in MODELS:
        print(f"\nEvaluating model: {model_name}")
        print(f"Test set: {test_set}")
        print("\n" + "="*60)
        print("AUTOMATIC EVALUATION")
        print("="*60)
        
        # Prepare file paths for evaluation
        output_dir = Path(OUTPUT_DIR) / model_name.replace('/', '_')
        model_safe_name = model_name.replace('/', '_')
        
        # Base model files with test set prefix
        print(f"\nEvaluating {test_set} model generations...")
        base_methods = ['zero_shot'] #['zero_shot', 'one_shot', 'few_shot', 'contrastive_few_shot', 'fine_tuned_zero_shot']
        all_files = []
        for method in base_methods:
            # New naming convention: [test_set_prefix]method_modelname.json
            file_path = output_dir / f"{test_set_prefix}{method}_{model_safe_name}.json"
            if file_path.exists():
                all_files.append(file_path)
                print(f"  Found: {file_path.name}")
            
        if all_files:
            evaluator = VerbalizationEvaluator(
                metrics=metrics,
            )
            results = evaluator.evaluate_multiple_files(
                file_paths=all_files,
                prediction_key='prediction_pe',
                skip_existing=True
            )

            #result = evaluator.calculate_perplexity_for_multiple_files(
            #    file_paths=all_files,
            #    prediction_key='prediction_pe',
            #    skip_existing=False 
            #)
            print("\n✓ Evaluation completed.")

            # Save results with test set in filename
            #(output_dir / "eval").mkdir(parents=True, exist_ok=True)
            #results_path = output_dir / f"eval/evaluation_results_{test_set.lower()}_{model_safe_name}.json"
            #summary_path = output_dir / f"eval/evaluation_summary_{test_set.lower()}_{model_safe_name}.txt"
            
            #VerbalizationEvaluator.save_results(results, results_path)
            #summary = VerbalizationEvaluator.summarize_results(results, summary_path)
            #print("\n" + summary)
        else:
            print(f"No {test_set} files found for evaluation")
            print(f"Expected files with prefix: {test_set_prefix}*_{model_safe_name}.json")


def analyze_statistics(test_set='TailNLG'):
    """
    Generate detailed statistics by metadata (language, type, subtype, category).
    
    Args:
        test_set: 'TailNLG' or 'WebNLG' - determines which files to analyze
    """
    
    # Determine file prefix based on test set
    test_set_prefix = "tail_" if test_set == "TailNLG" else "web_"
    
    for model_name in MODELS:
        print(f"\n{'='*100}")
        print(f"Analyzing statistics for model: {model_name}")
        print(f"Test set: {test_set}")
        print('='*100)
        
        output_dir = Path(OUTPUT_DIR) / model_name.replace('/', '_')
        model_safe_name = model_name.replace('/', '_')
        
        # Find all relevant files
        all_methods = ['zero_shot'] #, 'one_shot', 'few_shot', 'contrastive_few_shot', 'fine_tuned_zero_shot']
        
        for method in all_methods:
            file_path = output_dir / f"{test_set_prefix}{method}_{model_safe_name}.json"
            
            if not file_path.exists():
                print(f"\nSkipping {method}: file not found")
                continue
            
            print(f"\n{'-'*100}")
            print(f"Method: {method}")
            print(f"File: {file_path.name}")
            print('-'*100)
            
            try:
                # Compute aggregate statistics (overall + by language)
                if test_set == 'TailNLG':
                    # For TailNLG, compute statistics only for long_tail entries
                    stats = VerbalizationEvaluator.compute_statistics_longtail_only(
                        file_path,
                    )
                else:
                    # For WebNLG, compute statistics for all entries
                    stats = VerbalizationEvaluator.compute_statistics(
                        file_path,
                    )
                
                # Format and print aggregate tables
                tables = VerbalizationEvaluator.format_aggregate_statistics_table(stats)
                print(tables)
                
                # Save to text file
                stats_dir = output_dir / "statistics"
                stats_dir.mkdir(parents=True, exist_ok=True)
                
                output_txt = stats_dir / f"aggregate_statistics_{method}_{test_set.lower()}_{model_safe_name}.txt"
                VerbalizationEvaluator.format_aggregate_statistics_table(stats, output_path=output_txt)

                if test_set == 'TailNLG':
                    # Compute statistics by metadata
                    stats = VerbalizationEvaluator.compute_statistics_by_metadata(
                        file_path,
                    )
                    
                    # Format and print tables
                    tables = VerbalizationEvaluator.format_statistics_tables(stats)
                    print(tables)
                    
                    # Save to text file
                    stats_dir = output_dir / "statistics"
                    stats_dir.mkdir(parents=True, exist_ok=True)
                    
                    output_txt = stats_dir / f"statistics_{method}_{test_set.lower()}_{model_safe_name}.txt"
                    VerbalizationEvaluator.format_statistics_tables(stats, output_path=output_txt)
                    
                    # Save to JSON for further processing
                    #output_json = stats_dir / f"statistics_{method}_{test_set.lower()}_{model_safe_name}.json"
                    #with open(output_json, 'w', encoding='utf-8') as f:
                    #    json.dump(stats, f, indent=2, ensure_ascii=False)
                    
                    print(f"\n✓ Statistics saved to:")
                    print(f"  - {output_txt}")
                    #print(f"  - {output_json}")
                    
                    # ===========================
                    # NEW: Compute statistics by quality (silver vs gold)
                    # ===========================
                    print(f"\n{'-'*100}")
                    print("Computing statistics by QUALITY (silver vs gold)...")
                    print('-'*100)
                    
                    quality_json = stats_dir / f"quality_statistics_{method}_{test_set.lower()}_{model_safe_name}.json"
                    quality_md = stats_dir / f"quality_tables_{method}_{test_set.lower()}_{model_safe_name}.md"
                    
                    quality_stats = VerbalizationEvaluator.compute_statistics_by_quality(
                        file_path=file_path,
                        output_md_path=quality_md
                    )
                    
                    print(f"\n✓ Quality statistics saved to:")
                    print(f"  - {quality_json}")
                    print(f"  - {quality_md}")
                
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
                continue
        
        print(f"\n{'='*100}")
        print(f"Statistics analysis completed for {model_name}")
        print('='*100)


  
if __name__ == "__main__":
    # Test WebNLG Loader
    #mainTestWebNLGLoader()
    
    # Generate verbalizations for TailNLG
    #generateTailNLG()
    
    # Generate verbalizations for WebNLG
    #generateWebNLG()
    
    # Post-edit verbalizations for TailNLG (clean formatting issues)
    #postediting(test_set='TailNLG')
    
    # Post-edit verbalizations for WebNLG
    #postediting(test_set='WebNLG')
    
    # Evaluate TailNLG results
    #evaluate(test_set='TailNLG', metrics=['bertscore_rescaled', 'bertscore'])
    
    # Evaluate WebNLG results
    #evaluate(test_set='WebNLG',  metrics=['bertscore_rescaled', 'bleu', 'chrf', 'rouge1', 'rouge2', 'rougeL'])
    
    # Analyze statistics for TailNLG
    analyze_statistics(test_set='TailNLG')
    
    # Analyze statistics for WebNLG
    analyze_statistics(test_set='WebNLG')