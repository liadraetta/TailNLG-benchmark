from src.config import (
    MODELS, OUTPUT_DIR
)
from src.evaluator import VerbalizationEvaluator
from pathlib import Path
import json

def analyze_metric_correlations(test_set='TailNLG'):
    """
    Analyze correlations between different evaluation metrics.
    
    Computes Pearson and Spearman correlations to understand how different metrics
    relate to each other. This helps identify:
    - Which metrics are redundant (high correlation)
    - Which metrics capture different aspects of quality (low correlation)
    
    Args:
        test_set: 'TailNLG' or 'WebNLG' - determines which files to analyze
    """
    
    # Determine file prefix based on test set
    test_set_prefix = "tail_" if test_set == "TailNLG" else "web_"
    
    # Define metric pairs to analyze
    metric_pairs = [
        ('bertscore_rescaled', 'bleu'),
        ('bertscore_rescaled', 'rouge1'),
        ('bertscore_rescaled', 'rouge2'),
        ('bertscore_rescaled', 'rougeL'),

    ]
    
    for model_name in MODELS:
        print(f"\n{'='*100}")
        print(f"Analyzing metric correlations for model: {model_name}")
        print(f"Test set: {test_set}")
        print('='*100)
        
        output_dir = Path(OUTPUT_DIR) / model_name.replace('/', '_')
        model_safe_name = model_name.replace('/', '_')
        
        # Analyze correlations for each method
        all_methods = ['zero_shot']#, 'one_shot', 'few_shot', 'contrastive_few_shot', 'fine_tuned_zero_shot']
        
        for method in all_methods:
            file_path = output_dir / f"{test_set_prefix}{method}_{model_safe_name}.json"
            
            if not file_path.exists():
                print(f"\nSkipping {method}: file not found")
                continue
            
            print(f"\n{'-'*100}")
            print(f"Method: {method}")
            print(f"File: {file_path.name}")
            print('-'*100)
            
            # Create correlation results directory
            corr_dir = output_dir / "correlations"
            corr_dir.mkdir(parents=True, exist_ok=True)
            
            # Store all correlation results
            all_correlations = {}
            
            try:
                for metric1, metric2 in metric_pairs:
                    try:
                        print(f"\n{'─'*80}")
                        
                        # Compute overall correlation
                        overall_corr = VerbalizationEvaluator.compute_metric_correlations(
                            file_path=file_path,
                            metric1=metric1,
                            metric2=metric2,
                            group_by=None
                        )

                        # Store results
                        pair_key = f"{metric1}_vs_{metric2}"
                        all_correlations[pair_key] = {
                            'overall': overall_corr,
                            #'by_language': by_language_corr
                        }
                        
                    except ValueError as e:
                        print(f"  Skipping {metric1} vs {metric2}: {e}")
                        continue
                    except Exception as e:
                        print(f"  Error analyzing {metric1} vs {metric2}: {e}")
                        continue
                
                # Save all correlations to JSON
                if all_correlations:
                    # Create a summary text file
                    output_txt = corr_dir / f"correlations_{method}_{test_set.lower()}_{model_safe_name}.txt"
                    with open(output_txt, 'w', encoding='utf-8') as f:
                        f.write("="*100 + "\n")
                        f.write(f"METRIC CORRELATIONS SUMMARY\n")
                        f.write(f"Model: {model_name}\n")
                        f.write(f"Method: {method}\n")
                        f.write(f"Test set: {test_set}\n")
                        f.write("="*100 + "\n\n")
                        
                        for pair_key, correlations in all_correlations.items():
                            metric1, metric2 = pair_key.replace('_vs_', ' vs ').split(' vs ')
                            f.write(f"\n{'-'*100}\n")
                            f.write(f"{metric1.upper()} vs {metric2.upper()}\n")
                            f.write(f"{'-'*100}\n")
                            
                            # Overall correlation
                            overall = correlations['overall']
                            if overall['pearson']['r'] is not None:
                                f.write(f"\nOverall (n={overall['n']}):\n")
                                f.write(f"  Pearson r:  {overall['pearson']['r']:>6.3f} (p={overall['pearson']['p_value']:.4f}) [{overall['pearson']['interpretation']}]\n")
                                f.write(f"  Spearman ρ: {overall['spearman']['rho']:>6.3f} (p={overall['spearman']['p_value']:.4f}) [{overall['spearman']['interpretation']}]\n")

                        
                        f.write("\n" + "="*100 + "\n")
                    
                    print(f"\n✓ Correlations saved to:")
                    #print(f"  - {output_json}")
                    print(f"  - {output_txt}")
                else:
                    print(f"\n  No valid correlations computed")
                
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
                continue
        
        print(f"\n{'='*100}")
        print(f"Correlation analysis completed for {model_name}")
        print('='*100)

  
if __name__ == "__main__":    
    # Analyze metric correlations for TailNLG
    analyze_metric_correlations(test_set='TailNLG')
    
    # Analyze metric correlations for WebNLG
    #analyze_metric_correlations(test_set='WebNLG')