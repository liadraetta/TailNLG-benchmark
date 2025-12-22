from src.independency_test import IndependentIntrinsic,NewMetric
import csv
import argparse


parser = argparse.ArgumentParser(description="Run independency tests for different models and metrics.")
'''parser.add_argument('--output', type=str, default='experiments/output/independency_results.csv', 
                    help='Path to the output CSV file.')
parser.add_argument('--sample', type=int, default=500, 
                    help='Number of samples to use for the independency test.')
args = parser.parse_args()
'''
writer = csv.DictWriter(open('experiments/output/wilcoxon_intrinsic.csv','a'),fieldnames=['model','lang','statistics','p_value','direction','runs','metric'])



paths = {
    'llama':{
        'tail': 'experiments/output/meta-llama_Llama-3.2-3B-Instruct/tail_zero_shot_meta-llama_Llama-3.2-3B-Instruct.json',
        'web': 'experiments/output/meta-llama_Llama-3.2-3B-Instruct/web_zero_shot_meta-llama_Llama-3.2-3B-Instruct.json',
        'model_path': 'meta-llama/Llama-3.2-3B-Instruct'
    },
    'gemma':{
        'tail': 'experiments/output/google_gemma-3-12b-it/tail_zero_shot_google_gemma-3-12b-it.json',
        'web': 'experiments/output/google_gemma-3-12b-it/web_zero_shot_google_gemma-3-12b-it.json',
        'model_path': 'google/gemma-3-12b-it'
    },
    'qwen':{
        'tail': 'experiments/output/Qwen_Qwen2.5-7B-Instruct/tail_zero_shot_Qwen_Qwen2.5-7B-Instruct.json',
        'web': 'experiments/output/Qwen_Qwen2.5-7B-Instruct/web_zero_shot_Qwen_Qwen2.5-7B-Instruct.json',
        
    }
}


langs = [None,'en','it','es']

metrics = ['bertscore_rescaled_score','bleu_score','chrf_score','rouge1_score','rouge2_score','rougeL_score','perplexity_score']

my_tester = NewMetric()
for metric in metrics:
    for model in paths:
        for lang in langs:
            results = my_tester.test_independency_intrinsic(path=paths[model]['tail'],
                                                            #path_b=paths[model]['web'],
                                                  metric=metric,
                                                  lang=lang,
                                                  sample=500)
            row = {
                'model': model,
                'lang': lang,
                'statistics': results['statistics'],
                'p_value': results['p_value'],
                'direction': results['direction'],
                'runs': results['runs'],
                'metric': metric,
            }
            writer.writerow(row)
            print(row)

