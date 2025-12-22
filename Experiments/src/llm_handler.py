import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Tuple, Optional, List, Dict, Any
from huggingface_hub import login
from datetime import datetime
from tqdm import tqdm
from trl import SFTTrainer
from peft import PeftModel, LoraConfig
from transformers import TrainingArguments
from datasets import Dataset
import time
import os
import gc
import json
import shutil
from pathlib import Path
from dotenv import load_dotenv

class LLMHandler:
    def __init__(self, model_name, merged_webnlg_dataset=None, longtail_webnlg_dataset=None, use_quantization=False, output_dir=None, num_generations=3, temperature=0.7, inference_batch_size=8, test_set='TailNLG'):
        self.model_name = model_name
        self.original_model_name = model_name  # Keep track of original model name
        self.merged_webnlg_dataset = merged_webnlg_dataset
        self.longtail_webnlg_dataset = longtail_webnlg_dataset
        self.use_quantization = use_quantization
        self.output_dir = Path(output_dir) / model_name.replace('/', '_') if output_dir else Path('./output') / model_name.replace('/', '_')
        self.num_generations = num_generations
        self.temperature = temperature
        self.inference_batch_size = inference_batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_fine_tuned = False  # Flag to track if model is fine-tuned
        self.test_set = test_set  # 'TailNLG' or 'WebNLG'
        self.tokenizer, self.model = self._load_model()
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_test_dataset(self) -> List[Dict]:
        """
        Get the appropriate test dataset based on test_set parameter.
        
        Returns:
            List of test entries from either TailNLG or WebNLG test set
        """
        if self.test_set == 'TailNLG':
            if self.longtail_webnlg_dataset is None:
                raise ValueError("longtail_webnlg_dataset is required for TailNLG test set")
            return self.longtail_webnlg_dataset
        elif self.test_set == 'WebNLG':
            if self.merged_webnlg_dataset is None or 'test' not in self.merged_webnlg_dataset:
                raise ValueError("merged_webnlg_dataset with 'test' split is required for WebNLG test set")
            return self.merged_webnlg_dataset['test']
        else:
            raise ValueError(f"Invalid test_set: {self.test_set}. Must be 'TailNLG' or 'WebNLG'")

    def _get_prompt(self, language: str = "en") -> str:
        if language == "en":
            prompt = ("In English, structured data is commonly represented as triples, with the format "
                    "[subject, predicate, object]. Based on these triples, generate a single-paragraph text "
                    "composed of complete, grammatically correct, and natural sentences. "
                    "INSTRUCTIONS: "
                    "- Generate the text solely from the input triples "
                    "- Return the final verbalization with this format: The final verbalization is: [verbalization output] "
                    "- Insert the verbalization of the input triples within square brackets [], without adding anything else "
                    "INPUT TRIPLES: ")
        elif language == "es":
            prompt = ("En español, los datos estructurados se representan comúnmente como tríos, con el formato "
                    "[sujeto, predicado, objeto]. Basándose en estos tríos, genere un texto de un solo párrafo "
                    "compuesto por oraciones completas, gramaticalmente correctas y naturales."
                    "INSTRUCCIONES: "
                    "- Genere el texto únicamente a partir de las tripletas de entrada. "
                    "- Devuelva la verbalización final con este formato: La verbalización final es: [salida de verbalización] "
                    "- entre corchetes [] inserte la verbalización de las tripletas de entrada, sin añadir nada más "
                    "TRIPLETAS DE ENTRADA: ")
        elif language == "it" or language == "it-PE":
            prompt = ("In italiano, i dati strutturati sono comunemente rappresentati come triple, con il formato "
                    "[soggetto, predicato, oggetto]. Sulla base di queste triple, genera un testo di un solo paragrafo "
                    "composto da frasi complete, grammaticalmente corrette e naturali. "
                    "INSTRUZIONI: "
                    "- Genera il testo esclusivamente dalle triple in input "
                    "- Restituisci la verbalizzazione finale con questo formato: La verbalizzazione finale è: [output di verbalizzazione] "
                    "- tra le parentesi quadre [] inserisci la verbalizzazione delle triple in input, senza aggiungere altro "
                    "TRIPLE IN INPUT: ")
        
        return prompt
    
    def _get_negative_prompt(self, language: str = "en") -> str:
        if language == "en":
            prompt = ("In English, structured data is commonly represented as triples, with the format "
                    "[subject, predicate, object]. Based on these triples, generate an incorrect text. "
                    "INSTRUCTIONS:"
                    "- Generate the text solely from the input triples"
                    "- Return the final verbalization with this format: The final incorrect verbalization is: [verbalization output]"
                    "INPUT TRIPLES:")
        elif language == "es":
            prompt = ("En español, los datos estructurados se representan comúnmente como tríos, con el formato "
                    "[sujeto, predicado, objeto]. Basándose en estos tríos, genera un texto incorrecto. "
                    "INSTRUCCIONES: "
                    "- Genere el texto únicamente a partir de las tripletas de entrada. "
                    "- Devuelva la verbalización final con este formato: La verbalización final incorrecta es: [salida de verbalización]"
                    "TRIPLETAS DE ENTRADA:")
        elif language == "it":
            prompt = ("In italiano, i dati strutturati sono comunemente rappresentati come triple, con il formato "
                    "[soggetto, predicato, oggetto]. Sulla base di queste triple, genera un testo errato. "
                    "INSTRUZIONI:"
                    "- Genera il testo esclusivamente dalle triple in input"
                    "- Restituisci la verbalizzazione finale con questo formato: La verbalizzazione finale errata è: [output di verbalizzazione]"
                    "TRIPLE IN INPUT:")
        
        return prompt
    
    def _get_output_filename(self, method: str, model_safe_name: str) -> Path:
        """
        Generate output filename for a specific method.
        
        Format: [fine_tuned_][test_set_prefix]method_modelname.json
        Examples:
            - tail_zero_shot_model.json (TailNLG, base model)
            - web_zero_shot_model.json (WebNLG, base model)
            - fine_tuned_tail_zero_shot_model.json (TailNLG, fine-tuned)
            - fine_tuned_web_zero_shot_model.json (WebNLG, fine-tuned)
        """
        # Add test set prefix
        test_set_prefix = "tail_" if self.test_set == "TailNLG" else "web_"
        
        # Add 'fine_tuned' prefix if model has been fine-tuned
        fine_tuned_prefix = "fine_tuned_" if self.is_fine_tuned else ""
        
        return self.output_dir / f"{test_set_prefix}{fine_tuned_prefix}{method}_{model_safe_name}.json"
    
    def _load_existing_results(self, output_file: Path) -> Dict[Tuple[str, str], List[Dict]]:
        """
        Load existing results from file.
        Returns a dictionary mapping (eid, language) tuple to list of generation results.
        Each language needs its own set of generations for the same eid.
        """
        if not output_file.exists():
            return {}
        
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            # Group by (eid, language) tuple
            eid_lang_to_results = {}
            for result in results:
                eid = result.get('eid')
                language = result.get('language')
                if language is None:
                    raise ValueError(f"Result entry missing 'language': {result}")
                key = (eid, language)
                
                if key not in eid_lang_to_results:
                    eid_lang_to_results[key] = []
                eid_lang_to_results[key].append(result)
            
            return eid_lang_to_results
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load existing results from {output_file}: {e}")
            return {}
    
    def _save_result_incremental(self, result: Dict, output_file: Path):
        """
        Save a single result incrementally to file.
        Appends to existing results or creates new file.
        """
        # Load existing results
        existing_results = []
        if output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing_results = []
        
        # Append new result
        existing_results.append(result)
        
        # Save back to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(existing_results, f, indent=2, ensure_ascii=False)
    
    def _get_generations_needed(self, eid: str, language: str, existing_results: Dict[Tuple[str, str], List[Dict]]) -> int:
        """
        Determine how many more generations are needed for a given (eid, language) pair.
        Counts only unique generation_ids to avoid duplicates.
        
        Args:
            eid: Entry ID
            language: Language code (en, es, it, etc.)
            existing_results: Dictionary mapping (eid, language) to list of results
        
        Returns:
            Number of generations still needed (0 to num_generations)
        """
        key = (eid, language)
        
        if key not in existing_results:
            return self.num_generations
        
        # Count unique generation_ids for this (eid, language) pair
        unique_generation_ids = set()
        for result in existing_results[key]:
            gen_id = result.get('generation_id')
            if gen_id is not None:
                unique_generation_ids.add(gen_id)
        
        current_count = len(unique_generation_ids)
        return max(0, self.num_generations - current_count)
    
    def _get_missing_generation_ids(self, eid: str, language: str, existing_results: Dict[Tuple[str, str], List[Dict]]) -> List[int]:
        """
        Get the list of missing generation_ids for a given (eid, language) pair.
        Returns generation_ids that are missing from the complete set [0, 1, 2, ...].
        
        Args:
            eid: Entry ID
            language: Language code (en, es, it, etc.)
            existing_results: Dictionary mapping (eid, language) to list of results
        
        Returns:
            Sorted list of missing generation_ids
        """
        key = (eid, language)
        
        # Find existing unique generation_ids for this (eid, language) pair
        existing_gen_ids = set()
        if key in existing_results:
            for result in existing_results[key]:
                gen_id = result.get('generation_id')
                if gen_id is not None:
                    existing_gen_ids.add(gen_id)
        
        # Find missing generation_ids (should be 0, 1, 2, ...)
        all_gen_ids = set(range(self.num_generations))
        missing_gen_ids = sorted(all_gen_ids - existing_gen_ids)
        
        return missing_gen_ids

    def zero_shot(self, prompt: Optional[str] = None):
        """
        Perform zero-shot verbalization on the test dataset using batch processing.
        Generates multiple verbalizations per triple and saves incrementally.
        Supports resuming from previous runs.
        Works with both TailNLG and WebNLG test sets based on test_set parameter.
        
        Args:
            prompt: Custom prompt to use. If None, uses default prompt based on language.
        
        Returns:
            List of all results (including previously generated ones)
        """
        # Get the appropriate test dataset
        test_dataset = self._get_test_dataset()
        
        # Prepare output file - use original model name for consistency
        model_safe_name = self.original_model_name.replace('/', '_')
        output_file = self._get_output_filename('zero_shot', model_safe_name)
        
        # Load existing results
        print(f"Checking for existing results in {output_file}...")
        existing_results = self._load_existing_results(output_file)
        print(f"Found {len(existing_results)} unique (eid, language) pairs with existing generations")
        
        # Filter dataset to only include entries that need more generations
        entries_to_process = []
        for entry in test_dataset:
            eid = entry.get('eid')
            language = entry.get('language')
            if language is None:
                raise ValueError(f"Test entry missing 'language': {entry}")
            
            # Get missing generation_ids for this (eid, language) pair
            missing_gen_ids = self._get_missing_generation_ids(eid, language, existing_results)
            
            if missing_gen_ids:
                # Add entry for each missing generation_id
                for gen_id in missing_gen_ids:
                    entry_copy = entry.copy()
                    entry_copy['_generation_id'] = gen_id
                    entries_to_process.append(entry_copy)
        
        total_to_generate = len(entries_to_process)
        total_skipped = len(test_dataset) * self.num_generations - total_to_generate
        
        print(f"Total entries to generate: {total_to_generate}")
        print(f"Total skipped (already complete): {total_skipped}")
        
        # Process in batches
        total_generated = 0
        for start in tqdm(range(0, len(entries_to_process), self.inference_batch_size), desc="Generating verbalizations (zero-shot)"):
            batch = entries_to_process[start:start + self.inference_batch_size]
            
            # Prepare batch data
            batch_prompts = []
            for entry in batch:
                language = entry.get('language', 'en')
                if prompt is None:
                    base_prompt = self._get_prompt(language)
                else:
                    base_prompt = prompt
                batch_prompts.append(base_prompt)
            
            # Generate for batch
            start_time = time.time()
            batch_results = self._verbalize_triples_batch(batch, batch_prompts, messages_list=None)
            elapsed_time = time.time() - start_time
            
            # Save results
            for entry, (generated_text, full_prompt) in zip(batch, batch_results):
                eid = entry.get('eid')
                language = entry.get('language', 'en')
                
                if prompt is None:
                    base_prompt = self._get_prompt(language)
                else:
                    base_prompt = prompt
                
                result = {
                    'eid': eid,
                    'generation_id': entry['_generation_id'],
                    'category': entry.get('category'),
                    'num_triples': entry.get('num_triples'),
                    'data_unit': entry['data_unit'],
                    'actual': entry.get('sentence'),
                    'prediction': generated_text,
                    'prompt': full_prompt,
                    'base_prompt': base_prompt,
                    'time': elapsed_time / len(batch),  # Average time per entry
                    'language': language,
                    'temperature': self.temperature
                }
                
                # Save incrementally
                self._save_result_incremental(result, output_file)
                total_generated += 1
        
        print(f"\nGeneration complete!")
        print(f"Total new generations: {total_generated}")
        print(f"Results saved to: {output_file}")
        
        # Load and return all results
        with open(output_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
        
        return all_results

    def one_shot(self, examples_by_language: Dict[str, Tuple[List[str], str]], prompt: Optional[str] = None):
        """
        Perform one-shot verbalization with language-specific examples using batch processing.
        Generates multiple verbalizations per triple and saves incrementally.
        Supports resuming from previous runs.
        Works with both TailNLG and WebNLG test sets based on test_set parameter.
        
        Args:
            examples_by_language: Dictionary mapping language codes to (triples_list, verbalization) tuples
                                  e.g., {"en": ([triples], "text"), "es": ([triples], "texto"), ...}
            prompt: Custom prompt to use. If None, uses default prompt based on language.
        
        Returns:
            List of all results (including previously generated ones)
        """
        # Get the appropriate test dataset
        test_dataset = self._get_test_dataset()
        
        # Prepare output file - use original model name for consistency
        model_safe_name = self.original_model_name.replace('/', '_')
        output_file = self._get_output_filename('one_shot', model_safe_name)
        
        # Load existing results
        print(f"Checking for existing results in {output_file}...")
        existing_results = self._load_existing_results(output_file)
        print(f"Found {len(existing_results)} unique (eid, language) pairs with existing generations")
        
        # Filter dataset to only include entries that need more generations
        entries_to_process = []
        for entry in test_dataset:
            eid = entry.get('eid')
            language = entry.get('language')
            if language is None:
                raise ValueError(f"Test entry missing 'language': {entry}")
                        
            # Get missing generation_ids for this (eid, language) pair
            missing_gen_ids = self._get_missing_generation_ids(eid, language, existing_results)
            
            if missing_gen_ids:
                # Add entry for each missing generation_id
                for gen_id in missing_gen_ids:
                    entry_copy = entry.copy()
                    entry_copy['_generation_id'] = gen_id
                    entries_to_process.append(entry_copy)
        
        total_to_generate = len(entries_to_process)
        total_skipped = len(test_dataset) * self.num_generations - total_to_generate
        
        print(f"Total entries to generate: {total_to_generate}")
        print(f"Total skipped (already complete): {total_skipped}")
        
        # Process in batches
        total_generated = 0
        for start in tqdm(range(0, len(entries_to_process), self.inference_batch_size), desc="Generating verbalizations (one-shot)"):
            batch = entries_to_process[start:start + self.inference_batch_size]
            
            # Prepare batch data
            batch_prompts = []
            batch_messages = []
            
            for entry in batch:
                language = entry.get('language', 'en')
                lang_code = 'it' if 'it' in language.lower() else language.lower()
                
                if prompt is None:
                    base_prompt = self._get_prompt(lang_code)
                else:
                    base_prompt = prompt
                batch_prompts.append(base_prompt)
                
                # Get the example for this language
                if lang_code in examples_by_language:
                    example_triples, example_verbalization = examples_by_language[lang_code]
                else:
                    example_triples, example_verbalization = examples_by_language.get('en', ([], ""))
                
                assistant_message = ""
                if lang_code == "en":
                    assistant_message = f"The final verbalization is: [{example_verbalization}]"
                elif lang_code == "es":
                    assistant_message = f"La verbalización final es: [{example_verbalization}]"
                elif lang_code == "it":
                    assistant_message = f"La verbalizzazione finale è: [{example_verbalization}]"
                
                example_message = [
                    {"role": "user", "content": f"{base_prompt}\nTriples: " + " ".join(example_triples)},
                    {"role": "assistant", "content": assistant_message}
                ]
                batch_messages.append(example_message)
            
            # Generate for batch
            start_time = time.time()
            batch_results = self._verbalize_triples_batch(batch, batch_prompts, messages_list=batch_messages)
            elapsed_time = time.time() - start_time
            
            # Save results
            for entry, (generated_text, full_prompt) in zip(batch, batch_results):
                eid = entry.get('eid')
                language = entry.get('language', 'en')
                lang_code = 'it' if 'it' in language.lower() else language.lower()
                
                if prompt is None:
                    base_prompt = self._get_prompt(lang_code)
                else:
                    base_prompt = prompt
                
                result = {
                    'eid': eid,
                    'generation_id': entry['_generation_id'],
                    'category': entry.get('category'),
                    'num_triples': entry.get('num_triples'),
                    'data_unit': entry['data_unit'],
                    'actual': entry.get('sentence'),
                    'prediction': generated_text,
                    'prompt': full_prompt,
                    'base_prompt': base_prompt,
                    'time': elapsed_time / len(batch),
                    'language': language,
                    'temperature': self.temperature
                }
                
                # Save incrementally
                self._save_result_incremental(result, output_file)
                total_generated += 1
        
        print(f"\nGeneration complete!")
        print(f"Total new generations: {total_generated}")
        print(f"Results saved to: {output_file}")
        
        # Load and return all results
        with open(output_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
        
        return all_results

    def few_shot(self, examples_by_language: Dict[str, List[Tuple[List[str], str]]], prompt: Optional[str] = None):
        """
        Perform few-shot verbalization with language-specific examples using batch processing.
        Generates multiple verbalizations per triple and saves incrementally.
        Supports resuming from previous runs.
        Works with both TailNLG and WebNLG test sets based on test_set parameter.
        
        Args:
            examples_by_language: Dictionary mapping language codes to lists of (triples_list, verbalization) tuples
                                  e.g., {"en": [([triples], "text"), ...], "es": [([triples], "texto"), ...], ...}
            prompt: Custom prompt to use. If None, uses default prompt based on language.
        
        Returns:
            List of all results (including previously generated ones)
        """
        # Get the appropriate test dataset
        test_dataset = self._get_test_dataset()
        
        # Prepare output file - use original model name for consistency
        model_safe_name = self.original_model_name.replace('/', '_')
        output_file = self._get_output_filename('few_shot', model_safe_name)
        
        # Load existing results
        print(f"Checking for existing results in {output_file}...")
        existing_results = self._load_existing_results(output_file)
        print(f"Found {len(existing_results)} unique (eid, language) pairs with existing generations")
        
        # Filter dataset to only include entries that need more generations
        entries_to_process = []
        for entry in test_dataset:
            eid = entry.get('eid')
            language = entry.get('language')
            if language is None:
                raise ValueError(f"Test entry missing 'language': {entry}")
                        
            # Get missing generation_ids for this (eid, language) pair
            missing_gen_ids = self._get_missing_generation_ids(eid, language, existing_results)
            
            if missing_gen_ids:
                # Add entry for each missing generation_id
                for gen_id in missing_gen_ids:
                    entry_copy = entry.copy()
                    entry_copy['_generation_id'] = gen_id
                    entries_to_process.append(entry_copy)
        
        total_to_generate = len(entries_to_process)
        total_skipped = len(test_dataset) * self.num_generations - total_to_generate
        
        print(f"Total entries to generate: {total_to_generate}")
        print(f"Total skipped (already complete): {total_skipped}")
        
        # Process in batches
        total_generated = 0
        for start in tqdm(range(0, len(entries_to_process), self.inference_batch_size), desc="Generating verbalizations (few-shot)"):
            batch = entries_to_process[start:start + self.inference_batch_size]
            
            # Prepare batch data
            batch_prompts = []
            batch_messages = []
            batch_num_examples = []
            
            for entry in batch:
                language = entry.get('language', 'en')
                lang_code = 'it' if 'it' in language.lower() else language.lower()
                
                if prompt is None:
                    base_prompt = self._get_prompt(lang_code)
                else:
                    base_prompt = prompt
                batch_prompts.append(base_prompt)
                
                # Get the examples for this language
                if lang_code in examples_by_language:
                    language_examples = examples_by_language[lang_code]
                else:
                    language_examples = examples_by_language.get('en', [])
                
                batch_num_examples.append(len(language_examples))
                
                # Create the few-shot messages
                few_shot_messages = []
                for example_triples, example_verbalization in language_examples:
                    assistant_message = ""
                    if lang_code == "en":
                        assistant_message = f"The final verbalization is: [{example_verbalization}]"
                    elif lang_code == "es":
                        assistant_message = f"La verbalización final es: [{example_verbalization}]"
                    elif lang_code == "it":
                        assistant_message = f"La verbalizzazione finale è: [{example_verbalization}]"
                    
                    few_shot_messages.extend([
                        {"role": "user", "content": f"{base_prompt}\nTriples: " + " ".join(example_triples)},
                        {"role": "assistant", "content": assistant_message}
                    ])
                
                batch_messages.append(few_shot_messages)
            
            # Generate for batch
            start_time = time.time()
            batch_results = self._verbalize_triples_batch(batch, batch_prompts, messages_list=batch_messages)
            elapsed_time = time.time() - start_time
            
            # Save results
            for entry, (generated_text, full_prompt), num_examples in zip(batch, batch_results, batch_num_examples):
                eid = entry.get('eid')
                language = entry.get('language', 'en')
                lang_code = 'it' if 'it' in language.lower() else language.lower()
                
                if prompt is None:
                    base_prompt = self._get_prompt(lang_code)
                else:
                    base_prompt = prompt
                
                result = {
                    'eid': eid,
                    'generation_id': entry['_generation_id'],
                    'category': entry.get('category'),
                    'num_triples': entry.get('num_triples'),
                    'data_unit': entry['data_unit'],
                    'actual': entry.get('sentence'),
                    'prediction': generated_text,
                    'prompt': full_prompt,
                    'base_prompt': base_prompt,
                    'num_examples': num_examples,
                    'time': elapsed_time / len(batch),
                    'language': language,
                    'temperature': self.temperature
                }
                
                # Save incrementally
                self._save_result_incremental(result, output_file)
                total_generated += 1
        
        print(f"\nGeneration complete!")
        print(f"Total new generations: {total_generated}")
        print(f"Results saved to: {output_file}")
        
        # Load and return all results
        with open(output_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
        
        return all_results

    def contrastive_few_shot(self, examples_by_language: Dict[str, List[Tuple[List[str], str]]], negative_examples_by_language: Dict[str, List[Tuple[List[str], str]]], prompt: Optional[str] = None):
        """
        Perform contrastive few-shot verbalization with language-specific positive and negative examples using batch processing.
        Generates multiple verbalizations per triple and saves incrementally.
        Supports resuming from previous runs.
        Works with both TailNLG and WebNLG test sets based on test_set parameter.
        
        Args:
            examples_by_language: Dictionary mapping language codes to lists of (triples_list, verbalization) tuples
                                  e.g., {"en": [([triples], "text"), ...], "es": [([triples], "texto"), ...], ...}
            negative_examples_by_language: Dictionary mapping language codes to lists of negative examples
            prompt: Custom prompt to use. If None, uses default prompt based on language.
        
        Returns:
            List of all results (including previously generated ones)
        """
        # Get the appropriate test dataset
        test_dataset = self._get_test_dataset()
        
        # Prepare output file - use original model name for consistency
        model_safe_name = self.original_model_name.replace('/', '_')
        output_file = self._get_output_filename('contrastive_few_shot', model_safe_name)
        
        # Load existing results
        print(f"Checking for existing results in {output_file}...")
        existing_results = self._load_existing_results(output_file)
        print(f"Found {len(existing_results)} unique (eid, language) pairs with existing generations")
        
        # Filter dataset to only include entries that need more generations
        entries_to_process = []
        for entry in test_dataset:
            eid = entry.get('eid')
            language = entry.get('language')
            if language is None:
                raise ValueError(f"Test entry missing 'language': {entry}")
                        
            # Get missing generation_ids for this (eid, language) pair
            missing_gen_ids = self._get_missing_generation_ids(eid, language, existing_results)
            
            if missing_gen_ids:
                # Add entry for each missing generation_id
                for gen_id in missing_gen_ids:
                    entry_copy = entry.copy()
                    entry_copy['_generation_id'] = gen_id
                    entries_to_process.append(entry_copy)
        
        total_to_generate = len(entries_to_process)
        total_skipped = len(test_dataset) * self.num_generations - total_to_generate
        
        print(f"Total entries to generate: {total_to_generate}")
        print(f"Total skipped (already complete): {total_skipped}")
        
        # Process in batches
        total_generated = 0
        for start in tqdm(range(0, len(entries_to_process), self.inference_batch_size), desc="Generating verbalizations (contrastive few-shot)"):
            batch = entries_to_process[start:start + self.inference_batch_size]
            
            # Prepare batch data
            batch_prompts = []
            batch_messages = []
            batch_num_examples = []
            
            for entry in batch:
                language = entry.get('language', 'en')
                lang_code = 'it' if 'it' in language.lower() else language.lower()
                
                if prompt is None:
                    base_prompt = self._get_prompt(lang_code)
                    negative_base_prompt = self._get_negative_prompt(lang_code)
                else:
                    base_prompt = prompt
                    negative_base_prompt = None
                
                batch_prompts.append(base_prompt)
                
                # Get the examples for this language
                if lang_code in examples_by_language:
                    language_examples = examples_by_language[lang_code]
                    language_negative_examples = negative_examples_by_language.get(lang_code, [])
                else:
                    language_examples = examples_by_language.get('en', [])
                    language_negative_examples = negative_examples_by_language.get('en', [])
                
                batch_num_examples.append(len(language_examples))
                
                # Create the few-shot messages (negative examples first, then positive)
                few_shot_messages = []
                
                # Add negative examples
                for negative_example_triples, negative_verbalization in language_negative_examples:
                    assistant_message = ""
                    if lang_code == "en":
                        assistant_message = f"The final incorrect verbalization is: [{negative_verbalization}]"
                    elif lang_code == "es":
                        assistant_message = f"La verbalización final incorrecta es: [{negative_verbalization}]"
                    elif lang_code == "it":
                        assistant_message = f"La verbalizzazione finale errata è: [{negative_verbalization}]"
                    
                    few_shot_messages.extend([
                        {"role": "user", "content": f"{negative_base_prompt}\nTriples: " + " ".join(negative_example_triples)},
                        {"role": "assistant", "content": assistant_message}
                    ])
                
                # Add positive examples
                for example_triples, example_verbalization in language_examples:
                    assistant_message = ""
                    if lang_code == "en":
                        assistant_message = f"The final verbalization is: [{example_verbalization}]"
                    elif lang_code == "es":
                        assistant_message = f"La verbalización final es: [{example_verbalization}]"
                    elif lang_code == "it":
                        assistant_message = f"La verbalizzazione finale è: [{example_verbalization}]"
                    
                    few_shot_messages.extend([
                        {"role": "user", "content": f"{base_prompt}\nTriples: " + " ".join(example_triples)},
                        {"role": "assistant", "content": assistant_message}
                    ])
                
                batch_messages.append(few_shot_messages)
            
            # Generate for batch
            start_time = time.time()
            batch_results = self._verbalize_triples_batch(batch, batch_prompts, messages_list=batch_messages)
            elapsed_time = time.time() - start_time
            
            # Save results
            for entry, (generated_text, full_prompt), num_examples in zip(batch, batch_results, batch_num_examples):
                eid = entry.get('eid')
                language = entry.get('language', 'en')
                lang_code = 'it' if 'it' in language.lower() else language.lower()
                
                if prompt is None:
                    base_prompt = self._get_prompt(lang_code)
                else:
                    base_prompt = prompt
                
                result = {
                    'eid': eid,
                    'generation_id': entry['_generation_id'],
                    'category': entry.get('category'),
                    'num_triples': entry.get('num_triples'),
                    'data_unit': entry['data_unit'],
                    'actual': entry.get('sentence'),
                    'prediction': generated_text,
                    'prompt': full_prompt,
                    'base_prompt': base_prompt,
                    'num_examples': num_examples,
                    'time': elapsed_time / len(batch),
                    'language': language,
                    'temperature': self.temperature
                }
                
                # Save incrementally
                self._save_result_incremental(result, output_file)
                total_generated += 1
        
        print(f"\nGeneration complete!")
        print(f"Total new generations: {total_generated}")
        print(f"Results saved to: {output_file}")
        
        # Load and return all results
        with open(output_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
        
        return all_results

    def _clean_output(self, text: str) -> str:
        """
        Clean the generated output text by extracting content within ": [" and "]".
        
        Args:
            text: The generated text string
        Returns:
            Cleaned text string
        """
        # Extract content within :[...]
        start_tag = ": ["
        end_tag = "]"
        start_idx = text.find(start_tag)
        end_idx = text.find(end_tag)
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return text[start_idx + len(start_tag):end_idx].strip()
        return text.strip()

    def _verbalize_triples_batch(
        self, 
        batch_entries: List[Dict[str, Any]], 
        base_prompts: List[str], 
        messages_list: Optional[List[Optional[List[Dict[str, str]]]]] = None,
        temperature: Optional[float] = None
    ) -> List[Tuple[str, str]]:
        """
        Generate verbalizations for a batch of entries.
        
        Args:
            batch_entries: List of entry dictionaries, each containing 'data_unit' (triples)
            base_prompts: List of base prompts for each entry
            messages_list: List of message lists for few-shot learning (optional)
            temperature: Temperature for generation (if None, uses self.temperature)
        
        Returns:
            List of tuples (generated_text, full_prompt) for each entry
        """
        if temperature is None:
            temperature = self.temperature
        
        if messages_list is None:
            messages_list = [None] * len(batch_entries)
        
        # Save original padding side and switch to left for batch generation
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = 'left'
        
        try:
            # Prepare all prompts for the batch
            all_prompts = []
            for entry, base_prompt, messages in zip(batch_entries, base_prompts, messages_list):
                triples = entry['data_unit']
                
                # Format triples for the prompt
                triples_str = " ".join(
                    f"[subject: '{t['subject']}', predicate: '{t['predicate']}', object: '{t['object']}']"
                    for t in triples
                )
                
                # Build messages
                if messages is None:
                    msg_list = [{"role": "user", "content": f"{base_prompt}\nTriples: {triples_str}"}]
                else:
                    msg_list = messages + [{"role": "user", "content": f"{base_prompt}\nTriples: {triples_str}"}]
                
                # Format the prompt using chat template
                prompt_formatted = self.tokenizer.apply_chat_template(
                    msg_list, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                all_prompts.append(prompt_formatted)
            
            # Tokenize all prompts with padding
            inputs = self.tokenizer(
                all_prompts, 
                return_tensors="pt",
                add_special_tokens=True,
                padding=True,
                truncation=False
            ).to(self.device)
            
            # Generate for the batch
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=256,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    temperature=temperature,
                    do_sample=temperature > 0,
                )
            
            # Decode outputs
            results = []
            for i, (output, input_ids) in enumerate(zip(outputs, inputs['input_ids'])):
                # Decode the full generated text
                generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
                
                # Remove the input part to get only the generated text
                input_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
                generated_text_without_input = generated_text[len(input_text):]
                
                # Clean output
                final_output = self._clean_output(generated_text_without_input)
                
                # Complete prompt
                complete_prompt = all_prompts[i] + generated_text_without_input
                
                results.append((final_output, complete_prompt))
            
            return results
        
        finally:
            # Restore original padding side
            self.tokenizer.padding_side = original_padding_side

    def fine_tuning(
        self, 
        train_split: str = 'train',
        val_split: Optional[str] = 'dev',
        training_args: Optional[Dict[str, Any]] = None,
        peft_config: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None,
    ):
        """
        Fine-tune the model on WebNLG dataset using LoRA and SFTTrainer.
        
        Args:
            train_split: Split to use for training (default: 'train')
            val_split: Split to use for validation (default: 'dev', None to disable)
            training_args: Dictionary with training arguments (see default_training_args)
            peft_config: Dictionary with PEFT/LoRA configuration (see default_peft_config)
            output_path: Path to save the fine-tuned model
            instruction: Custom instruction for the model (default prompt will be used if None)
        
        Returns:
            trainer: The trained SFTTrainer instance
        """
        if self.merged_webnlg_dataset is None:
            raise ValueError("merged_webnlg_dataset is required for fine_tuning method")
        
        # Prepare datasets
        print("Preparing training dataset...")
        train_dataset = self._prepare_fine_tuning_dataset(
            self.merged_webnlg_dataset[train_split], 
        )
        
        eval_dataset = None
        if val_split is not None and val_split in self.merged_webnlg_dataset:
            print("Preparing validation dataset...")
            eval_dataset = self._prepare_fine_tuning_dataset(
                self.merged_webnlg_dataset[val_split], 
            )
        
        # Create model-specific output directory if not provided
        if output_path is None:
            model_safe_name = self.original_model_name.replace('/', '_')
            output_path = f'./fine_tuned_models/{model_safe_name}'
        
        output_path_obj = Path(output_path)
        
        # Check if fine-tuned model already exists (training completed)
        adapter_config_path = output_path_obj / "adapter_config.json"
        if adapter_config_path.exists():
            print(f"Fine-tuned model already exists at {output_path}")
            print("Skipping training and returning existing model path.")
            return output_path
        
        # Check for existing checkpoints to resume training
        checkpoint_path = None
        if output_path_obj.exists():
            checkpoints = sorted(output_path_obj.glob("checkpoint-*"))
            if checkpoints:
                # Get the latest checkpoint
                checkpoint_path = str(checkpoints[-1])
                print(f"Found existing checkpoint: {checkpoint_path}")
                print("Training will resume from this checkpoint.")
        
        # Default training arguments
        default_training_args = {
            'output_dir': output_path,
            'num_train_epochs': 3,
            'per_device_train_batch_size': 1,
            'per_device_eval_batch_size': 1,
            'gradient_accumulation_steps': 8,
            'learning_rate': 2e-4,
            'weight_decay': 0.01,
            'logging_steps': 25,
            'save_steps': 500,
            'eval_strategy': 'steps' if eval_dataset is not None else 'no',
            'eval_steps': 500 if eval_dataset is not None else None,
            'save_total_limit': 2,
            'fp16': torch.cuda.is_available(),
            'optim': 'paged_adamw_8bit',
            'warmup_ratio': 0.1,
            'max_grad_norm': 0.3,
            'group_by_length': True,
            'save_total_limit': 2,
            'report_to': 'none', 
        }
        
        # Merge with user-provided training args
        if training_args:
            default_training_args.update(training_args)
        
        # Default PEFT config (LoRA)
        default_peft_config = {
            'r': 4,
            'lora_alpha': 16,
            'lora_dropout': 0.1,
            'bias': 'none',
            'task_type': 'CAUSAL_LM',
            'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        }

        # Merge with user-provided PEFT config
        if peft_config:
            default_peft_config.update(peft_config)
        
        # Train the model
        print(f"Starting fine-tuning with {len(train_dataset)} training samples...")
        if eval_dataset:
            print(f"Validation set: {len(eval_dataset)} samples")
        
        trainer = self._train_model(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_args=default_training_args,
            peft_config=default_peft_config,
            resume_from_checkpoint=checkpoint_path
        )
        
        # Save the adapter model (LoRA weights only)
        output_dir = default_training_args['output_dir']
        
        print(f"Saving final adapter model to {output_dir}...")
        trainer.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Clean up intermediate checkpoints to save space
        print("Cleaning up intermediate checkpoints...")
        for checkpoint_dir in Path(output_dir).glob("checkpoint-*"):
            shutil.rmtree(checkpoint_dir)
            print(f"  Removed {checkpoint_dir.name}")
        
        print("Fine-tuning completed!")
        print(f"  - Adapter model saved to: {output_dir}")
        print(f"  - To use: call load_fine_tuned_model('{output_dir}')")
        
        return output_dir
    
    def load_fine_tuned_model(self, fine_tuned_model_path: str, merge: bool = True):
        """
        Load a fine-tuned model (LoRA adapter).
        
        Args:
            fine_tuned_model_path: Path to the adapter model directory
            merge: If True, merge adapter with base model for faster inference.
                   If False, keep adapter separate (saves memory but slower).
        """
        print(f"Loading fine-tuned adapter from {fine_tuned_model_path}...")
        
        # Destroy current model first to free memory
        if self.model is not None:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        if merge:
            # For merged model, we need to load and merge, then save and reload
            print("Loading base model for merging...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.original_model_name,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Load LoRA adapter
            print("Loading LoRA adapter...")
            peft_model = PeftModel.from_pretrained(base_model, fine_tuned_model_path)
            
            # Merge and unload
            print("Merging adapter with base model...")
            model = peft_model.merge_and_unload()
            
            # Clean up intermediate objects
            del base_model
            del peft_model
            torch.cuda.empty_cache()
            
            # Save merged model temporarily and reload with proper device_map
            print("Saving and reloading merged model for proper device placement...")
            temp_merged_path = Path(fine_tuned_model_path) / "temp_merged"
            temp_merged_path.mkdir(exist_ok=True)
            
            model.save_pretrained(temp_merged_path)
            del model
            torch.cuda.empty_cache()
            
            # Reload with device_map="auto" for proper distribution
            model = AutoModelForCausalLM.from_pretrained(
                temp_merged_path,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Clean up temporary directory
            shutil.rmtree(temp_merged_path)
            
        else:
            # Keep adapter separate
            base_model = AutoModelForCausalLM.from_pretrained(
                self.original_model_name,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True
            )
            model = PeftModel.from_pretrained(base_model, fine_tuned_model_path)
        
        # Update the instance model
        self.model = model
        
        # Set flag to indicate model is fine-tuned (affects output filenames)
        self.is_fine_tuned = True
        
        print("Fine-tuned model loaded successfully!")
        print(f"Output files will now be prefixed with 'fine_tuned_'")
        
        # Verify model device
        if torch.cuda.is_available():
            try:
                device = next(self.model.parameters()).device
                print(f"  Model device: {device}")
            except StopIteration:
                print("  Warning: Could not determine model device")
        
        return model
    
    def _prepare_fine_tuning_dataset(self, data: List[Dict]) -> Dataset:
        """
        Prepare WebNLG data for fine-tuning with chat template format.
        
        Args:
            data: List of WebNLG entries
        
        Returns:
            HuggingFace Dataset object
        """
        formatted_data = []
        
        for entry in data:
            # Get language and corresponding prompt
            language = entry.get('language', 'en')
            lang_code = 'it' if 'it' in language.lower() else language.lower()
            base_prompt = self._get_prompt(lang_code)
            
            # Format triples - convert list of dicts to formatted string
            triples = entry['data_unit']
            if isinstance(triples, list):
                combined_triples = " ".join([
                    f"[subject: '{t['subject']}', predicate: '{t['predicate']}', object: '{t['object']}']"
                    for t in triples
                ])
            else:
                combined_triples = triples
            
            # Create prompt with triples
            prompt = f"{base_prompt}\nTriples: {combined_triples}"
            
            # Output is the natural language sentence


            if lang_code == "en":
                response = f"The final verbalization is: [{entry['sentence']}]"
            elif lang_code == "es":
                response = f"La verbalización final es: [{entry['sentence']}]"
            elif lang_code == "it":
                response = f"La verbalizzazione finale è: [{entry['sentence']}]"
            
            # Format as chat messages
            chat_messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            
            formatted_data.append({"text": chat_messages})
        
        return Dataset.from_list(formatted_data)
    
    def _formatting_prompts_func(self, example):
        """
        Format prompts for training using chat template.
        
        Args:
            example: Example with 'text' field containing chat messages
        
        Returns:
            Formatted text string using tokenizer's chat template
        """
        # Apply chat template to format the messages
        return self.tokenizer.apply_chat_template(
            example['text'], 
            tokenize=False, 
            add_generation_prompt=False  # Don't add generation prompt for training
        )
    
    def _train_model(
        self, 
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
        training_args: Dict[str, Any],
        peft_config: Dict[str, Any],
        resume_from_checkpoint: Optional[str] = None
    ):
        """
        Train the model using SFTTrainer with LoRA.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Validation dataset (optional)
            training_args: Training arguments
            peft_config: PEFT/LoRA configuration
            resume_from_checkpoint: Path to checkpoint to resume training from
        
        Returns:
            Trained SFTTrainer instance
        """
        
        # Prepare model for training
        self.model.config.use_cache = False
        
        # Enable gradient checkpointing to save memory
        self.model.gradient_checkpointing_enable()
        
        # Create LoRA config
        lora_config = LoraConfig(**peft_config)
        
        # Create training arguments
        training_arguments = TrainingArguments(**training_args)
        
        # If no eval_dataset, force eval_strategy="no"
        if eval_dataset is None and training_arguments.eval_strategy != "no":
            training_arguments.eval_strategy = "no"
        
        # Tokenize datasets
        def tokenize_function(examples):
            # Apply chat template and tokenize
            formatted_text = self._formatting_prompts_func(examples)
            tokenized = self.tokenizer(
                formatted_text,
                padding="max_length",
                truncation=True,
                max_length=512
            )
            return tokenized
        
        # Apply tokenization
        train_dataset = train_dataset.map(tokenize_function, batched=False)
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(tokenize_function, batched=False)
        
        # Create trainer
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=lora_config,
            args=training_arguments,
            processing_class=self.tokenizer,  # Use processing_class instead of tokenizer
        )
        
        # Clear cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Train - resume from checkpoint if provided
        if resume_from_checkpoint:
            print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            print("Starting training from scratch...")
            trainer.train()
        
        return trainer

    def destroy_model(self):
        """
        Destroy the model and tokenizer to free memory.
        """
        if self.model is not None:
            del self.model
        
        if self.tokenizer is not None:
            del self.tokenizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Model destroyed and GPU cache cleared")
        else:
            print("Model destroyed")
    
    def _load_model(self, dtype=torch.float16):
        """
        Load an LLM model and its tokenizer.
        
        Args:
            dtype: Data type for the model (default: torch.float16)
        
        Returns:
            Tuple[tokenizer, model]
        """
        self._authenticate_huggingface()
        
        print(f"Loading model: {self.model_name}")
        
        # Setup quantization if requested
        quantization_config = None
        if self.use_quantization:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            print("  Using 8-bit quantization")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            dtype=dtype if not self.use_quantization else None,
            trust_remote_code=True,
            quantization_config=quantization_config
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        # Set pad token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Set padding side for decoder-only models
        # Use "right" for training, "left" for inference
        tokenizer.padding_side = "right"

        print(f"Model loaded: {self.model_name}")
        if torch.cuda.is_available():
            print(f"  Device: GPU (CUDA) - {torch.cuda.get_device_name(0)}")
        else:
            print(f"  Device: CPU")
        
        return tokenizer, model
    
    def _authenticate_huggingface(self):
        """Autentica con HuggingFace Hub usando il token da .env"""
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            login(token=hf_token)
            print("Autenticazione HuggingFace completata")
        else:
            print("Warning: HF_TOKEN non trovato nel file .env")