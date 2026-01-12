import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Tuple, Optional, List, Dict, Any
from huggingface_hub import login
from datetime import datetime
from tqdm import tqdm
import time
import json
from pathlib import Path

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

        # Add test set prefix
        test_set_prefix = "tail_" if self.test_set == "TailNLG" else "web_"
        
        # Add 'fine_tuned' prefix if model has been fine-tuned
        fine_tuned_prefix = "fine_tuned_" if self.is_fine_tuned else ""
        
        return self.output_dir / f"{test_set_prefix}{fine_tuned_prefix}{method}_{model_safe_name}.json"
    
    def _load_existing_results(self, output_file: Path) -> Dict[Tuple[str, str], List[Dict]]:

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

    def _clean_output(self, text: str) -> str:

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
