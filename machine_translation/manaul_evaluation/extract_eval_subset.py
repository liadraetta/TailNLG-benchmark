import json
import pandas as pd
import os
import argparse
import random

# python extract_eval_subset.py --run-folder run_20251120_100000 

def interpret_score(score):
    """
    Interpret XCOMET score into quality categories.
    Scores based on https://aclanthology.org/2022.wmt-1.70.pdf (p.11)

    Args:
        score: XCOMET score (0-1)
        
    Returns:
        Quality category string
    """
    if score is None:
        return None
    elif score >= 0.80:
        return "excellent"
    elif score >= 0.60:
        return "good"
    elif score >= 0.40:
        return "moderate"
    else:
        return "weak"


def is_low_quality(score):
    """
    Check if a score is moderate or weak (score <= 0.6).
    
    Args:
        score: XCOMET score (0-1)
        
    Returns:
        Boolean indicating if score is <= 0.6
    """
    if score is None:
        return False
    return score <= 0.6


def is_good_quality(score):
    """
    Check if a score is good (0.6 < score < 0.8).
    
    Args:
        score: XCOMET score (0-1)
        
    Returns:
        Boolean indicating if score is in the good range
    """
    if score is None:
        return False
    return 0.6 < score < 0.8


def extract_combinations(data, seed=None):
    """
    Extract all language combinations from the data, separating low-quality and good translations.
    
    Args:
        data: List of items from the JSON file
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of two dictionaries: (low_quality_combinations, good_quality_combinations)
    """
    if seed is not None:
        random.seed(seed)
    
    # Definisci le combinazioni possibili
    low_quality_combinations = {
        "it_to_en": [],
        "it_to_es": [],
        "en_to_it": [],
        "en_to_es": [],
        "es_to_it": [],
        "es_to_en": [],
    }
    
    good_quality_combinations = {
        "it_to_en": [],
        "it_to_es": [],
        "en_to_it": [],
        "en_to_es": [],
        "es_to_it": [],
        "es_to_en": [],
    }

    # Itera su ciascun record
    for item in data:
        if item.get("annotation_it_is_gold"):
            # IT -> EN
            en_score = item.get("annotation_en_xcomet_sentence_scores")
            record_en = {
                "unique id": item.get("unique_id"),
                "source language": "it",
                "target language": "en",
                "source verbalization": item.get("annotation_it"),
                "target verbalization": item.get("annotation_en"),
                "target verbalization score": en_score,
                "target verbalization quality category": interpret_score(en_score),
                "target verbalization detected errors": item.get("annotation_en_xcomet_error_spans"),
                "target verbalization minor errors": item.get("annotation_en_xcomet_minor_errors"),
                "target verbalization major errors": item.get("annotation_en_xcomet_major_errors"),
                "target verbalization critical errors": item.get("annotation_en_xcomet_critical_errors"),
                "corrected target verbalization explanations": item.get("annotation_en_xtower_explanations"),
                "corrected target verbalization": item.get("annotation_en_xtower_corrected_translation"),
            }
            if is_low_quality(en_score):
                low_quality_combinations["it_to_en"].append(record_en)
            elif is_good_quality(en_score):
                good_quality_combinations["it_to_en"].append(record_en)
            
            # IT -> ES
            es_score = item.get("annotation_es_xcomet_sentence_scores")
            record_es = {
                "unique id": item.get("unique_id"),
                "source language": "it",
                "target language": "es",
                "source verbalization": item.get("annotation_it"),
                "target verbalization": item.get("annotation_es"),
                "target verbalization score": es_score,
                "target verbalization quality category": interpret_score(es_score),
                "target verbalization detected errors": item.get("annotation_es_xcomet_error_spans"),
                "target verbalization minor errors": item.get("annotation_es_xcomet_minor_errors"),
                "target verbalization major errors": item.get("annotation_es_xcomet_major_errors"),
                "target verbalization critical errors": item.get("annotation_es_xcomet_critical_errors"),
                "corrected target verbalization explanations": item.get("annotation_es_xtower_explanations"),
                "corrected target verbalization": item.get("annotation_es_xtower_corrected_translation"),
            }
            if is_low_quality(es_score):
                low_quality_combinations["it_to_es"].append(record_es)
            elif is_good_quality(es_score):
                good_quality_combinations["it_to_es"].append(record_es)
        
        if item.get("annotation_en_is_gold"):
            # EN -> IT
            it_score = item.get("annotation_it_xcomet_sentence_scores")
            record_it = {
                "unique id": item.get("unique_id"),
                "source language": "en",
                "target language": "it",
                "source verbalization": item.get("annotation_en"),
                "target verbalization": item.get("annotation_it"),
                "target verbalization score": it_score,
                "target verbalization quality category": interpret_score(it_score),
                "target verbalization detected errors": item.get("annotation_it_xcomet_error_spans"),
                "target verbalization minor errors": item.get("annotation_it_xcomet_minor_errors"),
                "target verbalization major errors": item.get("annotation_it_xcomet_major_errors"),
                "target verbalization critical errors": item.get("annotation_it_xcomet_critical_errors"),
                "corrected target verbalization explanations": item.get("annotation_it_xtower_explanations"),
                "corrected target verbalization": item.get("annotation_it_xtower_corrected_translation"),
            }
            if is_low_quality(it_score):
                low_quality_combinations["en_to_it"].append(record_it)
            elif is_good_quality(it_score):
                good_quality_combinations["en_to_it"].append(record_it)
            
            # EN -> ES
            es_score = item.get("annotation_es_xcomet_sentence_scores")
            record_es = {
                "unique id": item.get("unique_id"),
                "source language": "en",
                "target language": "es",
                "source verbalization": item.get("annotation_en"),
                "target verbalization": item.get("annotation_es"),
                "target verbalization score": es_score,
                "target verbalization quality category": interpret_score(es_score),
                "target verbalization detected errors": item.get("annotation_es_xcomet_error_spans"),
                "target verbalization minor errors": item.get("annotation_es_xcomet_minor_errors"),
                "target verbalization major errors": item.get("annotation_es_xcomet_major_errors"),
                "target verbalization critical errors": item.get("annotation_es_xcomet_critical_errors"),
                "corrected target verbalization explanations": item.get("annotation_es_xtower_explanations"),
                "corrected target verbalization": item.get("annotation_es_xtower_corrected_translation"),
            }
            if is_low_quality(es_score):
                low_quality_combinations["en_to_es"].append(record_es)
            elif is_good_quality(es_score):
                good_quality_combinations["en_to_es"].append(record_es)
        
        if item.get("annotation_es_is_gold"):
            # ES -> IT
            it_score = item.get("annotation_it_xcomet_sentence_scores")
            record_it = {
                "unique id": item.get("unique_id"),
                "source language": "es",
                "target language": "it",
                "source verbalization": item.get("annotation_es"),
                "target verbalization": item.get("annotation_it"),
                "target verbalization score": it_score,
                "target verbalization quality category": interpret_score(it_score),
                "target verbalization detected errors": item.get("annotation_it_xcomet_error_spans"),
                "target verbalization minor errors": item.get("annotation_it_xcomet_minor_errors"),
                "target verbalization major errors": item.get("annotation_it_xcomet_major_errors"),
                "target verbalization critical errors": item.get("annotation_it_xcomet_critical_errors"),
                "corrected target verbalization explanations": item.get("annotation_it_xtower_explanations"),
                "corrected target verbalization": item.get("annotation_it_xtower_corrected_translation"),
            }
            if is_low_quality(it_score):
                low_quality_combinations["es_to_it"].append(record_it)
            elif is_good_quality(it_score):
                good_quality_combinations["es_to_it"].append(record_it)
            
            # ES -> EN
            en_score = item.get("annotation_en_xcomet_sentence_scores")
            record_en = {
                "unique id": item.get("unique_id"),
                "source language": "es",
                "target language": "en",
                "source verbalization": item.get("annotation_es"),
                "target verbalization": item.get("annotation_en"),
                "target verbalization score": en_score,
                "target verbalization quality category": interpret_score(en_score),
                "target verbalization detected errors": item.get("annotation_en_xcomet_error_spans"),
                "target verbalization minor errors": item.get("annotation_en_xcomet_minor_errors"),
                "target verbalization major errors": item.get("annotation_en_xcomet_major_errors"),
                "target verbalization critical errors": item.get("annotation_en_xcomet_critical_errors"),
                "corrected target verbalization explanations": item.get("annotation_en_xtower_explanations"),
                "corrected target verbalization": item.get("annotation_en_xtower_corrected_translation"),
            }
            if is_low_quality(en_score):
                low_quality_combinations["es_to_en"].append(record_en)
            elif is_good_quality(en_score):
                good_quality_combinations["es_to_en"].append(record_en)
    
    return low_quality_combinations, good_quality_combinations


def sample_combinations(low_quality_combinations, good_quality_combinations, sample_size, seed=None):
    """
    Sample a specified number of records from each combination.
    First uses low-quality translations, then adds good-quality translations if needed.
    
    Args:
        low_quality_combinations: Dictionary with low-quality combinations
        good_quality_combinations: Dictionary with good-quality combinations
        sample_size: Number of samples to extract per combination
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with sampled combinations
    """
    if seed is not None:
        random.seed(seed)
    
    sampled = {}
    stats = {}
    
    for combo in low_quality_combinations.keys():
        low_quality_records = low_quality_combinations[combo]
        good_quality_records = good_quality_combinations[combo]
        
        # Start with all low-quality records
        if len(low_quality_records) >= sample_size:
            # We have enough low-quality records
            sampled[combo] = random.sample(low_quality_records, sample_size)
            stats[combo] = {"low_quality": sample_size, "good_quality": 0}
        else:
            # We need to add good-quality records
            num_low = len(low_quality_records)
            num_needed_good = sample_size - num_low
            
            sampled[combo] = low_quality_records.copy()
            
            if len(good_quality_records) >= num_needed_good:
                # We have enough good-quality records
                sampled[combo].extend(random.sample(good_quality_records, num_needed_good))
                stats[combo] = {"low_quality": num_low, "good_quality": num_needed_good}
            else:
                # We don't have enough records in total
                sampled[combo].extend(good_quality_records)
                stats[combo] = {"low_quality": num_low, "good_quality": len(good_quality_records)}
            
            # Shuffle the combined list to mix low and good quality
            random.shuffle(sampled[combo])
    
    return sampled, stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract low-quality translations (moderate/weak, score <= 0.6) from unified corpus. "
                    "If not enough low-quality translations are found, adds good-quality translations (0.6 < score < 0.8) to reach sample size."
    )
    parser.add_argument(
        "--run-folder",
        type=str,
        required=True,
        help="Name of the run folder (e.g., run_20251120_100000)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of samples to extract per language combination (Random: 50)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="final_output",
        help="Output directory for Excel files (default: final_output)"
    )
    
    args = parser.parse_args()
    
    # Costruisci il percorso del file di input
    input_file = os.path.join("..", "output", args.run_folder, "unified_corpus.json")
    
    # Verifica che il file esista
    if not os.path.exists(input_file):
        print(f"Errore: il file {input_file} non esiste!")
        return
    
    print(f"Lettura del file: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Totale record caricati: {len(data)}")
    
    # Estrai tutte le combinazioni separando low-quality e good-quality
    print(f"\nEstrazione traduzioni...")
    low_quality_combinations, good_quality_combinations = extract_combinations(data, seed=args.seed)
    
    # Mostra statistiche prima del campionamento
    print("\nStatistiche traduzioni trovate:")
    print("Low-quality (score <= 0.6):")
    for combo, records in low_quality_combinations.items():
        print(f"  {combo}: {len(records)} record")
    
    print("\nGood-quality (0.6 < score < 0.8):")
    for combo, records in good_quality_combinations.items():
        print(f"  {combo}: {len(records)} record")
    
    # Campiona i record
    print(f"\nCampionamento di {args.sample_size} record per combinazione (seed={args.seed})...")
    sampled_combinations, stats = sample_combinations(
        low_quality_combinations, 
        good_quality_combinations, 
        args.sample_size, 
        seed=args.seed
    )
    
    # Mostra statistiche del campionamento
    print("\nComposizione del campione:")
    for combo, stat in stats.items():
        total = stat["low_quality"] + stat["good_quality"]
        print(f"  {combo}: {total} totali ({stat['low_quality']} low-quality, {stat['good_quality']} good-quality)")
    
    # Crea cartella di output
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Scrivi i file Excel
    print(f"\nCreazione file Excel in '{args.output_dir}':")
    for combo, records in sampled_combinations.items():
        if records:
            df = pd.DataFrame(records)
            output_path = os.path.join(args.output_dir, f"{combo}.xlsx")
            df.to_excel(output_path, index=False)
            print(f"{combo}.xlsx ({len(records)} record)")
        else:
            print(f"{combo}: nessun record trovato")
    
    print("\nEstrazione completata!")


if __name__ == "__main__":
    main()
