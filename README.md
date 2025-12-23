# TailNLG Benchmark

This repository collects resources, data processing scripts, and experiments for the TailNLG benchmark — a suite focused on evaluating natural language generation and translation models on long-tail (rare/low-frequency) phenomena.


## Repository structure

- [TailNLG Dataset](https://github.com/liadraetta/TailNLG-benchmark/tree/6648416436351fb8c356d2ba3f55122858fbeef5/TailNLG%20Dataset)  
  Dataset folder

- [Extraction](https://github.com/liadraetta/TailNLG-benchmark/tree/6648416436351fb8c356d2ba3f55122858fbeef5/Extraction)  
  Data extraction and preprocessing scripts. Use these to generate dataset subsets, cleaning, and feature extraction pipelines.

- [machine_translation](https://github.com/liadraetta/TailNLG-benchmark/tree/6648416436351fb8c356d2ba3f55122858fbeef5/machine_translation)  
  Scripts, configurations, and possibly model checkpoints for machine translation experiments included in the benchmark.

- [Experiments](https://github.com/liadraetta/TailNLG-benchmark/tree/6648416436351fb8c356d2ba3f55122858fbeef5/Experiments)  
  Experiment definitions, evaluation scripts, logs, and results. Reproducible training/evaluation pipelines and hyperparameter records should live here.

  
 # TailNLG Benchmark Documentation

A curated XML dataset for evaluating data-to-text generation and translation on long-tail / rare facts.
Each entry pairs is a sets of RDF-like triples with one or more human lexicalizations in multiple languages (English, Spanish, Italian) labeled by quality (gold / silver). 
Entries cover many semantic categories (Artist, City, Monument, Company, Astronaut, etc.), triple-set sizes, and graph shapes to test generation complexity and robustness.

## Why it’s useful

Designed to benchmark multilingual NLG (verbalization of KG triples), robustness on rare/long-tail entities and relations, and translation/localization of generated text.


## Labels

- **category**: high-level domain/class for the subject (e.g., Artist, City, Company, Monument).
- **eid**:internal entry id (e.g., Id1).
- **shape**, shape_type: indicates KG connectivity pattern: chain, sibling, mixed (used to vary reasoning / aggregation complexity).
- **size**: number of triples in the (original/modified) tripleset.
- **unique_id**: internal unique token (alphanumeric).
- **qid**: Wikidata Q‑ID linking the subject to an external KB when available.
- **type**: coarse difficulty or frequency bucket (e.g., long_tail, top_head).
- **subtype**: more specific designation (e.g., long_tail_en, long_tail_it, rare_claims, top_head).
triplesets
- **originaltripleset**: source triples (subject | predicate | object) as collected.
- **modifiedtripleset**: cleaned / normalized triples (used for generation input).

