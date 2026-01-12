"""Configurazione centrale per i path del progetto"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
OUTPUT_DIR = PROJECT_ROOT / "output"

MODELS = [

    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "google/gemma-3-4b-it",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-3-12b-it",
]

USE_QUANTIZATION = False

# Generation parameters
INFERENCE_BATCH_SIZE = 8  # Batch size for inference
NUM_GENERATIONS_PER_TRIPLE = 3  # Number of generations per triple
GENERATION_TEMPERATURE = 0.7  # Temperature for generation (>0 for variability)