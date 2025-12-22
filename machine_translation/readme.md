# Translation and Post-Editing Pipeline - Usage Guide

## Overview

The translation and post-editing pipeline can be executed in multiple ways, allowing you to:
- Run the complete pipeline from scratch
- Resume from a specific step using data from a previous run
- Test with mock models or use real models (GPU required)

## General Syntax

```bash
python main.py [OPTIONS]
```

## Available Options

### `--start-from {translation,xcomet,xtower,summary}`
Specifies which step to start execution from:
- `translation` (default): Start with DeepL translations
- `xcomet`: Start with XCOMET scoring (requires `--resume-from`)
- `xtower`: Start with xTower analysis (requires `--resume-from`)
- `summary`: Generate only the summary report (requires `--resume-from`)

### `--resume-from RUN_DIRECTORY`
Specifies the directory of a previous run from which to resume data.
Required when `--start-from` is different from `translation`.

Format: `run_YYYYMMDD_HHMMSS`

Example: `run_20241105_143000`

### `--test-mode`
Use mock models for testing (does not require GPU).
**Default: True**

### `--no-test-mode`
Use real models (requires GPU).

### `--test-limit N`
Number of records to process in test mode.
**Default: 1**

## Usage Examples

### 1. Run Complete Pipeline (from scratch)

```bash
# With mock models (default, does not require GPU)
python main.py

# With real models (requires GPU)
python main.py --no-test-mode

# With mock models and limit of 5 records
python main.py --test-limit 5
```

### 2. Resume from XCOMET

If you have already completed translations and want to run only XCOMET:

```bash
python main.py --start-from xcomet --resume-from run_20241105_143000
```

This will:
- Load data from `output/run_20241105_143000/unified_corpus.json`
- Execute XCOMET scoring
- Execute xTower analysis
- Generate summary report
- Save everything to a new directory `output/run_[NEW_TIMESTAMP]/`

### 3. Resume from xTower

If you already have XCOMET and want only xTower:

```bash
python main.py --start-from xtower --resume-from run_20241105_143000
```

### 4. Regenerate Only the Summary Report

If you have all data and want to regenerate only the report:

```bash
python main.py --start-from summary --resume-from run_20241105_143000
```

### 5. Use Real Models with Resume

```bash
python main.py --start-from xcomet --resume-from run_20241105_143000 --no-test-mode
```

## Output

Each execution creates a new output directory:
```
output/
  run_20241105_143000/          # Previous run
    unified_corpus.json
    unified_corpus.csv
    summary_report.txt
    pipeline.log
  run_20241105_150000/          # New run (resume)
    unified_corpus.json
    unified_corpus.csv
    summary_report.txt
    pipeline.log                # Includes info about which run was resumed from
```

## Log Files

The `pipeline.log` file in the new run directory will contain:
- Which step execution was started from
- Which previous run data was loaded from (if applicable)
- All details of the current execution

Example log header when resuming from a previous run:
```
================================================================================
MULTILINGUAL TRANSLATION AND POST-EDITING PIPELINE
================================================================================
Start time: 2024-11-05 15:00:00
Run output directory: c:\...\output\run_20241105_150000
Starting from step: xcomet
Resuming from run: run_20241105_143000
Resume directory: c:\...\output\run_20241105_143000
Test Mode: True
Test Records Limit: 1
```

## Typical Workflows

### Scenario 1: Development and Testing
```bash
# 1. Test with a few records
python main.py --test-limit 3

# 2. If OK, run with more records
python main.py --test-limit 100

# 3. If needed, restart from XCOMET with different parameters
python main.py --start-from xcomet --resume-from run_20241105_143000
```

### Scenario 2: Full Production Run
```bash
# 1. Run translations (may take time)
python main.py --no-test-mode --start-from translation

# 2. If an error occurs during XCOMET, restart from there
python main.py --no-test-mode --start-from xcomet --resume-from run_20241105_143000

# 3. If an error occurs during xTower, restart from there
python main.py --no-test-mode --start-from xtower --resume-from run_20241105_160000
```

## Argument Validation

The system automatically validates:
- If `--start-from` is `xcomet`, `xtower` or `summary`, requires `--resume-from`
- If `--resume-from` is specified, verifies that the directory exists
- If the directory exists, verifies that it contains `unified_corpus.json`

## Important Notes

1. **Each execution creates a new run directory**: even when resuming from a previous run, a new directory with updated timestamp is created.

2. **Original data is not modified**: the previous run remains intact, allowing you to retry if necessary.

3. **Test mode vs Production**: In test mode, mock models are used that simulate results without requiring GPU. For real results, use `--no-test-mode`.

4. **GPU memory management**: When using `--no-test-mode`, models are loaded sequentially and cleaned from memory after use to avoid GPU memory exhaustion.
