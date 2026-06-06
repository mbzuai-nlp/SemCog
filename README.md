# SemCog - Arabic-Hebrew Cognate Recognition and Semantic Disambiguation

## Project Overview

SemCog is a research project studying how Large Language Models (LLMs) process **Arabic-Hebrew cognate relationships**. The project evaluates LLMs on two core tasks:

1. **Cognate Identification (CI)**: Classifying Arabic-Hebrew word pairs as `TRUE_COGNATE`, `FALSE_FRIEND`, or `LOANWORD` across multiple input formats (undiacritized, diacritized, uroman transliteration, IPA transcription).

2. **Semantic Disambiguation (SD)**: Determining whether Arabic/Hebrew sentences containing cognate/loanword pairs are semantically appropriate, using ABC-randomized multiple choice.

The project evaluates 13 LLMs: 9 open-source models (7B-9B) and 4 API models (GPT-4o, GPT-5.4, DeepSeek-V4, Qwen3.6-Plus).

## Directory Structure

```
submission/
├── README.md                          # This file
├── .env.example                       # Environment variable template
├── .gitignore                         # Git ignore rules
├── requirements.txt                   # Python dependencies
│
├── data/                              # Dataset files
│   ├── merged_dataset.json            # Core merged dataset (1,858 entries)
│   ├── 3class_undiac.jsonl            # Cognate ID eval data (3-class, 1,858 entries)
│   ├── 3class_diac.jsonl
│   ├── 3class_uroman.jsonl
│   ├── 3class_ipa.jsonl
│   ├── 2class_undiac.jsonl            # Cognate ID eval data (2-class, 1,222 entries)
│   ├── 2class_diac.jsonl
│   ├── 2class_uroman.jsonl
│   ├── 2class_ipa.jsonl
│   └── semantic_disambiguation/       # Semantic disambiguation eval data
│       ├── true_cognate/              # 858 entries × 4 formats
│       ├── false_friend/              # 728 entries × 4 formats
│       └── loanword/                  # 636 entries × 4 formats
│
├── prompts/                           # Prompt templates
│   ├── cognate_identification_2class_prompts.json
│   ├── cognate_identification_3class_prompts.json
│   ├── true_cognate_prompts.json
│   ├── false_friend_prompts.json
│   ├── loanword_prompts.json
│   ├── semantic_disambiguation_prompts.json
│   └── gemini_sentence_correction_prompt.md
│
├── scripts/                           # Utility scripts
│   ├── generate_ipa_loanwords.py      # IPA generation for loanwords
│   ├── generate_ipa_loanwords_v2.py   # IPA generation (incremental)
│   └── ipa_tool/
│       └── arabic_ipa.py              # Arabic-to-IPA converter
│
└── experiment/                        # Experiment scripts
    ├── cognate_identification/
    │   ├── evaluate_api_models.py     # API model evaluation
    │   ├── analyze_results.py         # Result analysis & metrics
    │   ├── prepare_eval_data.py       # Data preparation
    │   ├── run_all_experiments.py     # Experiment orchestrator
    │   └── tasks/                     # lm-eval-harness task configs
    │       ├── cognate_2class_diac.yaml
    │       ├── cognate_2class_ipa.yaml
    │       ├── cognate_2class_undiac.yaml
    │       ├── cognate_2class_uroman.yaml
    │       ├── cognate_3class_diac.yaml
    │       ├── cognate_3class_ipa.yaml
    │       ├── cognate_3class_undiac.yaml
    │       └── cognate_3class_uroman.yaml
    └── semantic_disambiguation/
    │       ├── run_api.py                 # API model evaluation
    │       ├── analyze_results.py         # Result analysis
    │       ├── run_lm_eval.py             # Open-source model evaluation
    │       └── convert_to_lmeval.py       # Data format conversion
    └── lm_eval/                           # lm-eval-harness configs for SD
        └── tasks/
            ├── true_cognate_disambiguation/  # 4 format yamls + template
            ├── false_friend_disambiguation/
            └── loanword_disambiguation/
```

## Environment Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up environment variables

Copy the example env file and fill in your API keys:

```bash
cp .env.example .env
# Edit .env with your actual API keys
```

Required environment variables for API model evaluation:
- `OPENAI_API_KEY` - for GPT-4o and GPT-5.4
- `DEEPSEEK_API_KEY` - for DeepSeek-V4
- `DASHSCOPE_API_KEY` - for Qwen3.6-Plus

### 3. (Optional) Install lm-evaluation-harness

For evaluating open-source models locally:

```bash
pip install lm-eval
```

## How to Run

### Cognate Identification

#### API Models

```bash
# Single model/format
cd experiment/cognate_identification
python evaluate_api_models.py \
    --model gpt-4o \
    --format undiac \
    --task_type 3class \
    --data ../../data/3class_undiac.jsonl \
    --output results/3class/undiac/gpt-4o.json

# All API experiments
python run_all_experiments.py --mode api

# Test mode (10 samples)
python run_all_experiments.py --mode api --test
```

#### Open-source Models (via lm-eval)

```bash
# Requires GPU and model weights
python run_all_experiments.py --mode local --model Qwen2.5-7B-Instruct --task_type 3class
```

#### Analyze Results

```bash
python analyze_results.py --task_type 3class
python analyze_results.py --task_type 2class --model gpt-4o
```

### Semantic Disambiguation

#### API Models

```bash
cd experiment/semantic_disambiguation

# Run all API experiments
python run_api.py

# Specific model/sub-experiment
python run_api.py --models gpt-4o --sub-experiments true_cognate

# Test mode
python run_api.py --limit 10
```

#### Open-source Models (via lm-eval)

```bash
# Convert data to lm-eval format first
python convert_to_lmeval.py

# Run evaluation (requires GPU)
python run_lm_eval.py --models Qwen2.5-7B-Instruct --dual-gpu
```

#### Analyze Results

```bash
python analyze_results.py
```

### Data Preparation

```bash
# Prepare cognate identification eval data from merged dataset
cd experiment/cognate_identification
python prepare_eval_data.py --input ../../data/merged_dataset.json --output_dir ../../data/
```

## Data Description

### Core Dataset (`merged_dataset.json`)

The merged dataset contains three categories of Arabic-Hebrew word pairs:

| Category | Description | Count |
|----------|-------------|-------|
| True Cognates | Words sharing a Semitic root with overlapping meanings | 858 |
| False Friends | Similar form but completely different meanings | 364 |
| Loanwords | Borrowed from a third language, similar form | 636 |

Each entry contains:
- Arabic and Hebrew words in multiple formats: undiacritized, diacritized, uroman, IPA
- Type label (TRUE_COGNATE, FALSE_FRIEND, LOANWORD)
- Semantic fields and example sentences

### Input Formats

| Format | Description | Example |
|--------|-------------|---------|
| `undiac` | Undiacritized script | كتاب / ספר |
| `diac` | Fully diacritized script | كِتَاب / סֵפֶר |
| `uroman` | Uroman transliteration | kitAb / sefer |
| `ipa` | IPA transcription | /kitaab/ / /sefer/ |

## Prompt Files

All prompt templates are in `prompts/`:

| File | Task | Description |
|------|------|-------------|
| `cognate_identification_3class_prompts.json` | CI | 3-class (TC/FF/LW) word-level prompts for 4 formats |
| `cognate_identification_2class_prompts.json` | CI | 2-class (TC/FF) word-level prompts for 4 formats |
| `true_cognate_prompts.json` | SD | True cognate sentence disambiguation |
| `false_friend_prompts.json` | SD | False friend sentence disambiguation |
| `loanword_prompts.json` | SD | Loanword sentence disambiguation |
| `semantic_disambiguation_prompts.json` | SD | General semantic disambiguation |
| `gemini_sentence_correction_prompt.md` | Data | Sentence correction prompt for data generation |

Each JSON prompt file contains:
- `task`: Task identifier
- `description`: Task description
- `labels`: Valid output labels
- `definitions`: Label definitions
- `prompts`: Per-format prompt templates with `{variable}` placeholders
- `data_fields`: Mapping from template variables to data fields

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall classification accuracy |
| **P / R / F1** | Per-class Precision, Recall, and F1 (macro-averaged) |
| **Per-class Accuracy** | Accuracy for each individual class |
| **LCR** | Loanword Confusion Rate (3-class only, used in Directionality Analysis) |
| **CUR_TC** | Cognate Under-Recognition Rate for True Cognates (3-class only, used in Directionality Analysis) |
| **CUR_FF** | Cognate Under-Recognition Rate for False Friends (3-class only, used in Directionality Analysis) |
| **SDB** | Semantic Drift Bias (used in Directionality Analysis) |
| **CLB** | Cognate–Loanword Directional Bias (3-class only, used in Directionality Analysis) |


## License

This project is for research purposes. Please cite appropriately if you use this dataset or methodology.
