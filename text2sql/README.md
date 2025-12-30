# Text-to-SQL Fine-tuning Project

This project implements a fine-tuned text-to-SQL model using Qwen2.5-1.5B-Instruct with LoRA (Low-Rank Adaptation) for efficient parameter-efficient fine-tuning.

## Project Structure

### 📁 Root Files

- **`evaluate_models_comparison.py`**: Comprehensive evaluation script that compares base models, fine-tuned models, and alternative models (like Llama3). Evaluates on unbiased test sets with automatic API fallback and saves results with model predictions.

### 📁 `src/` - Source Code

Core implementation files for dataset creation, training, and evaluation:

- **`phase3_train_lora_optimized.py`**: Main training script for LoRA fine-tuning with balanced datasets. Features optimized hyperparameters, enhanced LoRA configuration, and schema pruning support.

- **`phase4_semantic_sql_score.py`**: Semantic SQL evaluation module that measures similarity between predicted and ground truth SQL queries using keyword and clause matching.

- **`create_hybrid_dataset.py`**: Creates balanced hybrid datasets by combining BIRD dataset examples with synthetic augmentation. Ensures balanced distribution across different query types (anti-joins, multi-table joins, etc.).

- **`schema_pruning.py`**: Schema pruning utility that reduces schema complexity by keeping only relevant tables/columns based on question tokens. Helps reduce memorization burden on small LoRA ranks while preserving PK/FK relationships.

- **`reformat_schema_clarity.py`**: Reformats dataset schemas from compressed format to a clearer, more readable structure that helps models better understand table relationships, data types, and key constraints.

### 📁 `configs/` - Configuration Files

- **`sft_qwen.yaml`**: Configuration file for supervised fine-tuning of Qwen models. Contains settings for LoRA parameters, training hyperparameters, data paths, and inference settings.

- **`rag.yaml`**: Configuration for RAG (Retrieval-Augmented Generation) based approaches (if applicable).

### 📁 `data/` - Datasets

- **`bird_jsonl/`**: Contains BIRD dataset in JSONL format:
  - `train.jsonl`: Training dataset
  - `dev.jsonl`: Development/validation dataset

### 📁 `outputs/` - Output Directory

- **`checkpoints/`**: Contains fine-tuned model checkpoints:
  - Multiple LoRA checkpoints with different configurations (1k, 5k, optimized variants)
  - Each checkpoint includes adapter weights, tokenizer files, and configuration

- **`jsonl/`**: Generated datasets and evaluation results:
  - `bird_train.jsonl`: Processed training data
  - `bird_semantic_balanced.jsonl`: Balanced semantic dataset
  - `bird_spider_5k_balanced.jsonl`: Balanced 5k dataset
  - `comprehensive_test_set.jsonl`: Comprehensive test dataset
  - `unbiased_test_set.jsonl`: Unbiased test set for evaluation
  - `final_evaluation_100.json`: Final evaluation results

- **`evaluation/`**: Detailed evaluation outputs and metrics

- **`artifacts/`**: Additional training artifacts

- **`serving/`**: Model serving configurations and files

- **`evaluation_results.json`**: JSON format evaluation results
- **`evaluation_results.txt`**: Text format evaluation results
- **`training_log.txt`**: Training logs and metrics

### 📁 `cards/` - Model Cards

Contains model cards and documentation for trained models.

### 📁 `eval/` - Evaluation Scripts

Additional evaluation utilities and scripts.

## Key Features

1. **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning using Low-Rank Adaptation
2. **Schema Pruning**: Reduces schema complexity to improve model performance
3. **Balanced Datasets**: Ensures balanced distribution across different SQL query types
4. **Semantic Evaluation**: Advanced evaluation metrics that consider SQL semantics, not just exact match
5. **Model Comparison**: Comprehensive evaluation comparing base, fine-tuned, and alternative models

## Usage

### Training

```bash
python src/phase3_train_lora_optimized.py
```

### Evaluation

```bash
python evaluate_models_comparison.py
```

### Dataset Creation

```bash
# Create hybrid balanced dataset
python src/create_hybrid_dataset.py

# Reformat schema for clarity
python src/reformat_schema_clarity.py
```

## Model Checkpoints

The repository includes several pre-trained checkpoints:
- `qwen_1.5b_text2sql_lora_1k_r16`: 1k examples, rank 16
- `qwen_1.5b_text2sql_lora_5k_r16`: 5k examples, rank 16
- `qwen_1.5b_text2sql_lora_final`: Final optimized checkpoint
- `qwen_1.5b_text2sql_lora_optimized`: Optimized variant
- `qwen_1.5b_text2sql_lora_optimized_2k`: 2k optimized variant

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- PEFT (for LoRA)
- Other dependencies as specified in requirements

## Notes

- The virtual environment (`text2sql_env/`) is excluded from the repository
- Model checkpoints are included but may be large files
- All datasets are in JSONL format for efficient processing

