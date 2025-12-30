#!/usr/bin/env python3
"""
Phase 3 Optimized: LoRA Fine-tuning with Balanced Dataset

Improvements:
- Uses balanced dataset covering all query types
- Optimized hyperparameters for better convergence
- Enhanced LoRA configuration for complex patterns
- Better evaluation and checkpointing
- Schema pruning to reduce memorization burden (optional)
"""

import json
import torch
import time
from pathlib import Path
from typing import List, Dict
from datetime import datetime

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
from datasets import Dataset
import logging

# Import schema pruning
from schema_pruning import prune_schema

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"
JSONL_DIR = OUTPUTS / "jsonl"
CHECKPOINT_DIR = OUTPUTS / "checkpoints"
REPORTS_DIR = OUTPUTS / "reports"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# OPTIMIZED LORA CONFIGURATION
# ============================================================================

class OptimizedLoRAConfig:
    """Optimized LoRA configuration for comprehensive SQL generation"""
    
    # Model
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    model_max_length = 512
    
    # LoRA parameters - r=16 for better capacity
    lora_r = 16  # Doubled rank for 2x capacity (fewer detail errors)
    lora_alpha = 32  # Maintain 2.0 scaling ratio (32/16 = 2.0)
    lora_dropout = 0.05  # Reduced dropout (more data = less overfitting)
    lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # All attention projections
    
    # Training - optimized for 4668 examples with r=16
    num_train_epochs = 1  # CPU training: 1 epoch = ~30 hours (was 3 = 98 hours)
    per_device_train_batch_size = 2  # Increased from 1 (faster training)
    gradient_accumulation_steps = 2  # Reduced (effective batch=4)
    learning_rate = 3e-5  # Reduced from 5e-5 for larger dataset
    warmup_ratio = 0.15  # Increased warmup for stability
    max_steps = -1
    
    # Logging and checkpointing
    logging_steps = 10
    eval_steps = 100
    save_steps = 100
    save_total_limit = 5  # Keep more checkpoints
    
    # Optimization
    optim = "adamw_torch"
    weight_decay = 0.01
    max_grad_norm = 1.0
    lr_scheduler_type = "cosine"
    
    # Dataset
    use_balanced_dataset = True
    balanced_file = "bird_spider_5k_balanced.jsonl"  # 4668 balanced BIRD+Spider examples
    dev_subset_size = 200  # Increased for better evaluation
    
    # Schema pruning (reduces complexity, helps small LoRA ranks)
    use_schema_pruning = False  # DISABLED - dataset already has good schemas
    schema_min_tables = 1
    schema_max_columns = 10


# ============================================================================
# DATA LOADING
# ============================================================================

def load_jsonl(file_path: Path) -> List[Dict]:
    """Load JSONL file"""
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def prepare_optimized_datasets(config: OptimizedLoRAConfig):
    """Load balanced training dataset with proper train/dev split"""
    logger.info("=" * 70)
    logger.info("LOADING BALANCED DATASET")
    logger.info("=" * 70)
    
    # Load balanced training data
    if config.use_balanced_dataset:
        train_path = JSONL_DIR / config.balanced_file
        if not train_path.exists():
            logger.error(f"Balanced dataset not found: {train_path}")
            logger.error("Please run create_balanced_dataset.py first!")
            raise FileNotFoundError(f"Balanced dataset not found: {train_path}")
    else:
        train_path = JSONL_DIR / "bird_train.jsonl"
    
    logger.info(f"Loading training data: {train_path}")
    all_data = load_jsonl(train_path)
    logger.info(f"Loaded {len(all_data)} total examples from balanced dataset")
    
    # Split balanced dataset: use last dev_subset_size examples for validation
    # This ensures dev set has same quality/balance as training set
    dev_data = all_data[-config.dev_subset_size:]
    train_data = all_data[:-config.dev_subset_size]
    
    logger.info(f"Split into:")
    logger.info(f"  Training: {len(train_data)} examples")
    logger.info(f"  Dev (from same balanced set): {len(dev_data)} examples")
    
    return train_data, dev_data


# ============================================================================
# MODEL SETUP
# ============================================================================

def setup_optimized_lora_model(config: OptimizedLoRAConfig):
    """Load model with optimized LoRA configuration"""
    logger.info("=" * 70)
    logger.info("SETTING UP OPTIMIZED LORA MODEL")
    logger.info("=" * 70)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    logger.info(f"Loading base model: {config.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    
    # Configure optimized LoRA
    logger.info("Applying optimized LoRA configuration...")
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    logger.info(f"LoRA Config:")
    logger.info(f"  Rank (r): {config.lora_r}")
    logger.info(f"  Alpha: {config.lora_alpha}")
    logger.info(f"  Scaling: {config.lora_alpha/config.lora_r}")
    logger.info(f"  Target modules: {config.lora_target_modules}")
    logger.info(f"  Dropout: {config.lora_dropout}")
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_mb = trainable_params * 4 / (1024 * 1024)  # 4 bytes per float32
    
    logger.info(f"\nParameter Statistics:")
    logger.info(f"  Trainable params: {trainable_params:,} ({trainable_mb:.2f} MB)")
    logger.info(f"  Total params: {total_params:,}")
    logger.info(f"  Trainable %: {100 * trainable_params / total_params:.4f}%")
    
    model.print_trainable_parameters()
    
    return model, tokenizer


# ============================================================================
# DATASET FORMATTING
# ============================================================================

def format_prompt(example: Dict) -> str:
    """Format example into training prompt - CORRECTED FORMAT
    
    CRITICAL: SQL should NOT be in input prompt!
    Model must learn to GENERATE SQL from question+schema, not copy it.
    
    Returns prompt WITHOUT SQL answer (for inference/input)
    """
    question = example.get('question', '')
    schema = example.get('db_schema', '')
    
    # INPUT ONLY - no SQL answer included
    prompt = f"""[INSTRUCTION]
You are a SQL expert. Generate a valid SQL query based on the question and database schema.

The schema uses this format:
- Each table: TableName(col1 TYPE, col2 TYPE; PRIMARY KEY(...); FK(col REFERENCES OtherTable(col)))
- Tables are separated by newlines
- Use only the tables and columns shown in the schema

[DATABASE SCHEMA]
{schema}

[QUESTION]
{question}

[SQL QUERY]
"""
    
    return prompt


def format_training_example(example: Dict, config: OptimizedLoRAConfig) -> Dict:
    """Format example for training with proper input/output separation
    
    IMPORTANT: Schema format is preserved EXACTLY as in dataset.
    No conversions applied - model learns to parse compressed format directly.
    db_id field is metadata only (not used in prompt).
    
    Optionally applies schema pruning to reduce complexity.
    
    Returns dict with:
    - 'input': prompt without SQL (what model sees)
    - 'output': just the SQL (what model should generate)
    - 'full_text': input + output for tokenization
    """
    question = example.get('question', '')
    schema = example.get('db_schema', '')  # Used AS-IS from dataset
    sql = example.get('SQL', '')
    # db_id = example.get('db_id', '')  # Metadata only, not used in training prompt
    
    # Apply schema pruning if enabled
    if config.use_schema_pruning:
        schema = prune_schema(
            schema, 
            question, 
            min_tables=config.schema_min_tables,
            max_columns_per_table=config.schema_max_columns
        )
    
    # Input part (no SQL)
    input_text = f"""[INSTRUCTION]
You are a SQL expert. Generate a valid SQL query based on the question and database schema.

The schema uses this format:
- Each table: TableName(col1 TYPE, col2 TYPE; PRIMARY KEY(...); FK(col REFERENCES OtherTable(col)))
- Tables are separated by newlines
- Use only the tables and columns shown in the schema

[DATABASE SCHEMA]
{schema}

[QUESTION]
{question}

[SQL QUERY]
"""
    
    # Output part (just SQL)
    output_text = sql
    
    # Full sequence for training
    full_text = input_text + output_text
    
    return {
        'input': input_text,
        'output': output_text,
        'full_text': full_text
    }


def preprocess_function(example: Dict, tokenizer, config: OptimizedLoRAConfig):
    """Preprocess single example for training with CORRECTED label masking
    
    Key fix: Only compute loss on SQL generation tokens, not input tokens.
    This teaches model to GENERATE SQL, not memorize input patterns.
    Optionally applies schema pruning.
    """
    # Get input/output split (with optional schema pruning)
    formatted = format_training_example(example, config)
    input_text = formatted['input']
    full_text = formatted['full_text']
    
    # Tokenize input portion (for masking)
    input_tokens = tokenizer(
        input_text,
        truncation=False,
        padding=False,
        return_tensors=None
    )
    input_length = len(input_tokens['input_ids'])
    
    # Tokenize full sequence (input + output)
    model_inputs = tokenizer(
        full_text,
        truncation=True,
        max_length=config.model_max_length,
        padding=False,
        return_tensors=None,
        add_special_tokens=True
    )
    
    # Create labels with input tokens masked
    # -100 = ignore in loss calculation (input portion)
    # actual token IDs = compute loss (SQL output portion)
    labels = list(model_inputs["input_ids"])  # Ensure it's a plain list
    for i in range(min(input_length, len(labels))):
        labels[i] = -100  # Mask input tokens
    
    # Return all as plain Python lists (not nested)
    return {
        "input_ids": list(model_inputs["input_ids"]),
        "attention_mask": list(model_inputs.get("attention_mask", [1] * len(model_inputs["input_ids"]))),
        "labels": labels
    }


# ============================================================================
# TRAINING
# ============================================================================

def train_optimized_model():
    """Main training function with optimized configuration"""
    
    print("\n" + "=" * 70)
    print("PHASE 3 OPTIMIZED: LoRA FINE-TUNING")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    start_time = time.time()
    config = OptimizedLoRAConfig()
    
    # Load datasets
    train_data, dev_data = prepare_optimized_datasets(config)
    
    # Setup model
    model, tokenizer = setup_optimized_lora_model(config)
    
    # Create HuggingFace datasets
    logger.info("\nCreating HuggingFace datasets...")
    train_dataset = Dataset.from_list(train_data)
    dev_dataset = Dataset.from_list(dev_data)
    
    # Preprocess
    logger.info("Preprocessing datasets...")
    
    def tokenize_function(example):
        return preprocess_function(example, tokenizer, config)
    
    train_dataset = train_dataset.map(
        tokenize_function,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train"
    )
    
    dev_dataset = dev_dataset.map(
        tokenize_function,
        remove_columns=dev_dataset.column_names,
        desc="Tokenizing dev"
    )
    
    # Custom data collator that pads labels with -100
    from transformers import default_data_collator
    from dataclasses import dataclass
    from typing import Any, Dict, List
    import torch
    
    @dataclass
    class DataCollatorForSeq2Seq:
        """Data collator that pads labels with -100"""
        tokenizer: Any
        padding: bool = True
        max_length: int = None
        pad_to_multiple_of: int = 8
        
        def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
            # Separate labels from inputs
            labels = [feature["labels"] for feature in features] if "labels" in features[0] else None
            
            # Remove labels from features for padding
            features_without_labels = [{k: v for k, v in f.items() if k != "labels"} for f in features]
            
            # Pad input_ids and attention_mask
            batch = self.tokenizer.pad(
                features_without_labels,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt"
            )
            
            # Pad labels manually with -100
            if labels is not None:
                max_label_length = max(len(l) for l in labels)
                if self.pad_to_multiple_of is not None:
                    max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                    )
                
                padded_labels = []
                for label in labels:
                    remainder = [-100] * (max_label_length - len(label))
                    padded_labels.append(label + remainder)
                
                batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
            
            return batch
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8
    )
    
    # Training arguments
    output_dir = CHECKPOINT_DIR / f"qwen_5k_r16_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        eval_strategy="steps",
        save_strategy="steps",
        optim=config.optim,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        fp16=False,
        logging_dir=str(output_dir / "logs"),
        report_to="none",
        gradient_checkpointing=False,  # Disabled - conflicts with LoRA
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("\n" + "=" * 70)
    logger.info("STARTING OPTIMIZED TRAINING")
    logger.info("=" * 70)
    
    train_result = trainer.train()
    
    # Save final model
    final_model_dir = CHECKPOINT_DIR / "qwen_1.5b_text2sql_lora_5k_r16"
    logger.info(f"\nSaving final optimized model to: {final_model_dir}")
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    
    # Training report
    training_time = time.time() - start_time
    
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'configuration': {
            'model': config.model_name,
            'lora_rank': config.lora_r,
            'lora_alpha': config.lora_alpha,
            'lora_modules': config.lora_target_modules,
            'epochs': config.num_train_epochs,
            'learning_rate': config.learning_rate,
            'batch_size': config.per_device_train_batch_size,
            'gradient_accumulation': config.gradient_accumulation_steps,
        },
        'dataset': {
            'train_size': len(train_data),
            'dev_size': len(dev_data),
            'balanced': config.use_balanced_dataset,
        },
        'training_results': {
            'total_time_minutes': training_time / 60,
            'final_train_loss': train_result.metrics.get('train_loss'),
            'final_eval_loss': train_result.metrics.get('eval_loss'),
        },
        'model_output': str(final_model_dir),
    }
    
    report_path = REPORTS_DIR / f"optimized_training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Total time: {training_time/60:.2f} minutes")
    print(f"Final model: {final_model_dir}")
    print(f"Report: {report_path}")
    print("=" * 70)
    
    print("\nNext steps:")
    print("1. Run evaluation: python src/phase4_evaluate_lora.py")
    print("2. Test interactively: streamlit run src/demo_streamlit.py")
    print("3. Deploy: python src/phase5_merge_lora.py")


if __name__ == "__main__":
    train_optimized_model()
