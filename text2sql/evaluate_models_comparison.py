#!/usr/bin/env python3
"""
Comprehensive Model Evaluation: Base vs Fine-tuned vs Llama3

Evaluates on unbiased test set with automatic API fallback.
Saves results back to JSONL with model predictions.
"""

import json
import os
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datetime import datetime
import requests

# Add src to path for schema pruning
sys.path.insert(0, str(Path(__file__).parent / "src"))
from schema_pruning import prune_schema

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
JSONL_DIR = PROJECT_ROOT / "outputs" / "jsonl"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"

# Models
BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Use the best checkpoint from 5k training (checkpoint-3000)
FINETUNED_MODEL_PATH = CHECKPOINT_DIR / "qwen_5k_r16_20251225_083035" / "checkpoint-3000"

# Test dataset (use unbiased if available, otherwise comprehensive)
UNBIASED_TEST_FILE = JSONL_DIR / "unbiased_test_set.jsonl"
COMPREHENSIVE_TEST_FILE = JSONL_DIR / "comprehensive_test_set.jsonl"

# Schema pruning configuration (PICARD-inspired, less strict)
USE_SCHEMA_PRUNING = True  # Enable schema pruning for fine-tuned model
SCHEMA_MIN_TABLES = 2      # Keep at least 2 tables (not too aggressive)
SCHEMA_MAX_COLUMNS = 12    # Max 12 columns per table (generous limit)

# Ollama configuration
OLLAMA_MODEL = "llama3:latest"
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_URL = f"{OLLAMA_BASE}/api/generate"

# Evaluation configuration
MAX_TEST_SAMPLES = 100  # Evaluate exactly 100 queries

# ============================================================================
# MODEL LOADING
# ============================================================================

class ModelManager:
    """Manages base, fine-tuned, and Gemini models"""
    
    def __init__(self):
        self.base_model = None
        self.base_tokenizer = None
        self.finetuned_model = None
        self.finetuned_tokenizer = None
        self.llama_available = False
        
    def load_base_model(self):
        """Load base Qwen model"""
        print("\n[1/3] Loading base model...")
        self.base_tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_NAME,
            trust_remote_code=True
        )
        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
            
        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        self.base_model.eval()
        print("  ✓ Base model loaded")
    
    def load_finetuned_model(self):
        """Load fine-tuned LoRA model"""
        print("\n[2/3] Loading fine-tuned model...")
        
        if FINETUNED_MODEL_PATH is None or not FINETUNED_MODEL_PATH.exists():
            print(f"  ✗ Fine-tuned model not found at {FINETUNED_MODEL_PATH}")
            print(f"  Skipping fine-tuned model evaluation")
            return False
        
        # Check if adapter files exist
        adapter_file = FINETUNED_MODEL_PATH / "adapter_model.safetensors"
        adapter_config = FINETUNED_MODEL_PATH / "adapter_config.json"
        if not adapter_file.exists() and not adapter_config.exists():
            print(f"  ✗ Adapter files not found in {FINETUNED_MODEL_PATH}")
            print(f"  Expected: adapter_model.safetensors or adapter_config.json")
            print(f"  Skipping fine-tuned model evaluation")
            return False
        
        # Load tokenizer from BASE model (not checkpoint) - this is correct for LoRA
        print(f"  Loading tokenizer from base model...")
        self.finetuned_tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_NAME,
            trust_remote_code=True
        )
        if self.finetuned_tokenizer.pad_token is None:
            self.finetuned_tokenizer.pad_token = self.finetuned_tokenizer.eos_token
        
        # Load base model first
        print(f"  Loading base model...")
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Load LoRA weights from checkpoint
        print(f"  Loading LoRA adapter from {FINETUNED_MODEL_PATH.name}...")
        self.finetuned_model = PeftModel.from_pretrained(
            base,
            str(FINETUNED_MODEL_PATH)
        )
        self.finetuned_model.eval()
        checkpoint_name = f"{FINETUNED_MODEL_PATH.parent.name}/{FINETUNED_MODEL_PATH.name}"
        print(f"  ✓ Fine-tuned model loaded from {checkpoint_name} (5K dataset, best checkpoint)")
        return True
    
    def load_llama(self):
        """Check Llama3 availability via Ollama (reliable)"""
        print("\n[3/3] Setting up Llama3 (Ollama)...")

        base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        tags_url = f"{base}/api/tags"
        gen_url  = f"{base}/api/generate"

        try:
            # 1) Verify server is up
            tags = requests.get(tags_url, timeout=5).json()
            names = {m.get("name") for m in tags.get("models", [])}

            # 2) Verify model is installed
            if OLLAMA_MODEL not in names:
                print(f"  ✗ Model not installed: {OLLAMA_MODEL}")
                print(f"  Installed models: {sorted(n for n in names if n)}")
                print(f"  Fix: ollama pull {OLLAMA_MODEL}")
                self.llama_available = False
                return

            # 3) Warm-up generate (give it a realistic timeout)
            r = requests.post(
                gen_url,
                json={"model": OLLAMA_MODEL, "prompt": "Test", "stream": False},
                timeout=30
            )
            r.raise_for_status()

            self.llama_available = True
            print(f"  ✓ Llama3 ({OLLAMA_MODEL}) available")

        except Exception as e:
            self.llama_available = False
            print(f"  ✗ Llama3 check failed: {repr(e)}")
            print(f"  Try: {tags_url}")
            print(f"  And: ollama list")
    
    def generate_sql(self, model_type: str, prompt: str, original_schema: str = None, question: str = None, use_pruning: bool = False) -> Tuple[str, str]:
        """Generate SQL with optional schema pruning
        
        Args:
            model_type: "base", "finetuned", or "llama"
            prompt: Formatted prompt
            original_schema: Original full schema for pruning
            question: Question text for pruning
            use_pruning: Whether to apply schema pruning (for fine-tuned model)
        
        Returns: (generated_sql, model_used)
        """
        # Apply schema pruning if requested
        effective_prompt = prompt
        if use_pruning and original_schema and question:
            pruned_schema = prune_schema(
                original_schema, 
                question,
                min_tables=SCHEMA_MIN_TABLES,
                max_columns_per_table=SCHEMA_MAX_COLUMNS
            )
            effective_prompt = format_prompt(question, pruned_schema)
        
        if model_type == "base":
            return self._generate_local(self.base_model, self.base_tokenizer, prompt), "Base Qwen"
        
        elif model_type == "finetuned":
            if self.finetuned_model is None:
                return "SKIPPED: Model not loaded", "Fine-tuned (unavailable)"
            return self._generate_local(self.finetuned_model, self.finetuned_tokenizer, effective_prompt), "Fine-tuned"
        
        elif model_type == "llama":
            return self._generate_llama(prompt)
        
        return "", "Unknown"
    
    def _generate_local(self, model, tokenizer, prompt: str) -> str:
        """Generate SQL from local model"""
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,  # Increased to allow full SQL queries
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract SQL after prompt
            sql = full_response[len(prompt):].strip() if len(full_response) > len(prompt) else full_response
            
            # Clean up
            if "```sql" in sql:
                sql = sql.split("```sql")[1].split("```")[0].strip()
            elif "```" in sql:
                sql = sql.split("```")[1].split("```")[0].strip()
            
            # Remove trailing whitespace and normalize newlines
            sql = sql.strip()
            
            return sql
        
        except Exception as e:
            return f"ERROR: {str(e)[:50]}"
    
    def _generate_llama(self, prompt: str) -> Tuple[str, str]:
        """Generate SQL from Llama3 via Ollama"""
        if not self.llama_available:
            return "ERROR: Llama3 not available", "Llama3 (unavailable)"
        
        try:
            response = requests.post(OLLAMA_URL, json={
                "model": OLLAMA_MODEL,
                "prompt": prompt + "\n\nGenerate ONLY the SQL query, nothing else.",
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 512
                }
            }, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                sql = result.get("response", "").strip()
                
                # Clean up markdown
                if "```sql" in sql:
                    sql = sql.split("```sql")[1].split("```")[0].strip()
                elif "```" in sql:
                    sql = sql.split("```")[1].split("```")[0].strip()
                
                return sql, "Llama3"
            else:
                return f"ERROR: Status {response.status_code}", "Llama3 (error)"
        
        except Exception as e:
            return f"ERROR: {str(e)[:50]}", "Llama3 (error)"


# ============================================================================
# DATASET LOADING
# ============================================================================

def load_test_set() -> List[Dict]:
    """Load test set (prefer comprehensive with schemas, fallback to unbiased)"""
    
    # Prefer comprehensive test set (has all schemas)
    if COMPREHENSIVE_TEST_FILE.exists():
        print(f"  Loading comprehensive test set from {COMPREHENSIVE_TEST_FILE.name}...")
        test_data = []
        with open(COMPREHENSIVE_TEST_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    test_data.append(json.loads(line))
        
        # Filter to only samples with valid schemas
        filtered_data = []
        for sample in test_data:
            schema = sample.get('db_schema', '')
            if schema and schema != '/* Schema not available */' and len(schema.strip()) > 10:
                filtered_data.append(sample)
        
        print(f"  ✓ Loaded {len(test_data)} examples from comprehensive test set")
        print(f"  ✓ Filtered to {len(filtered_data)} examples with valid schemas")
        print(f"  ⚠ WARNING: May have schema overlap with training data")
        return filtered_data
    
    # Fallback to unbiased test set
    if UNBIASED_TEST_FILE.exists():
        print(f"  ⚠ Comprehensive test set not found")
        print(f"  Loading unbiased test set from {UNBIASED_TEST_FILE.name}...")
        test_data = []
        with open(UNBIASED_TEST_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    test_data.append(json.loads(line))
        
        # Filter to only samples with valid schemas
        filtered_data = []
        for sample in test_data:
            schema = sample.get('db_schema', '')
            if schema and schema != '/* Schema not available */' and len(schema.strip()) > 10:
                filtered_data.append(sample)
        
        print(f"  ✓ Loaded {len(test_data)} examples from UNBIASED test set")
        print(f"  ✓ Filtered to {len(filtered_data)} examples with valid schemas")
        print(f"  ✓ NO OVERLAP with training data - TRUE GENERALIZATION TEST")
        return filtered_data
    
    print(f"  ✗ No test set found!")
    print(f"  Please run: python create_unbiased_test_set.py or create_comprehensive_test_set.py")
    return []


# ============================================================================
# EVALUATION
# ============================================================================

def format_prompt(question: str, schema: str = None) -> str:
    """Format prompt for model (handles missing schema)"""
    if schema:
        return f"""[INSTRUCTION]
You are a SQL expert. Generate a valid SQL query based on the question and database schema.

The schema format:
- Tables: TableName(col1 TYPE, col2 TYPE; PRIMARY KEY(...); FK(col REFERENCES OtherTable(col)))

[DATABASE SCHEMA]
{schema}

[QUESTION]
{question}

[SQL QUERY]
"""
    else:
        return f"""[INSTRUCTION]
You are a SQL expert. Generate a valid SQL query based on the question.

[QUESTION]
{question}

[SQL QUERY]
"""


def simple_sql_match(pred: str, ground_truth: str) -> bool:
    """Simple SQL comparison (normalized)"""
    def normalize(sql):
        return ' '.join(sql.upper().split()).strip()
    
    return normalize(pred) == normalize(ground_truth)


def evaluate_models(manager: ModelManager, test_samples: List[Dict]) -> List[Dict]:
    """Evaluate all models on test samples"""
    results = []
    
    print("\n" + "=" * 100)
    print("EVALUATION STARTED")
    print("=" * 100)
    
    for idx, sample in enumerate(test_samples, 1):
        question = sample.get('question', '')
        schema = sample.get('db_schema', '')
        ground_truth = sample.get('SQL', '')
        query_type_name = sample.get('query_type_name', sample.get('query_type', 'Unknown'))
        
        prompt = format_prompt(question, schema)
        
        print(f"\n{'='*100}")
        print(f"[{idx}/{len(test_samples)}] {query_type_name}")
        print(f"{'='*100}")
        print(f"Question: {question}")
        print(f"\n� Schema:")
        print(f"  {schema}")
        print(f"\n�💎 Ground Truth SQL:")
        print(f"  {ground_truth}")
        
        # Test base model (without pruning - baseline)
        print(f"\n[🔵 Base Model] Generating...")
        print(f"  Using FULL schema (baseline)")
        base_sql, _ = manager.generate_sql("base", prompt, original_schema=schema, question=question)
        base_correct = simple_sql_match(base_sql, ground_truth)
        print(f"  Predicted: {base_sql}")
        print(f"  {'✅ CORRECT' if base_correct else '❌ WRONG'}")
        
        # Test fine-tuned model WITHOUT schema pruning
        print(f"\n[🟢 Fine-tuned Model (NO Pruning)] Generating...")
        print(f"  Using FULL schema (no pruning)")
        ft_no_prune_sql, _ = manager.generate_sql("finetuned", prompt, original_schema=schema, question=question, use_pruning=False)
        ft_no_prune_correct = simple_sql_match(ft_no_prune_sql, ground_truth) if not ft_no_prune_sql.startswith("SKIPPED") else None
        print(f"  Predicted: {ft_no_prune_sql}")
        print(f"  {'✅ CORRECT' if ft_no_prune_correct else '⚠️ SKIPPED' if ft_no_prune_correct is None else '❌ WRONG'}")
        
        # Test fine-tuned model WITH schema pruning
        print(f"\n[🟢 Fine-tuned Model (WITH Pruning)] Generating...")
        print(f"  Using pruned schema (min_tables={SCHEMA_MIN_TABLES}, max_cols={SCHEMA_MAX_COLUMNS})")
        ft_prune_sql, _ = manager.generate_sql("finetuned", prompt, original_schema=schema, question=question, use_pruning=True)
        ft_prune_correct = simple_sql_match(ft_prune_sql, ground_truth) if not ft_prune_sql.startswith("SKIPPED") else None
        print(f"  Predicted: {ft_prune_sql}")
        print(f"  {'✅ CORRECT' if ft_prune_correct else '⚠️ SKIPPED' if ft_prune_correct is None else '❌ WRONG'}")
        
        # Test Llama3 (with delay to avoid rate limits, uses FULL schema)
        print(f"\n[🔴 Llama3] Generating...")
        print(f"  Using FULL schema (large context window)")
        time.sleep(2)  # Small delay to avoid overwhelming Ollama
        llama_sql, llama_model = manager.generate_sql("llama", prompt)
        llama_correct = simple_sql_match(llama_sql, ground_truth) if not llama_sql.startswith("ERROR") else None
        print(f"  Model: {llama_model}")
        print(f"  Predicted: {llama_sql}")
        print(f"  {'✅ CORRECT' if llama_correct else '⚠️ ERROR' if llama_correct is None else '❌ WRONG'}")
        
        # Update sample with results
        sample['base_model_answer'] = base_sql
        sample['finetuned_no_prune_answer'] = ft_no_prune_sql
        sample['finetuned_prune_answer'] = ft_prune_sql
        sample['llama_answer'] = llama_sql
        sample['base_correct'] = base_correct
        sample['finetuned_no_prune_correct'] = ft_no_prune_correct
        sample['finetuned_prune_correct'] = ft_prune_correct
        sample['llama_correct'] = llama_correct
        sample['llama_model_used'] = llama_model
        sample['evaluated_at'] = datetime.now().isoformat()
        
        results.append(sample)
        
        print(f"\n{'='*100}")
        print(f"SUMMARY: Base={'✅' if base_correct else '❌'} | FT(NoPrune)={'✅' if ft_no_prune_correct else '⚠️' if ft_no_prune_correct is None else '❌'} | FT(Prune)={'✅' if ft_prune_correct else '⚠️' if ft_prune_correct is None else '❌'} | Llama3={'✅' if llama_correct else '⚠️' if llama_correct is None else '❌'}")
        print(f"{'='*100}\n")
    
    return results


# ============================================================================
# REPORTING
# ============================================================================

def print_results_table(results: List[Dict]):
    """Print comprehensive results table"""
    print("\n" + "=" * 150)
    print("EVALUATION RESULTS - BY QUERY TYPE")
    print("=" * 150)
    
    # Group by query type
    by_type = {}
    for r in results:
        qtype = r.get('query_type_name', r.get('query_type', 'Unknown'))
        if qtype not in by_type:
            by_type[qtype] = []
        by_type[qtype].append(r)
    
    # Calculate accuracies per type
    print(f"\n{'Query Type':<30} {'Base':<10} {'FT(NoPrune)':<12} {'FT(Prune)':<12} {'Llama3':<10} {'Winner':<15}")
    print("-" * 150)
    
    type_summaries = []
    for qtype_name, results_for_type in sorted(by_type.items()):
        total = len(results_for_type)
        
        base_correct = sum(1 for r in results_for_type if r.get('base_correct') == True)
        ft_no_prune_correct = sum(1 for r in results_for_type if r.get('finetuned_no_prune_correct') == True)
        ft_prune_correct = sum(1 for r in results_for_type if r.get('finetuned_prune_correct') == True)
        llama_correct = sum(1 for r in results_for_type if r.get('llama_correct') == True)
        
        base_pct = (base_correct / total * 100) if total > 0 else 0
        ft_no_prune_pct = (ft_no_prune_correct / total * 100) if total > 0 else 0
        ft_prune_pct = (ft_prune_correct / total * 100) if total > 0 else 0
        llama_pct = (llama_correct / total * 100) if total > 0 else 0
        
        # Determine winner
        max_pct = max(base_pct, ft_no_prune_pct, ft_prune_pct, llama_pct)
        if max_pct == base_pct and base_pct > 0:
            winner = "Base"
        elif max_pct == ft_no_prune_pct and ft_no_prune_pct > 0:
            winner = "FT(NoPrune)"
        elif max_pct == ft_prune_pct and ft_prune_pct > 0:
            winner = "FT(Prune)"
        elif max_pct == llama_pct and llama_pct > 0:
            winner = "Llama3"
        else:
            winner = "Tie/None"
        
        print(f"{qtype_name:<30} {base_pct:>5.0f}%    {ft_no_prune_pct:>5.0f}%      {ft_prune_pct:>5.0f}%      {llama_pct:>5.0f}%    {winner:<15}")
        
        type_summaries.append({
            'type': qtype_name,
            'base': base_pct,
            'ft_no_prune': ft_no_prune_pct,
            'ft_prune': ft_prune_pct,
            'llama': llama_pct,
            'winner': winner,
            'total': total
        })
    
    # Overall summary
    print("\n" + "=" * 150)
    print("OVERALL SUMMARY")
    print("=" * 150)
    
    total = len(results)
    base_total = sum(1 for r in results if r.get('base_correct') == True)
    ft_no_prune_total = sum(1 for r in results if r.get('finetuned_no_prune_correct') == True)
    ft_prune_total = sum(1 for r in results if r.get('finetuned_prune_correct') == True)
    llama_total = sum(1 for r in results if r.get('llama_correct') == True)
    
    base_acc = (base_total / total * 100) if total > 0 else 0
    ft_no_prune_acc = (ft_no_prune_total / total * 100) if total > 0 else 0
    ft_prune_acc = (ft_prune_total / total * 100) if total > 0 else 0
    llama_acc = (llama_total / total * 100) if total > 0 else 0
    
    print(f"\nTotal Test Cases: {total}")
    print(f"\nOverall Accuracy:")
    print(f"  Base Model (Qwen 2.5-1.5B):              {base_acc:5.1f}% ({base_total}/{total})")
    print(f"  Fine-tuned NO PRUNING (5K, r=16):       {ft_no_prune_acc:5.1f}% ({ft_no_prune_total}/{total})")
    print(f"  Fine-tuned WITH PRUNING (5K, r=16):    {ft_prune_acc:5.1f}% ({ft_prune_total}/{total})")
    print(f"  Llama3 (llama3:latest):                  {llama_acc:5.1f}% ({llama_total}/{total})")
    
    # Pruning comparison
    print(f"\n📊 Schema Pruning Impact:")
    if ft_no_prune_acc > 0 or ft_prune_acc > 0:
        diff = ft_prune_acc - ft_no_prune_acc
        print(f"  Pruning vs No Pruning: {diff:+.1f}% difference")
        if diff > 0:
            print(f"  ✅ Pruning HELPS (+{diff:.1f}%)")
        elif diff < 0:
            print(f"  ❌ Pruning HURTS ({diff:.1f}%)")
        else:
            print(f"  ➖ Pruning has NO EFFECT")
    
    # Best model
    best_model = max([
        ("Base", base_acc), 
        ("Fine-tuned (No Prune)", ft_no_prune_acc), 
        ("Fine-tuned (Prune)", ft_prune_acc),
        ("Llama3", llama_acc)
    ], key=lambda x: x[1])
    print(f"\n🏆 Best Overall Model: {best_model[0]} ({best_model[1]:.1f}%)")
    
    # Category wins
    wins = {"Base": 0, "FT(NoPrune)": 0, "FT(Prune)": 0, "Llama3": 0, "Tie/None": 0}
    for summary in type_summaries:
        wins[summary['winner']] += 1
    
    print(f"\nCategory Wins:")
    print(f"  Base: {wins['Base']}/{len(type_summaries)}")
    print(f"  Fine-tuned (No Prune): {wins['FT(NoPrune)']}/{len(type_summaries)}")
    print(f"  Fine-tuned (Prune): {wins['FT(Prune)']}/{len(type_summaries)}")
    print(f"  Llama3: {wins['Llama3']}/{len(type_summaries)}")
    
    print("=" * 150)


def save_results(results: List[Dict], test_file: Path):
    """Save results as JSON with model predictions"""
    output_file = test_file.parent / f"final_evaluation_100.json"
    
    total = len(results)
    
    output_data = {
        "metadata": {
            "evaluated_at": datetime.now().isoformat(),
            "total_examples": total,
            "models": {
                "base": "Qwen/Qwen2.5-1.5B-Instruct",
                "finetuned": f"{FINETUNED_MODEL_PATH.parent.name}/{FINETUNED_MODEL_PATH.name}" if FINETUNED_MODEL_PATH else "not_loaded",
                "llama": "llama3:latest"
            },
            "schema_pruning": {
                "tested_both": True,
                "min_tables": SCHEMA_MIN_TABLES,
                "max_columns": SCHEMA_MAX_COLUMNS
            },
            "dataset_filter": {
                "only_valid_schemas": True,
                "schema_required": True
            }
        },
        "overall_accuracy": {
            "base": sum(1 for r in results if r.get('base_correct') == True) / total if total > 0 else 0,
            "finetuned_no_prune": sum(1 for r in results if r.get('finetuned_no_prune_correct') == True) / total if total > 0 else 0,
            "finetuned_prune": sum(1 for r in results if r.get('finetuned_prune_correct') == True) / total if total > 0 else 0,
            "llama": sum(1 for r in results if r.get('llama_correct') == True) / total if total > 0 else 0
        },
        "pruning_comparison": {
            "difference": (sum(1 for r in results if r.get('finetuned_prune_correct') == True) - 
                          sum(1 for r in results if r.get('finetuned_no_prune_correct') == True)) / total if total > 0 else 0,
            "pruning_helps": sum(1 for r in results if r.get('finetuned_prune_correct') == True and r.get('finetuned_no_prune_correct') != True),
            "pruning_hurts": sum(1 for r in results if r.get('finetuned_no_prune_correct') == True and r.get('finetuned_prune_correct') != True)
        },
        "results": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Final evaluation saved to: {output_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 100)
    print("MODEL COMPARISON EVALUATION")
    print("=" * 100)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Show checkpoint being used
    if FINETUNED_MODEL_PATH and FINETUNED_MODEL_PATH.exists():
        checkpoint_name = f"{FINETUNED_MODEL_PATH.parent.name}/{FINETUNED_MODEL_PATH.name}"
        print(f"\n✓ Using best checkpoint: {checkpoint_name} (5K dataset, step 3000)")
    else:
        print(f"\n⚠ No fine-tuned checkpoint found - will skip fine-tuned evaluation")
    
    # Load models
    manager = ModelManager()
    manager.load_base_model()
    manager.load_finetuned_model()
    manager.load_llama()
    
    # Load test set
    print("\n" + "=" * 100)
    print("LOADING TEST DATASET")
    print("=" * 100)
    test_samples = load_test_set()
    
    if not test_samples:
        print("\n❌ No test data available. Exiting.")
        return
    
    # Limit to exactly 100 samples
    if len(test_samples) > MAX_TEST_SAMPLES:
        print(f"\n⚠ Limiting evaluation to {MAX_TEST_SAMPLES} samples (from {len(test_samples)})")
        test_samples = test_samples[:MAX_TEST_SAMPLES]
    
    print(f"\n✓ Loaded {len(test_samples)} test cases")
    
    # Show distribution
    type_counts = {}
    for s in test_samples:
        qtype = s.get('query_type_name', s.get('query_type', 'Unknown'))
        type_counts[qtype] = type_counts.get(qtype, 0) + 1
    
    print("\nDistribution:")
    for qtype, count in sorted(type_counts.items()):
        print(f"  {qtype}: {count} examples")
    
    # Evaluate
    results = evaluate_models(manager, test_samples)
    
    # Report
    print_results_table(results)
    
    # Save results
    test_file = UNBIASED_TEST_FILE if UNBIASED_TEST_FILE.exists() else COMPREHENSIVE_TEST_FILE
    save_results(results, test_file)
    
    print("\n" + "=" * 100)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)


if __name__ == "__main__":
    main()
