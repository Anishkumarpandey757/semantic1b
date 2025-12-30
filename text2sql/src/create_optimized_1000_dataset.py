#!/usr/bin/env python3
"""
Create Optimized 1000-Example Dataset for r=8 Training

Strategy:
- 1000 examples is optimal for r=8 (proven with 500 examples success)
- Balanced across query complexity
- Diverse databases to prevent overfitting
- Quality over quantity - each example teaches something unique

Distribution:
- Simple (300): Basic SELECT, WHERE, single table
- Moderate (450): JOINs, GROUP BY, aggregations
- Complex (250): Multi-table JOINs, subqueries, CASE, date functions
"""

import json
import random
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JSONL_DIR = PROJECT_ROOT / "outputs" / "jsonl"
INPUT_FILE = JSONL_DIR / "bird_semantic_balanced.jsonl"  # Source: 2003 examples
OUTPUT_FILE = JSONL_DIR / "bird_optimized_1000.jsonl"

# Target distribution for 1000 examples
TARGET_DISTRIBUTION = {
    # Simple queries (300 total)
    'simple_select': 60,           # Basic SELECT
    'simple_where': 80,             # WHERE conditions
    'simple_aggregation': 80,       # COUNT, SUM, AVG, MIN, MAX
    'simple_order_limit': 80,       # ORDER BY, LIMIT
    
    # Moderate queries (450 total)
    '2_table_join': 150,            # Simple 2-table JOIN
    '2_table_join_agg': 100,        # 2-table JOIN with aggregation
    '3_table_join': 100,            # 3-table JOIN
    'group_by_having': 100,         # GROUP BY with HAVING
    
    # Complex queries (250 total)
    '4plus_table_join': 60,         # 4+ table JOINs
    'subquery': 80,                 # Nested SELECT
    'complex_case': 40,             # CASE statements
    'date_operations': 40,          # Date filtering/manipulation
    'window_functions': 15,         # OVER, PARTITION BY
    'anti_join': 15,                # NOT EXISTS, LEFT JOIN IS NULL
}

def classify_query_complexity(sql: str) -> str:
    """Classify SQL query into specific categories"""
    sql_upper = sql.upper()
    
    # Count JOINs
    join_count = sql_upper.count(' JOIN ')
    
    # Complex patterns first (most specific)
    if 'OVER' in sql_upper or 'PARTITION BY' in sql_upper:
        return 'window_functions'
    
    if 'NOT EXISTS' in sql_upper or ('LEFT JOIN' in sql_upper and 'IS NULL' in sql_upper):
        return 'anti_join'
    
    if 'CASE' in sql_upper and 'WHEN' in sql_upper:
        return 'complex_case'
    
    if any(x in sql_upper for x in ['YEAR(', 'DATE(', 'MONTH(', 'DAY(', 'BETWEEN', 'CURDATE', 'NOW(']):
        return 'date_operations'
    
    if sql_upper.count('SELECT') > 1:  # Subquery
        return 'subquery'
    
    # Multi-table JOINs
    if join_count >= 3:
        return '4plus_table_join'
    if join_count == 2:
        return '3_table_join'
    if join_count == 1:
        if 'GROUP BY' in sql_upper or any(x in sql_upper for x in ['COUNT(', 'SUM(', 'AVG(', 'MAX(', 'MIN(']):
            return '2_table_join_agg'
        return '2_table_join'
    
    # Single table queries
    if 'GROUP BY' in sql_upper and 'HAVING' in sql_upper:
        return 'group_by_having'
    
    if any(x in sql_upper for x in ['COUNT(', 'SUM(', 'AVG(', 'MAX(', 'MIN(']):
        return 'simple_aggregation'
    
    if 'ORDER BY' in sql_upper or 'LIMIT' in sql_upper:
        return 'simple_order_limit'
    
    if 'WHERE' in sql_upper:
        return 'simple_where'
    
    return 'simple_select'


def calculate_quality_score(example: Dict) -> float:
    """Calculate quality score for an example
    
    Factors:
    - SQL length (moderate is better - not too simple, not too complex)
    - Schema complexity (more tables = more learning)
    - Question clarity
    """
    sql = example.get('SQL', '')
    schema = example.get('db_schema', '')
    question = example.get('question', '')
    
    score = 0.0
    
    # SQL length (sweet spot: 50-150 characters)
    sql_len = len(sql)
    if 50 <= sql_len <= 150:
        score += 3.0
    elif 30 <= sql_len <= 200:
        score += 2.0
    else:
        score += 1.0
    
    # Schema complexity (more tables = better learning)
    table_count = schema.count('(') - schema.count('PRIMARY KEY(')
    score += min(table_count * 0.5, 3.0)
    
    # Question length (detailed questions are better)
    question_len = len(question.split())
    if 10 <= question_len <= 25:
        score += 2.0
    elif 5 <= question_len <= 35:
        score += 1.0
    
    # Bonus for diverse SQL patterns
    sql_upper = sql.upper()
    if 'DISTINCT' in sql_upper:
        score += 0.5
    if 'AS T' in sql_upper:  # Table aliases (better SQL practice)
        score += 0.5
    
    return score


def main():
    print("=" * 70)
    print("CREATING OPTIMIZED 1000-EXAMPLE DATASET")
    print("Strategy: Balanced complexity for r=8 training")
    print("=" * 70)
    print()
    
    # Load source dataset
    print(f"Loading source dataset: {INPUT_FILE}")
    source_examples = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                source_examples.append(json.loads(line))
    print(f"  ✅ Loaded {len(source_examples)} source examples")
    print()
    
    # Classify and score all examples
    print("Classifying examples by complexity...")
    classified = defaultdict(list)
    
    for ex in source_examples:
        sql = ex.get('SQL', '')
        category = classify_query_complexity(sql)
        quality_score = calculate_quality_score(ex)
        ex['_quality_score'] = quality_score
        classified[category].append(ex)
    
    print("\nSource distribution:")
    for cat in sorted(classified.keys()):
        print(f"  {cat}: {len(classified[cat])} examples")
    print()
    
    # Sample balanced dataset
    print("Sampling balanced 1000-example dataset...")
    final_dataset = []
    stats = defaultdict(int)
    
    for category, target_count in TARGET_DISTRIBUTION.items():
        available = classified[category]
        
        if len(available) == 0:
            print(f"  ⚠️  {category}: No examples available!")
            continue
        
        # Sort by quality score and take top examples
        available_sorted = sorted(available, key=lambda x: x['_quality_score'], reverse=True)
        
        # Take target count (or all if less available)
        selected_count = min(target_count, len(available_sorted))
        selected = available_sorted[:selected_count]
        
        # Remove quality score before adding to final dataset
        for ex in selected:
            ex_clean = {k: v for k, v in ex.items() if k != '_quality_score'}
            final_dataset.append(ex_clean)
        
        stats[category] = selected_count
        status = "✅" if selected_count >= target_count * 0.9 else "⚠️"
        print(f"  {status} {category}: {selected_count}/{target_count}")
    
    print()
    print(f"Total selected: {len(final_dataset)} examples")
    
    # If less than 1000, add best remaining examples
    if len(final_dataset) < 1000:
        remaining_needed = 1000 - len(final_dataset)
        print(f"\n📊 Need {remaining_needed} more examples to reach 1000")
        
        # Get all examples not yet selected
        selected_ids = {id(ex) for ex in final_dataset}
        remaining = []
        for cat_examples in classified.values():
            for ex in cat_examples:
                if id(ex) not in selected_ids:
                    remaining.append(ex)
        
        # Sort by quality and take top
        remaining_sorted = sorted(remaining, key=lambda x: x['_quality_score'], reverse=True)
        additional = remaining_sorted[:remaining_needed]
        
        for ex in additional:
            ex_clean = {k: v for k, v in ex.items() if k != '_quality_score'}
            final_dataset.append(ex_clean)
        
        print(f"  ✅ Added {len(additional)} high-quality examples")
        print(f"  Total now: {len(final_dataset)} examples")
    
    # Shuffle for training
    random.seed(42)
    random.shuffle(final_dataset)
    
    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for ex in final_dataset:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    print(f"✅ Saved to: {OUTPUT_FILE}")
    print()
    
    # Verify distribution
    print("Final distribution verification:")
    final_classified = defaultdict(int)
    for ex in final_dataset:
        sql = ex.get('SQL', '')
        cat = classify_query_complexity(sql)
        final_classified[cat] += 1
    
    simple_count = sum(final_classified[k] for k in ['simple_select', 'simple_where', 'simple_aggregation', 'simple_order_limit'])
    moderate_count = sum(final_classified[k] for k in ['2_table_join', '2_table_join_agg', '3_table_join', 'group_by_having'])
    complex_count = sum(final_classified[k] for k in ['4plus_table_join', 'subquery', 'complex_case', 'date_operations', 'window_functions', 'anti_join'])
    
    print(f"  Simple queries: {simple_count} ({simple_count/10:.1f}%)")
    print(f"  Moderate queries: {moderate_count} ({moderate_count/10:.1f}%)")
    print(f"  Complex queries: {complex_count} ({complex_count/10:.1f}%)")
    print()
    
    # Check database diversity
    db_ids = defaultdict(int)
    for ex in final_dataset:
        db_id = ex.get('db_id', 'unknown')
        if db_id:
            db_ids[db_id] += 1
    
    print(f"Database diversity: {len(db_ids)} unique databases")
    print(f"  Average examples per database: {len(final_dataset)/len(db_ids):.1f}")
    print()
    
    print("=" * 70)
    print("✅ OPTIMIZED 1000-EXAMPLE DATASET COMPLETE!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Update phase3_train_lora_optimized.py:")
    print("   - Set: balanced_file = 'bird_optimized_1000.jsonl'")
    print("   - Keep: lora_r = 8 (proven to work with this size)")
    print("   - Set: num_train_epochs = 5 (like your successful model)")
    print("2. Run: python src/phase3_train_lora_optimized.py")
    print()


if __name__ == "__main__":
    main()
