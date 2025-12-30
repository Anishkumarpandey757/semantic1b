#!/usr/bin/env python3
"""
Reformat Dataset with Enhanced Schema Clarity

Problem: Current schema format is too compressed:
  "game(id INTEGER, genre_id INTEGER; PK(id); FK genre_id->genre(id))\ngame_platform(...)"

Solution: Format with clear structure:
  Table: game
    - id: INTEGER (PRIMARY KEY)
    - genre_id: INTEGER (FOREIGN KEY -> genre.id)
    - game_name: TEXT
  
  Table: game_platform
    - id: INTEGER (PRIMARY KEY)
    ...

This helps the model understand:
1. Which columns belong to which table
2. Data types clearly
3. Key relationships explicitly
"""

import json
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JSONL_DIR = PROJECT_ROOT / "outputs" / "jsonl"

INPUT_FILE = JSONL_DIR / "bird_optimized_1000.jsonl"
OUTPUT_FILE = JSONL_DIR / "bird_optimized_1000_clear.jsonl"

def format_schema_clearly(schema_text: str) -> str:
    """Reformat compressed schema into clear structure"""
    
    # Split by table definitions
    tables = []
    current_table = ""
    paren_depth = 0
    
    for char in schema_text:
        current_table += char
        if char == '(':
            paren_depth += 1
        elif char == ')':
            paren_depth -= 1
            if paren_depth == 0:
                tables.append(current_table.strip())
                current_table = ""
    
    # Format each table
    formatted_tables = []
    for table_def in tables:
        if not table_def:
            continue
            
        # Extract table name
        table_name = table_def.split('(')[0].strip()
        
        # Extract column definitions
        content = table_def[table_def.find('(')+1:table_def.rfind(')')].strip()
        
        # Parse columns
        parts = []
        current_part = ""
        paren_depth = 0
        
        for char in content:
            if char == '(' :
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
            
            if char == ';' and paren_depth == 0:
                if current_part.strip():
                    parts.append(current_part.strip())
                current_part = ""
            else:
                current_part += char
        
        if current_part.strip():
            parts.append(current_part.strip())
        
        # Build formatted output
        formatted = f"Table: {table_name}\n"
        
        for part in parts:
            if part.startswith('PRIMARY KEY'):
                formatted += f"  {part}\n"
            elif part.startswith('FK ') or part.startswith('FOREIGN KEY'):
                formatted += f"  {part}\n"
            else:
                # Column definition
                # Format: name TYPE or name TYPE, name TYPE
                cols = part.split(',')
                for col in cols:
                    col = col.strip()
                    if col:
                        formatted += f"  - {col}\n"
        
        formatted_tables.append(formatted.strip())
    
    return "\n\n".join(formatted_tables)


def main():
    print("=" * 70)
    print("REFORMATTING SCHEMA FOR CLARITY")
    print("=" * 70)
    print()
    
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print()
    
    converted_count = 0
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as fin, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            if not line.strip():
                continue
                
            example = json.loads(line)
            
            # Reformat schema
            old_schema = example.get('db_schema', '')
            new_schema = format_schema_clearly(old_schema)
            
            # Update example
            example['db_schema'] = new_schema
            
            # Write
            fout.write(json.dumps(example, ensure_ascii=False) + '\n')
            converted_count += 1
    
    print(f"✅ Converted {converted_count} examples")
    print()
    
    # Show example
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        first_example = json.loads(f.readline())
    
    print("Sample formatted schema:")
    print("=" * 70)
    print(first_example['db_schema'][:500])
    print("..." if len(first_example['db_schema']) > 500 else "")
    print()
    print("=" * 70)
    print("✅ COMPLETE!")
    print()
    print("Next: Update phase3_train_lora_optimized.py:")
    print("  balanced_file = 'bird_optimized_1000_clear.jsonl'")
    print()


if __name__ == "__main__":
    main()
