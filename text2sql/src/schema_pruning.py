#!/usr/bin/env python3
"""
Schema Pruning for Text-to-SQL

Reduces schema complexity by keeping only relevant tables/columns
based on question tokens. This is deterministic (same for train+eval)
and dramatically reduces the "memorization burden" on small LoRA ranks.

Key benefits:
- Reduces r=8 underfitting without changing model capacity
- Same format (just shorter schemas)
- Keeps PK/FK columns to preserve join capability
"""

import re
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TableSchema:
    """Single table schema"""
    name: str
    columns: Dict[str, str]  # {col_name: type}
    primary_keys: List[str]
    foreign_keys: List[Tuple[str, str, str]]  # (col, ref_table, ref_col)


def parse_full_schema(db_schema: str) -> Dict[str, TableSchema]:
    """
    Parse compressed schema into structured table info
    
    Input: "table1(col1 TYPE, col2 TYPE; PK(col1), FK(col2 REFERENCES table2(id))); ..."
    Output: {table_name: TableSchema}
    """
    tables = {}
    
    # Split by ); to separate tables
    table_parts = db_schema.split(');')
    
    for part in table_parts:
        part = part.strip()
        if not part or '(' not in part:
            continue
            
        # Extract table name
        table_name_match = re.match(r'(\w+)\(', part)
        if not table_name_match:
            continue
            
        table_name = table_name_match.group(1)
        
        # Extract content
        content = part[part.index('(')+1:]
        
        # Split columns and constraints
        if ';' in content:
            col_part, constraint_part = content.split(';', 1)
        else:
            col_part = content
            constraint_part = ''
            
        # Parse columns
        columns = {}
        for col_def in col_part.split(','):
            col_def = col_def.strip()
            if not col_def:
                continue
                
            parts = col_def.split()
            if len(parts) >= 2:
                col_name = parts[0].strip()
                col_type = ' '.join(parts[1:]).strip()
                columns[col_name.lower()] = col_type
        
        # Parse primary keys
        primary_keys = []
        pk_matches = re.findall(r'PK\(([^)]+)\)', constraint_part)
        for pk in pk_matches:
            primary_keys.extend([col.strip().lower() for col in pk.split(',')])
        
        # Parse foreign keys
        foreign_keys = []
        fk_matches = re.findall(r'FK\(([^)]+)\)', constraint_part)
        for fk in fk_matches:
            # Format: "col REFERENCES table(ref_col)"
            fk_match = re.match(r'(\w+)\s+REFERENCES\s+(\w+)\((\w+)\)', fk.strip())
            if fk_match:
                col, ref_table, ref_col = fk_match.groups()
                foreign_keys.append((col.lower(), ref_table.lower(), ref_col.lower()))
        
        tables[table_name.lower()] = TableSchema(
            name=table_name,
            columns=columns,
            primary_keys=primary_keys,
            foreign_keys=foreign_keys
        )
    
    return tables


def extract_relevant_tokens(question: str) -> Set[str]:
    """
    Extract potentially relevant tokens from question
    
    Normalizes to lowercase, removes stopwords, keeps meaningful words
    """
    # Simple tokenization
    tokens = re.findall(r'\b\w+\b', question.lower())
    
    # Remove common stopwords (these rarely match schema names)
    stopwords = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
        'could', 'can', 'may', 'might', 'must', 'shall', 'in', 'on', 'at',
        'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into',
        'through', 'during', 'before', 'after', 'above', 'below', 'between',
        'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
        'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
        'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than',
        'too', 'very', 'that', 'this', 'these', 'those', 'what', 'which', 'who',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them', 'their', 'our',
        'your', 'his', 'her', 'its', 'my', 'me', 'him', 'us', 'but', 'or', 'and'
    }
    
    relevant = {t for t in tokens if t not in stopwords and len(t) > 1}
    
    # Add common synonyms for SQL patterns
    synonym_map = {
        'number': {'num', 'no', 'count', 'quantity', 'qty'},
        'id': {'identifier', 'code', 'key'},
        'name': {'title', 'label', 'description'},
        'date': {'time', 'year', 'month', 'day', 'when'},
        'amount': {'total', 'sum', 'value', 'price', 'cost'},
        'type': {'kind', 'category', 'class'},
        'status': {'state', 'condition'},
    }
    
    expanded = set(relevant)
    for word in relevant:
        for key, synonyms in synonym_map.items():
            if word in synonyms or word == key:
                expanded.update(synonyms)
                expanded.add(key)
    
    return expanded


def name_matches_tokens(name: str, tokens: Set[str]) -> bool:
    """
    Check if table/column name matches any question tokens
    
    Handles:
    - Direct matches: "customer" matches "customer"
    - Substring matches: "customer_id" matches "customer"
    - Compound words: "FirstName" matches "name"
    """
    name_lower = name.lower()
    
    # Direct match
    if name_lower in tokens:
        return True
    
    # Split snake_case and check parts
    parts = name_lower.split('_')
    for part in parts:
        if part in tokens and len(part) > 1:
            return True
    
    # Split camelCase (e.g., "FirstName" -> "first", "name")
    camel_parts = re.findall(r'[a-z]+', name_lower)
    for part in camel_parts:
        if part in tokens and len(part) > 2:
            return True
    
    # Check if any token is substring of name (or vice versa)
    for token in tokens:
        if len(token) > 2:
            if token in name_lower or name_lower in token:
                return True
    
    return False


def prune_schema(
    db_schema: str,
    question: str,
    min_tables: int = 1,
    max_columns_per_table: int = 10
) -> str:
    """
    Prune schema to include only relevant tables/columns
    
    Algorithm:
    1. Extract question tokens
    2. Include tables whose names match tokens
    3. For each included table:
       - Always keep PK columns
       - Always keep FK columns (to preserve joins)
       - Keep columns whose names match tokens
       - Keep up to max_columns_per_table most relevant columns
    4. Include referenced tables (via FK) even if not directly mentioned
    
    Args:
        db_schema: Original compressed schema
        question: Natural language question
        min_tables: Minimum tables to keep (prevents over-pruning)
        max_columns_per_table: Max columns per table (prevents bloat)
    
    Returns:
        Pruned schema in same compressed format
    """
    # Parse schema
    tables = parse_full_schema(db_schema)
    if not tables:
        return db_schema  # Can't parse, return original
    
    # Extract relevant tokens
    tokens = extract_relevant_tokens(question)
    
    # Find relevant tables
    relevant_tables = set()
    
    # Step 1: Tables whose names match question tokens
    for table_name, table_schema in tables.items():
        if name_matches_tokens(table_name, tokens):
            relevant_tables.add(table_name)
    
    # Step 2: Add tables referenced by FK (to preserve join capability)
    for table_name in list(relevant_tables):
        for _, ref_table, _ in tables[table_name].foreign_keys:
            if ref_table in tables:
                relevant_tables.add(ref_table)
    
    # Step 3: Add tables that reference selected tables (reverse FK)
    for table_name, table_schema in tables.items():
        for _, ref_table, _ in table_schema.foreign_keys:
            if ref_table in relevant_tables:
                relevant_tables.add(table_name)
    
    # Ensure minimum tables
    if len(relevant_tables) < min_tables:
        # Add most "central" tables (most FKs)
        table_scores = []
        for table_name, table_schema in tables.items():
            if table_name not in relevant_tables:
                # Count FK connections
                score = len(table_schema.foreign_keys)
                # Count incoming FKs
                for other_table in tables.values():
                    for _, ref_table, _ in other_table.foreign_keys:
                        if ref_table == table_name:
                            score += 1
                table_scores.append((score, table_name))
        
        table_scores.sort(reverse=True)
        for _, table_name in table_scores[:min_tables - len(relevant_tables)]:
            relevant_tables.add(table_name)
    
    # Prune columns in each relevant table
    pruned_tables = {}
    
    for table_name in relevant_tables:
        table_schema = tables[table_name]
        
        # Always keep PK and FK columns
        keep_columns = set()
        keep_columns.update(table_schema.primary_keys)
        for col, _, _ in table_schema.foreign_keys:
            keep_columns.add(col)
        
        # Add columns whose names match question tokens
        for col_name in table_schema.columns:
            if name_matches_tokens(col_name, tokens):
                keep_columns.add(col_name)
        
        # If we have too few columns, add more
        if len(keep_columns) < 3:
            remaining_cols = [c for c in table_schema.columns if c not in keep_columns]
            keep_columns.update(remaining_cols[:3])
        
        # If we have too many, limit to most relevant
        if len(keep_columns) > max_columns_per_table:
            # Priority: PK > FK > matched > others
            priority_cols = []
            for col in keep_columns:
                priority = 0
                if col in table_schema.primary_keys:
                    priority += 100
                if any(col == fk[0] for fk in table_schema.foreign_keys):
                    priority += 50
                if name_matches_tokens(col, tokens):
                    priority += 10
                priority_cols.append((priority, col))
            
            priority_cols.sort(reverse=True)
            keep_columns = {col for _, col in priority_cols[:max_columns_per_table]}
        
        # Build pruned table schema
        pruned_columns = {
            col: table_schema.columns[col]
            for col in keep_columns
            if col in table_schema.columns
        }
        
        pruned_tables[table_name] = TableSchema(
            name=table_schema.name,
            columns=pruned_columns,
            primary_keys=table_schema.primary_keys,
            foreign_keys=table_schema.foreign_keys
        )
    
    # Reconstruct compressed schema
    schema_parts = []
    for table_name, table_schema in pruned_tables.items():
        # Columns
        col_strs = [f"{col} {typ}" for col, typ in table_schema.columns.items()]
        col_part = ', '.join(col_strs)
        
        # Constraints
        constraint_parts = []
        if table_schema.primary_keys:
            pk_str = ', '.join(table_schema.primary_keys)
            constraint_parts.append(f"PK({pk_str})")
        
        for col, ref_table, ref_col in table_schema.foreign_keys:
            if col in table_schema.columns:  # Only include if column kept
                constraint_parts.append(f"FK({col} REFERENCES {ref_table}({ref_col}))")
        
        # Combine
        if constraint_parts:
            constraint_str = ', '.join(constraint_parts)
            table_str = f"{table_schema.name}({col_part}; {constraint_str})"
        else:
            table_str = f"{table_schema.name}({col_part})"
        
        schema_parts.append(table_str)
    
    pruned_schema = '); '.join(schema_parts)
    if pruned_schema:
        pruned_schema += ')'
    
    # Log pruning stats
    original_table_count = len(tables)
    pruned_table_count = len(pruned_tables)
    original_col_count = sum(len(t.columns) for t in tables.values())
    pruned_col_count = sum(len(t.columns) for t in pruned_tables.values())
    
    logger.debug(
        f"Schema pruning: {original_table_count} -> {pruned_table_count} tables, "
        f"{original_col_count} -> {pruned_col_count} columns"
    )
    
    return pruned_schema if pruned_schema else db_schema


def prune_dataset(input_path: str, output_path: str, min_tables: int = 1, max_columns: int = 10):
    """
    Apply schema pruning to entire dataset (for retraining)
    
    Args:
        input_path: Input JSONL file
        output_path: Output JSONL file (pruned schemas)
        min_tables: Minimum tables to keep per example
        max_columns: Max columns per table
    """
    import json
    
    pruned_count = 0
    total_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            line = line.strip()
            if not line:
                continue
                
            example = json.loads(line)
            total_count += 1
            
            # Prune schema
            original_schema = example['db_schema']
            pruned = prune_schema(
                original_schema,
                example['question'],
                min_tables=min_tables,
                max_columns_per_table=max_columns
            )
            
            if pruned != original_schema:
                pruned_count += 1
            
            example['db_schema'] = pruned
            
            # Write
            fout.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    logger.info(
        f"Pruned {pruned_count}/{total_count} schemas ({100*pruned_count/total_count:.1f}%)"
    )


if __name__ == "__main__":
    # Test pruning
    import sys
    import json
    
    if len(sys.argv) < 3:
        print("Usage: python schema_pruning.py <input.jsonl> <output.jsonl>")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print(f"Pruning schemas: {input_file} -> {output_file}")
    prune_dataset(input_file, output_file)
    print("Done!")
