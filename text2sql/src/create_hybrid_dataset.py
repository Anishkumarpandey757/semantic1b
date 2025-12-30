#!/usr/bin/env python3
"""
Create Hybrid Balanced Dataset: BIRD + Synthetic Augmentation

Strategy:
1. Load all 7542 BIRD examples and classify by query type
2. Sample from BIRD to get balanced distribution
3. Generate synthetic examples ONLY for categories where BIRD is weak
4. Result: Real-world diversity + targeted improvement

Target: ~1400-1500 examples with balanced distribution
"""

import json
import random
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JSONL_DIR = PROJECT_ROOT / "outputs" / "jsonl"
BIRD_TRAIN = JSONL_DIR / "bird_train.jsonl"
OUTPUT_PATH = JSONL_DIR / "bird_hybrid_balanced.jsonl"

# Target distribution
TARGET_DISTRIBUTION = {
    'anti_join_not_exists': 80,      # BIRD has ~50, add 30 synthetic
    'anti_join_left_null': 60,       # BIRD has ~20, add 40 synthetic
    'anti_join_not_in': 40,          # BIRD has ~10, add 30 synthetic
    '4plus_table_join': 50,          # BIRD has ~30, add 20 synthetic
    '3_table_join': 150,             # BIRD has many
    '2_table_join_agg': 200,         # BIRD has many
    '2_table_join': 100,             # BIRD has many
    'date_filtering': 120,           # BIRD has many
    'complex_case': 80,              # BIRD has ~50, add 30 synthetic
    'subquery': 100,                 # BIRD has many
    'single_table_agg': 150,         # BIRD has many
    'single_table_where': 150,       # BIRD has many
    'simple_select': 120,            # BIRD has many
}

# ============================================================================
# QUERY CLASSIFICATION
# ============================================================================

def classify_query_type(sql: str) -> str:
    """Classify SQL query into categories"""
    sql_upper = sql.upper()
    
    # Anti-join patterns (priority)
    if 'NOT EXISTS' in sql_upper:
        return 'anti_join_not_exists'
    if 'LEFT JOIN' in sql_upper and 'IS NULL' in sql_upper:
        return 'anti_join_left_null'
    if 'NOT IN' in sql_upper and 'SELECT' in sql_upper.split('NOT IN')[1][:100]:
        return 'anti_join_not_in'
    
    # Count JOINs
    join_count = sql_upper.count(' JOIN ')
    
    # Complex patterns
    if 'CASE' in sql_upper or 'WHEN' in sql_upper:
        return 'complex_case'
    
    # Subqueries
    if sql_upper.count('SELECT') > 1:
        return 'subquery'
    
    # Date operations
    if any(x in sql_upper for x in ['YEAR(', 'DATE(', 'MONTH(', 'DAY(', 'CURDATE', 'NOW(', 'TIMESTAMP', 'BETWEEN']):
        return 'date_filtering'
    
    # Multi-table JOINs
    if join_count >= 3:
        return '4plus_table_join'
    if join_count == 2:
        return '3_table_join'
    if join_count == 1 and 'GROUP BY' in sql_upper:
        return '2_table_join_agg'
    if join_count == 1:
        return '2_table_join'
    
    # Single-table queries
    if 'GROUP BY' in sql_upper or any(x in sql_upper for x in ['COUNT(', 'AVG(', 'SUM(', 'MAX(', 'MIN(']):
        return 'single_table_agg'
    if 'WHERE' in sql_upper:
        return 'single_table_where'
    
    return 'simple_select'


# ============================================================================
# SYNTHETIC GENERATORS (Only for weak categories)
# ============================================================================

def generate_anti_join_synthetic(num_needed: int, pattern: str) -> List[Dict]:
    """Generate synthetic anti-join examples"""
    examples = []
    
    templates = {
        'anti_join_not_exists': [
            ('Which students have never enrolled in any course?',
             'SELECT name FROM Student s WHERE NOT EXISTS (SELECT 1 FROM Enrollment e WHERE s.student_id = e.student_id);',
             'Student(student_id PK, name VARCHAR, age INT, major VARCHAR, gpa FLOAT);\nEnrollment(enrollment_id PK, student_id FK, course_id FK, grade VARCHAR);'),
            ('Find products that have never been ordered',
             'SELECT product_name FROM Product p WHERE NOT EXISTS (SELECT 1 FROM OrderItem oi WHERE p.product_id = oi.product_id);',
             'Product(product_id PK, product_name VARCHAR, price DECIMAL, category VARCHAR);\nOrderItem(item_id PK, order_id FK, product_id FK, quantity INT);'),
            ('List employees who have not worked on any project',
             'SELECT name FROM Employee e WHERE NOT EXISTS (SELECT 1 FROM Assignment a WHERE e.emp_id = a.emp_id);',
             'Employee(emp_id PK, name VARCHAR, salary DECIMAL, dept_id FK);\nAssignment(assignment_id PK, emp_id FK, project_id FK, hours INT);'),
        ],
        'anti_join_left_null': [
            ('Which customers have no orders?',
             'SELECT c.name FROM Customer c LEFT JOIN Order o ON c.customer_id = o.customer_id WHERE o.order_id IS NULL;',
             'Customer(customer_id PK, name VARCHAR, email VARCHAR, city VARCHAR);\nOrder(order_id PK, customer_id FK, order_date DATE, total_amount DECIMAL);'),
            ('Find books that have never been loaned',
             'SELECT b.title FROM Book b LEFT JOIN Loan l ON b.book_id = l.book_id WHERE l.loan_id IS NULL;',
             'Book(book_id PK, title VARCHAR, author VARCHAR, isbn VARCHAR);\nLoan(loan_id PK, book_id FK, member_id FK, loan_date DATE);'),
            ('Show departments with no employees',
             'SELECT d.dept_name FROM Department d LEFT JOIN Employee e ON d.dept_id = e.dept_id WHERE e.emp_id IS NULL;',
             'Department(dept_id PK, dept_name VARCHAR, location VARCHAR, budget DECIMAL);\nEmployee(emp_id PK, name VARCHAR, salary DECIMAL, dept_id FK);'),
        ],
        'anti_join_not_in': [
            ('List students not enrolled in any course',
             'SELECT name FROM Student WHERE student_id NOT IN (SELECT student_id FROM Enrollment);',
             'Student(student_id PK, name VARCHAR, age INT, major VARCHAR);\nEnrollment(enrollment_id PK, student_id FK, course_id FK);'),
            ('Find products not in any order',
             'SELECT product_name FROM Product WHERE product_id NOT IN (SELECT product_id FROM OrderItem);',
             'Product(product_id PK, product_name VARCHAR, price DECIMAL);\nOrderItem(item_id PK, product_id FK, quantity INT);'),
        ],
    }
    
    if pattern not in templates:
        return []
    
    for _ in range((num_needed // len(templates[pattern])) + 1):
        for q, sql, schema in templates[pattern]:
            if len(examples) >= num_needed:
                break
            examples.append({
                'question': q,
                'SQL': sql,
                'db_schema': schema,
            })
    
    return examples[:num_needed]


def generate_complex_join_synthetic(num_needed: int) -> List[Dict]:
    """Generate synthetic 4+ table JOIN examples"""
    examples = []
    
    templates = [
        ('Show customer orders with products',
         'SELECT c.name, o.order_date, oi.quantity, p.product_name FROM Customer c JOIN Order o ON c.customer_id = o.customer_id JOIN OrderItem oi ON o.order_id = oi.order_id JOIN Product p ON oi.product_id = p.product_id;',
         'Customer(customer_id PK, name VARCHAR);\nOrder(order_id PK, customer_id FK, order_date DATE);\nOrderItem(item_id PK, order_id FK, product_id FK, quantity INT);\nProduct(product_id PK, product_name VARCHAR);'),
        ('List employees with their departments and projects',
         'SELECT e.name, d.dept_name, a.role, p.project_name FROM Employee e JOIN Department d ON e.dept_id = d.dept_id JOIN Assignment a ON e.emp_id = a.emp_id JOIN Project p ON a.project_id = p.project_id;',
         'Employee(emp_id PK, name VARCHAR, dept_id FK);\nDepartment(dept_id PK, dept_name VARCHAR);\nAssignment(assignment_id PK, emp_id FK, project_id FK, role VARCHAR);\nProject(project_id PK, project_name VARCHAR);'),
    ]
    
    for _ in range((num_needed // len(templates)) + 1):
        for q, sql, schema in templates:
            if len(examples) >= num_needed:
                break
            examples.append({
                'question': q,
                'SQL': sql,
                'db_schema': schema,
            })
    
    return examples[:num_needed]


def generate_complex_case_synthetic(num_needed: int) -> List[Dict]:
    """Generate synthetic CASE statement examples"""
    examples = []
    
    templates = [
        ('Classify students by GPA',
         "SELECT name, CASE WHEN gpa >= 3.5 THEN 'Excellent' WHEN gpa >= 3.0 THEN 'Good' WHEN gpa >= 2.0 THEN 'Average' ELSE 'Below Average' END AS performance FROM Student;",
         'Student(student_id PK, name VARCHAR, gpa FLOAT);'),
        ('Categorize products by price',
         "SELECT product_name, CASE WHEN price > 100 THEN 'Premium' WHEN price > 50 THEN 'Standard' ELSE 'Budget' END AS price_tier FROM Product;",
         'Product(product_id PK, product_name VARCHAR, price DECIMAL);'),
        ('Classify employees by salary',
         "SELECT name, CASE WHEN salary >= 100000 THEN 'High' WHEN salary >= 60000 THEN 'Medium' ELSE 'Low' END AS salary_level FROM Employee;",
         'Employee(emp_id PK, name VARCHAR, salary DECIMAL);'),
    ]
    
    for _ in range((num_needed // len(templates)) + 1):
        for q, sql, schema in templates:
            if len(examples) >= num_needed:
                break
            examples.append({
                'question': q,
                'SQL': sql,
                'db_schema': schema,
            })
    
    return examples[:num_needed]


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 70)
    print("CREATING HYBRID BALANCED DATASET")
    print("Strategy: BIRD examples + Synthetic augmentation")
    print("=" * 70)
    print()
    
    # Load BIRD dataset
    print(f"Loading BIRD dataset from: {BIRD_TRAIN}")
    bird_examples = []
    with open(BIRD_TRAIN, 'r', encoding='utf-8') as f:
        for line in f:
            bird_examples.append(json.loads(line))
    print(f"  ✅ Loaded {len(bird_examples)} BIRD examples")
    print()
    
    # Classify BIRD examples
    print("Classifying BIRD examples by query type...")
    classified = defaultdict(list)
    for ex in bird_examples:
        sql = ex.get('SQL', ex.get('sql', ''))
        query_type = classify_query_type(sql)
        classified[query_type].append(ex)
    
    print("BIRD distribution:")
    for qtype in sorted(classified.keys()):
        print(f"  {qtype}: {len(classified[qtype])} examples")
    print()
    
    # Build balanced dataset
    print("Building balanced dataset...")
    final_dataset = []
    synthetic_needed = defaultdict(int)
    
    for qtype, target_count in TARGET_DISTRIBUTION.items():
        available = len(classified[qtype])
        
        if available >= target_count:
            # Sample from BIRD
            sampled = random.sample(classified[qtype], target_count)
            final_dataset.extend(sampled)
            print(f"  ✅ {qtype}: {target_count} from BIRD (had {available})")
        else:
            # Take all BIRD + generate synthetic
            final_dataset.extend(classified[qtype])
            synthetic_needed[qtype] = target_count - available
            print(f"  ⚠️  {qtype}: {available} from BIRD + {synthetic_needed[qtype]} synthetic needed")
    
    print()
    
    # Generate synthetic for weak categories
    if synthetic_needed:
        print("Generating synthetic examples for weak categories...")
        
        for qtype, num_needed in synthetic_needed.items():
            if num_needed <= 0:
                continue
            
            if qtype == 'anti_join_not_exists':
                synthetic = generate_anti_join_synthetic(num_needed, 'anti_join_not_exists')
            elif qtype == 'anti_join_left_null':
                synthetic = generate_anti_join_synthetic(num_needed, 'anti_join_left_null')
            elif qtype == 'anti_join_not_in':
                synthetic = generate_anti_join_synthetic(num_needed, 'anti_join_not_in')
            elif qtype == '4plus_table_join':
                synthetic = generate_complex_join_synthetic(num_needed)
            elif qtype == 'complex_case':
                synthetic = generate_complex_case_synthetic(num_needed)
            else:
                print(f"  ⚠️  No synthetic generator for {qtype}, skipping")
                continue
            
            final_dataset.extend(synthetic)
            print(f"  ✅ Generated {len(synthetic)} synthetic {qtype} examples")
    
    print()
    print(f"Total dataset size: {len(final_dataset)} examples")
    
    # Shuffle
    random.shuffle(final_dataset)
    
    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for ex in final_dataset:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    print(f"✅ Saved to: {OUTPUT_PATH}")
    print()
    
    # Final distribution
    print("Final balanced distribution:")
    final_classified = defaultdict(int)
    for ex in final_dataset:
        sql = ex.get('SQL', ex.get('sql', ''))
        qtype = classify_query_type(sql)
        final_classified[qtype] += 1
    
    for qtype in sorted(final_classified.keys()):
        target = TARGET_DISTRIBUTION.get(qtype, 0)
        actual = final_classified[qtype]
        status = "✅" if abs(actual - target) <= 5 else "⚠️"
        print(f"  {status} {qtype}: {actual} (target: {target})")
    
    print()
    print("=" * 70)
    print("✅ HYBRID DATASET COMPLETE!")
    print("=" * 70)
    print()
    print("Next: Update training script to use 'bird_hybrid_balanced.jsonl'")
    print()


if __name__ == "__main__":
    main()
