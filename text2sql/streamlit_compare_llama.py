#!/usr/bin/env python3
"""
Streamlit App: Compare Base vs Fine-tuned vs Llama3 for Text-to-SQL

Inspired by evaluate_models_comparison.py
Supports custom input and default test questions with diverse schemas.
"""

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path
import time
import os
import re
import pandas as pd
import requests
import json
import sys
from datetime import datetime

# Add src to path for schema pruning
sys.path.insert(0, str(Path(__file__).parent / "src"))
from schema_pruning import prune_schema

# Set page config
st.set_page_config(
    page_title="SQL Generation Comparison - Base vs Fine-tuned vs Llama3",
    page_icon="🔍",
    layout="wide"
)

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
JSONL_DIR = PROJECT_ROOT / "outputs" / "jsonl"

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
FINETUNED_PATH = CHECKPOINT_DIR / "qwen_5k_r16_20251225_083035" / "checkpoint-3000"

# Ollama configuration
OLLAMA_MODEL = "llama3:latest"
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_URL = f"{OLLAMA_BASE}/api/generate"

# Schema pruning (same as evaluate_models_comparison.py)
USE_SCHEMA_PRUNING = True
SCHEMA_MIN_TABLES = 2
SCHEMA_MAX_COLUMNS = 12

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_prompt(question: str, schema: str = None) -> str:
    """Format prompt for model (same as evaluate_models_comparison.py)"""
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

def normalize_sql(sql):
    """Normalize SQL for comparison"""
    if not sql or sql.startswith('⚠️') or sql.startswith('ERROR') or sql.startswith('SKIPPED'):
        return ""
    sql = sql.upper().strip()
    sql = re.sub(r'\s+', ' ', sql)
    sql = sql.rstrip(';')
    return sql

def simple_sql_match(pred: str, ground_truth: str) -> bool:
    """Simple SQL comparison (same as evaluate_models_comparison.py)"""
    def normalize(sql):
        return ' '.join(sql.upper().split()).strip()
    
    return normalize(pred) == normalize(ground_truth)

# ============================================================================
# DEFAULT TEST QUESTIONS (Diverse schemas from BIRD/Spider)
# ============================================================================

DEFAULT_QUESTIONS = [
    {
        "type": "Simple SELECT - Student Database",
        "question": "List all student names.",
        "schema": "Student(StuID number, LName text, Fname text, Age number, Sex text, Major number, Advisor number, city_code text; PRIMARY KEY(StuID))",
        "expected": "SELECT Fname, LName FROM Student"
    },
    {
        "type": "1 Join - Student & Dorm",
        "question": "List student names and their dorm names.",
        "schema": "Student(StuID number, LName text, Fname text, Age number, Sex text, Major number, Advisor number, city_code text; PRIMARY KEY(StuID))\nDorm(dormid number, dorm_name text, student_capacity number, gender text)\nLives_in(stuid number, dormid number, room_number number; FK(stuid REFERENCES Student(StuID)); FK(dormid REFERENCES Dorm(dormid)))",
        "expected": "SELECT T1.Fname, T1.LName, T2.dorm_name FROM Student AS T1 INNER JOIN Lives_in AS T3 ON T1.StuID = T3.stuid INNER JOIN Dorm AS T2 ON T3.dormid = T2.dormid"
    },
    {
        "type": "Aggregation - Products",
        "question": "What is the average price of all products?",
        "schema": "Products(ProductID INTEGER, ProductName TEXT, CategoryID INTEGER, Unit TEXT, Price REAL)",
        "expected": "SELECT AVG(Price) FROM Products"
    },
    {
        "type": "GROUP BY - Sales",
        "question": "Count orders by customer.",
        "schema": "Customers(CustomerID INTEGER, CustomerName TEXT); Orders(OrderID INTEGER, CustomerID INTEGER, OrderDate TEXT; FK(CustomerID REFERENCES Customers(CustomerID)))",
        "expected": "SELECT T2.CustomerName, COUNT(*) FROM Orders AS T1 INNER JOIN Customers AS T2 ON T1.CustomerID = T2.CustomerID GROUP BY T2.CustomerName"
    },
    {
        "type": "2 Joins - Movies",
        "question": "List movie titles with their genres and directors.",
        "schema": "Movie(movie_id INTEGER, title TEXT, release_date DATE; PRIMARY KEY(movie_id))\nGenre(genre_id INTEGER, genre_name TEXT; PRIMARY KEY(genre_id))\nDirector(director_id INTEGER, director_name TEXT; PRIMARY KEY(director_id))\nMovie_Genre(movie_id INTEGER, genre_id INTEGER; FK(movie_id REFERENCES Movie(movie_id)); FK(genre_id REFERENCES Genre(genre_id)))\nMovie_Director(movie_id INTEGER, director_id INTEGER; FK(movie_id REFERENCES Movie(movie_id)); FK(director_id REFERENCES Director(director_id)))",
        "expected": "SELECT T1.title, T2.genre_name, T3.director_name FROM Movie AS T1 INNER JOIN Movie_Genre AS T4 ON T1.movie_id = T4.movie_id INNER JOIN Genre AS T2 ON T4.genre_id = T2.genre_id INNER JOIN Movie_Director AS T5 ON T1.movie_id = T5.movie_id INNER JOIN Director AS T3 ON T5.director_id = T3.director_id"
    },
    {
        "type": "Subquery - Products",
        "question": "Find products with price higher than average.",
        "schema": "Products(ProductID INTEGER, ProductName TEXT, Price REAL)",
        "expected": "SELECT ProductName FROM Products WHERE Price > (SELECT AVG(Price) FROM Products)"
    },
    {
        "type": "ORDER BY + LIMIT - Employees",
        "question": "Find the 5 highest paid employees.",
        "schema": "Employees(EmployeeID INTEGER, FirstName TEXT, LastName TEXT, Salary REAL)",
        "expected": "SELECT FirstName, LastName, Salary FROM Employees ORDER BY Salary DESC LIMIT 5"
    },
    {
        "type": "HAVING - Sales",
        "question": "Find customers who placed more than 5 orders.",
        "schema": "Orders(OrderID INTEGER, CustomerID INTEGER, OrderDate TEXT); Customers(CustomerID INTEGER, CustomerName TEXT; PRIMARY KEY(CustomerID))",
        "expected": "SELECT T2.CustomerName, COUNT(*) AS OrderCount FROM Orders AS T1 INNER JOIN Customers AS T2 ON T1.CustomerID = T2.CustomerID GROUP BY T2.CustomerName HAVING COUNT(*) > 5"
    },
    {
        "type": "Complex - HR Analytics",
        "question": "Find departments where the average salary is above company average.",
        "schema": "Employees(EmployeeID INTEGER, Name TEXT, Salary REAL, DepartmentID INTEGER); Departments(DepartmentID INTEGER, DepartmentName TEXT; PRIMARY KEY(DepartmentID))",
        "expected": "SELECT T2.DepartmentName, AVG(T1.Salary) AS AvgSalary FROM Employees AS T1 INNER JOIN Departments AS T2 ON T1.DepartmentID = T2.DepartmentID GROUP BY T2.DepartmentName HAVING AVG(T1.Salary) > (SELECT AVG(Salary) FROM Employees)"
    },
    {
        "type": "NOT EXISTS - Products",
        "question": "Find products that have never been ordered.",
        "schema": "Products(ProductID INTEGER, ProductName TEXT, Price REAL); Orders(OrderID INTEGER, ProductID INTEGER, Quantity INTEGER; FK(ProductID REFERENCES Products(ProductID)))",
        "expected": "SELECT ProductName FROM Products AS T1 WHERE NOT EXISTS (SELECT 1 FROM Orders AS T2 WHERE T2.ProductID = T1.ProductID)"
    }
]

# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_models():
    """Load base and fine-tuned models"""
    st.info("🔄 Loading models... (this may take a minute)")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    base_model.eval()
    
    # Load fine-tuned model
    finetuned_model = None
    if FINETUNED_PATH.exists():
        try:
            base_for_ft = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            finetuned_model = PeftModel.from_pretrained(base_for_ft, str(FINETUNED_PATH))
            finetuned_model.eval()
            st.success(f"✅ Fine-tuned model loaded from {FINETUNED_PATH.name}")
        except Exception as e:
            st.error(f"❌ Error loading fine-tuned model: {str(e)}")
    else:
        st.warning(f"⚠️ Fine-tuned model not found at: {FINETUNED_PATH}")
    
    # Check Llama3 availability
    llama_available = False
    try:
        tags_url = f"{OLLAMA_BASE}/api/tags"
        response = requests.get(tags_url, timeout=5)
        if response.status_code == 200:
            tags = response.json()
            names = {m.get("name") for m in tags.get("models", [])}
            if OLLAMA_MODEL in names:
                llama_available = True
                st.success(f"✅ Llama3 ({OLLAMA_MODEL}) available via Ollama")
            else:
                st.warning(f"⚠️ Llama3 model '{OLLAMA_MODEL}' not found. Run: ollama pull {OLLAMA_MODEL}")
        else:
            st.warning("⚠️ Ollama server not responding")
    except Exception as e:
        st.warning(f"⚠️ Llama3 check failed: {str(e)}")
    
    st.success("✅ Models ready!")
    return tokenizer, base_model, finetuned_model, llama_available

# ============================================================================
# SQL GENERATION
# ============================================================================

def generate_sql_local(model, tokenizer, question, schema, max_tokens=512, use_pruning=False):
    """Generate SQL using local model (Base or Fine-tuned)
    
    Args:
        model: The model to use
        tokenizer: The tokenizer
        question: Natural language question
        schema: Database schema
        max_tokens: Maximum tokens to generate
        use_pruning: Whether to apply schema pruning (for fine-tuned model only)
    """
    # Apply schema pruning if requested
    effective_schema = schema
    if use_pruning and schema:
        try:
            effective_schema = prune_schema(
                schema,
                question,
                min_tables=SCHEMA_MIN_TABLES,
                max_columns_per_table=SCHEMA_MAX_COLUMNS
            )
        except Exception as e:
            st.warning(f"⚠️ Schema pruning failed: {str(e)}. Using full schema.")
            effective_schema = schema
    
    prompt = format_prompt(question, effective_schema)
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode only generated part
    input_length = inputs['input_ids'].shape[1]
    generated_ids = outputs[0][input_length:]
    full_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    # Clean output - extract SQL query
    sql = full_output
    
    # Remove markdown code blocks
    if "```sql" in sql:
        sql = sql.split("```sql")[1].split("```")[0].strip()
    elif "```" in sql:
        sql = sql.split("```")[1].split("```")[0].strip()
    
    # Stop at markers
    for marker in ["Human:", "Assistant:", "\n\n[", "[INSTRUCTION]", "[DATABASE", "[QUESTION]"]:
        if marker in sql:
            sql = sql.split(marker)[0].strip()
    
    return sql

def generate_sql_llama(question, schema):
    """Generate SQL using Llama3 via Ollama (same as evaluate_models_comparison.py)"""
    if not st.session_state.get('llama_available', False):
        return "ERROR: Llama3 not available", "Llama3 (unavailable)"
    
    try:
        prompt = format_prompt(question, schema)
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
# MAIN UI
# ============================================================================

st.title("🔍 SQL Generation Comparison")
st.markdown("**Compare Base vs Fine-tuned (No Prune) vs Fine-tuned (Prune) vs Llama3**")
st.markdown("*Inspired by evaluate_models_comparison.py - Tests schema pruning impact*")

# Load models
tokenizer, base_model, finetuned_model, llama_available = load_models()
st.session_state['llama_available'] = llama_available

# Sidebar - Test selection
st.sidebar.header("📝 Select Test")

# Option to test all queries
test_all = st.sidebar.checkbox("🔬 Test All Query Types", value=False)

if not test_all:
    selected_test = st.sidebar.selectbox(
        "Choose a test question:",
        [q["type"] for q in DEFAULT_QUESTIONS]
    )
    
    # Get selected question
    test_data = next(q for q in DEFAULT_QUESTIONS if q["type"] == selected_test)
    
    # Custom question option
    use_custom = st.sidebar.checkbox("Use Custom Question")
    
    if use_custom:
        st.sidebar.subheader("Custom Input")
        question = st.sidebar.text_area("Question:", "List all product names")
        schema = st.sidebar.text_area("Schema:", "Products(ProductID INTEGER, ProductName TEXT)")
        expected = None
    else:
        question = test_data["question"]
        schema = test_data["schema"]
        expected = test_data.get("expected")

# Display inputs
if not test_all:
    st.header("📋 Input")
    
    st.markdown("### ❓ Question")
    with st.container(border=True):
        st.markdown(question)
    
    st.markdown("### 🗄️ Database Schema")
    with st.container(border=True):
        st.code(schema, language="sql")
    
    if expected:
        st.markdown("### ⭐ Expected SQL (Ground Truth)")
        with st.container(border=True):
            st.code(expected, language="sql")

# Generate button
if test_all:
    if st.button("🚀 Test All Query Types", type="primary"):
        st.header("📊 Complete Test Results")
        
        all_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, test in enumerate(DEFAULT_QUESTIONS):
            status_text.text(f"Testing {idx+1}/{len(DEFAULT_QUESTIONS)}: {test['type']}")
            
            # Generate with all models
            with st.spinner(f"Base model..."):
                base_sql = generate_sql_local(base_model, tokenizer, test['question'], test['schema'])
            
            ft_no_prune_sql = ""
            ft_prune_sql = ""
            if finetuned_model:
                with st.spinner(f"Fine-tuned (no prune)..."):
                    ft_no_prune_sql = generate_sql_local(finetuned_model, tokenizer, test['question'], test['schema'], use_pruning=False)
                with st.spinner(f"Fine-tuned (prune)..."):
                    ft_prune_sql = generate_sql_local(finetuned_model, tokenizer, test['question'], test['schema'], use_pruning=True)
            
            llama_sql = ""
            if llama_available:
                with st.spinner(f"Llama3..."):
                    llama_sql, _ = generate_sql_llama(test['question'], test['schema'])
            
            # Check correctness if expected SQL available
            base_correct = None
            ft_no_prune_correct = None
            ft_prune_correct = None
            llama_correct = None
            if 'expected' in test:
                base_correct = simple_sql_match(base_sql, test['expected'])
                if ft_no_prune_sql:
                    ft_no_prune_correct = simple_sql_match(ft_no_prune_sql, test['expected'])
                if ft_prune_sql:
                    ft_prune_correct = simple_sql_match(ft_prune_sql, test['expected'])
                if llama_sql and not llama_sql.startswith("ERROR"):
                    llama_correct = simple_sql_match(llama_sql, test['expected'])
            
            all_results.append({
                "Query Type": test['type'],
                "Base SQL": base_sql[:50] + "..." if len(base_sql) > 50 else base_sql,
                "FT(NoPrune)": ft_no_prune_sql[:50] + "..." if len(ft_no_prune_sql) > 50 else ft_no_prune_sql if ft_no_prune_sql else "N/A",
                "FT(Prune)": ft_prune_sql[:50] + "..." if len(ft_prune_sql) > 50 else ft_prune_sql if ft_prune_sql else "N/A",
                "Llama3": llama_sql[:50] + "..." if len(llama_sql) > 50 else llama_sql if llama_sql else "N/A",
                "Base ✓": "✅" if base_correct else "❌" if base_correct is False else "-",
                "FT(NoPrune) ✓": "✅" if ft_no_prune_correct else "❌" if ft_no_prune_correct is False else "-",
                "FT(Prune) ✓": "✅" if ft_prune_correct else "❌" if ft_prune_correct is False else "-",
                "Llama3 ✓": "✅" if llama_correct else "❌" if llama_correct is False else "-"
            })
            
            progress_bar.progress((idx + 1) / len(DEFAULT_QUESTIONS))
        
        status_text.text("✅ All tests complete!")
        
        # Display results table
        st.markdown("### 📋 Results Summary")
        df = pd.DataFrame(all_results)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Statistics
        st.divider()
        st.markdown("### 📈 Statistics")
        
        base_correct_count = sum(1 for r in all_results if r['Base ✓'] == '✅')
        ft_no_prune_correct_count = sum(1 for r in all_results if r['FT(NoPrune) ✓'] == '✅')
        ft_prune_correct_count = sum(1 for r in all_results if r['FT(Prune) ✓'] == '✅')
        llama_correct_count = sum(1 for r in all_results if r['Llama3 ✓'] == '✅')
        total = len(all_results)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Tests", total)
        with col2:
            st.metric("Base", base_correct_count, f"{base_correct_count/total*100:.1f}%")
        with col3:
            st.metric("FT(NoPrune)", ft_no_prune_correct_count, f"{ft_no_prune_correct_count/total*100:.1f}%")
        with col4:
            st.metric("FT(Prune)", ft_prune_correct_count, f"{ft_prune_correct_count/total*100:.1f}%")
        with col5:
            st.metric("Llama3", llama_correct_count, f"{llama_correct_count/total*100:.1f}%")
        
        # Pruning comparison
        if ft_no_prune_correct_count > 0 or ft_prune_correct_count > 0:
            st.divider()
            diff = ft_prune_correct_count - ft_no_prune_correct_count
            diff_pct = (ft_prune_correct_count - ft_no_prune_correct_count) / total * 100
            if diff > 0:
                st.success(f"✅ Schema Pruning HELPS: +{diff} correct ({diff_pct:+.1f}%)")
            elif diff < 0:
                st.error(f"❌ Schema Pruning HURTS: {diff} correct ({diff_pct:.1f}%)")
            else:
                st.info(f"➖ Schema Pruning has NO EFFECT")

elif st.button("🚀 Generate SQL (All Models)", type="primary"):
    st.header("📊 Results")
    
    # Base Model
    st.markdown("### 🔵 Base Model (Qwen 2.5-1.5B)")
    with st.container(border=True):
        with st.spinner("Generating..."):
            start = time.time()
            base_sql = generate_sql_local(base_model, tokenizer, question, schema)
            base_time = time.time() - start
        
        st.code(base_sql, language="sql")
        
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"⏱️ **Time:** {base_time:.2f}s")
        with col2:
            if expected:
                base_correct = simple_sql_match(base_sql, expected)
                st.caption(f"✅ **Correct:** {'Yes' if base_correct else 'No'}")
    
    st.divider()
    
    # Fine-tuned Model WITHOUT Pruning
    st.markdown("### 🟢 Fine-tuned Model (NO Pruning) - 5K dataset, checkpoint-3000")
    with st.container(border=True):
        if finetuned_model:
            with st.spinner("Generating without schema pruning..."):
                start = time.time()
                ft_no_prune_sql = generate_sql_local(finetuned_model, tokenizer, question, schema, use_pruning=False)
                ft_no_prune_time = time.time() - start
            
            st.code(ft_no_prune_sql, language="sql")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"⏱️ **Time:** {ft_no_prune_time:.2f}s")
            with col2:
                st.caption(f"📏 **Length:** {len(ft_no_prune_sql)} chars")
            with col3:
                if expected:
                    ft_no_prune_correct = simple_sql_match(ft_no_prune_sql, expected)
                    st.caption(f"✅ **Correct:** {'Yes' if ft_no_prune_correct else 'No'}")
        else:
            st.error("Model not loaded")
    
    st.divider()
    
    # Fine-tuned Model WITH Pruning
    st.markdown("### 🟢 Fine-tuned Model (WITH Pruning) - 5K dataset, checkpoint-3000")
    with st.container(border=True):
        if finetuned_model:
            # Show pruned schema
            try:
                pruned_schema = prune_schema(schema, question, min_tables=SCHEMA_MIN_TABLES, max_columns_per_table=SCHEMA_MAX_COLUMNS)
                with st.expander("🔍 View Pruned Schema"):
                    st.code(pruned_schema, language="sql")
                    st.caption(f"Original: {len(schema)} chars → Pruned: {len(pruned_schema)} chars")
            except Exception as e:
                st.warning(f"⚠️ Could not show pruned schema: {str(e)}")
            
            with st.spinner("Generating with schema pruning..."):
                start = time.time()
                ft_prune_sql = generate_sql_local(finetuned_model, tokenizer, question, schema, use_pruning=True)
                ft_prune_time = time.time() - start
            
            st.code(ft_prune_sql, language="sql")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"⏱️ **Time:** {ft_prune_time:.2f}s")
            with col2:
                st.caption(f"📏 **Length:** {len(ft_prune_sql)} chars")
            with col3:
                if expected:
                    ft_prune_correct = simple_sql_match(ft_prune_sql, expected)
                    st.caption(f"✅ **Correct:** {'Yes' if ft_prune_correct else 'No'}")
            
            # Compare pruning vs no pruning
            if expected and finetuned_model:
                st.divider()
                if ft_no_prune_correct is not None and ft_prune_correct is not None:
                    if ft_no_prune_correct != ft_prune_correct:
                        if ft_prune_correct and not ft_no_prune_correct:
                            st.success("✅ **Pruning IMPROVES result** (correct with pruning, wrong without)")
                        elif ft_no_prune_correct and not ft_prune_correct:
                            st.error("❌ **Pruning HURTS result** (correct without pruning, wrong with)")
                    else:
                        st.info("➖ **Pruning has same correctness**")
        else:
            st.error("Model not loaded")
    
    st.divider()
    
    # Llama3
    st.markdown("### 🔴 Llama3 (llama3:latest via Ollama)")
    with st.container(border=True):
        if llama_available:
            with st.spinner("Generating..."):
                start = time.time()
                llama_sql, llama_model_name = generate_sql_llama(question, schema)
                llama_time = time.time() - start
            
            st.code(llama_sql, language="sql")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"⏱️ **Time:** {llama_time:.2f}s")
            with col2:
                st.caption(f"📏 **Length:** {len(llama_sql)} chars")
            with col3:
                if expected and not llama_sql.startswith("ERROR"):
                    llama_correct = simple_sql_match(llama_sql, expected)
                    st.caption(f"✅ **Correct:** {'Yes' if llama_correct else 'No'}")
        else:
            st.error("Llama3 not available")
    
    # Summary comparison
    st.divider()
    st.markdown("### 📈 Performance Summary")
    
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        st.metric("🔵 Base Model", f"{base_time:.2f}s", help="Qwen 2.5-1.5B base model")
    with metrics_col2:
        if finetuned_model:
            st.metric("🟢 FT(NoPrune)", f"{ft_no_prune_time:.2f}s", delta=f"{ft_no_prune_time-base_time:+.2f}s", help="Fine-tuned without pruning")
    with metrics_col3:
        if finetuned_model:
            st.metric("🟢 FT(Prune)", f"{ft_prune_time:.2f}s", delta=f"{ft_prune_time-base_time:+.2f}s", help="Fine-tuned with pruning")
    with metrics_col4:
        if llama_available:
            st.metric("🔴 Llama3", f"{llama_time:.2f}s", delta=f"{llama_time-base_time:+.2f}s", help="llama3:latest via Ollama")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Query Types Available:")
for q in DEFAULT_QUESTIONS:
    st.sidebar.markdown(f"- {q['type']}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 💾 Results Saving")
st.sidebar.info("Full evaluation results are saved in `evaluate_models_comparison.py` to `outputs/jsonl/final_evaluation_100.json`")

