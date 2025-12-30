#!/usr/bin/env python3
"""
Streamlit App: Compare Base vs Fine-tuned vs Gemini for Text-to-SQL
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

# Set page config
st.set_page_config(
    page_title="SQL Generation Comparison",
    page_icon="🔍",
    layout="wide"
)

# Paths
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
FINETUNED_PATH = "outputs/checkpoints/qwen_optimized_lora_2k_20251218_091402/checkpoint-1250"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

def normalize_sql(sql):
    """Normalize SQL for semantic comparison"""
    if not sql or sql.startswith('⚠️') or sql.startswith('ERROR'):
        return ""
    sql = sql.upper().strip()
    sql = re.sub(r'\s+', ' ', sql)
    sql = sql.rstrip(';')
    return sql

def check_semantic_similarity(sql1, sql2):
    """Check if two SQL queries are semantically similar"""
    norm1 = normalize_sql(sql1)
    norm2 = normalize_sql(sql2)
    if not norm1 or not norm2:
        return False, 0.0
    
    def extract_components(sql):
        return {
            'has_select': 'SELECT' in sql,
            'has_where': 'WHERE' in sql,
            'has_join': 'JOIN' in sql,
            'has_group_by': 'GROUP BY' in sql,
            'has_order_by': 'ORDER BY' in sql,
            'has_limit': 'LIMIT' in sql,
            'has_count': 'COUNT(' in sql,
            'has_sum': 'SUM(' in sql,
            'has_avg': 'AVG(' in sql,
            'has_not_exists': 'NOT EXISTS' in sql,
            'has_subquery': '(SELECT' in sql,
        }
    
    comp1 = extract_components(norm1)
    comp2 = extract_components(norm2)
    matches = sum(1 for k in comp1 if comp1[k] == comp2[k])
    similarity = matches / len(comp1)
    return similarity >= 0.7, similarity

DEFAULT_QUESTIONS = [
    {
        "type": "Simple (No Join) - Products",
        "question": "List all product names.",
        "schema": "Products(ProductID INTEGER, ProductName TEXT, CategoryID INTEGER, Unit TEXT, Price REAL)",
        "expected": "SELECT ProductName FROM Products"
    },
    {
        "type": "Simple - HR",
        "question": "List all employee names and their salaries.",
        "schema": "Employees(EmployeeID INTEGER, FirstName TEXT, LastName TEXT, Salary REAL, DepartmentID INTEGER)",
        "expected": "SELECT FirstName, LastName, Salary FROM Employees"
    },
    {
        "type": "1 Join - Products",
        "question": "List all product names and their category names.",
        "schema": "Products(ProductID INTEGER, ProductName TEXT, CategoryID INTEGER); Categories(CategoryID INTEGER, CategoryName TEXT)",
        "expected": "SELECT T1.ProductName, T2.CategoryName FROM Products AS T1 INNER JOIN Categories AS T2 ON T1.CategoryID = T2.CategoryID"
    },
    {
        "type": "1 Join - HR",
        "question": "List all employees with their department names.",
        "schema": "Employees(EmployeeID INTEGER, FirstName TEXT, LastName TEXT, DepartmentID INTEGER); Departments(DepartmentID INTEGER, DepartmentName TEXT)",
        "expected": "SELECT T1.FirstName, T1.LastName, T2.DepartmentName FROM Employees AS T1 INNER JOIN Departments AS T2 ON T1.DepartmentID = T2.DepartmentID"
    },
    {
        "type": "2 Joins - Sales",
        "question": "List customer names, their orders, and product names.",
        "schema": "Customers(CustomerID INTEGER, CustomerName TEXT); Orders(OrderID INTEGER, CustomerID INTEGER, ProductID INTEGER); Products(ProductID INTEGER, ProductName TEXT)",
        "expected": "SELECT T1.CustomerName, T2.OrderID, T3.ProductName FROM Customers AS T1 INNER JOIN Orders AS T2 ON T1.CustomerID = T2.CustomerID INNER JOIN Products AS T3 ON T2.ProductID = T3.ProductID"
    },
    {
        "type": "2 Joins - HR & Projects",
        "question": "List employee names, their projects, and project managers.",
        "schema": "Employees(EmployeeID INTEGER, Name TEXT, ManagerID INTEGER); Projects(ProjectID INTEGER, ProjectName TEXT, ManagerID INTEGER); Managers(ManagerID INTEGER, ManagerName TEXT)",
        "expected": "SELECT T1.Name AS Employee, T2.ProjectName, T3.ManagerName FROM Employees AS T1 INNER JOIN Projects AS T2 ON T1.ManagerID = T2.ManagerID INNER JOIN Managers AS T3 ON T2.ManagerID = T3.ManagerID"
    },
    {
        "type": "Aggregation - Sales",
        "question": "What is the average price of all products?",
        "schema": "Products(ProductID INTEGER, ProductName TEXT, Price REAL)",
        "expected": "SELECT AVG(Price) FROM Products"
    },
    {
        "type": "Aggregation - HR",
        "question": "What is the total payroll cost for all employees?",
        "schema": "Employees(EmployeeID INTEGER, Name TEXT, Salary REAL)",
        "expected": "SELECT SUM(Salary) FROM Employees"
    },
    {
        "type": "GROUP BY - Products",
        "question": "Count products in each category.",
        "schema": "Products(ProductID INTEGER, ProductName TEXT, CategoryID INTEGER); Categories(CategoryID INTEGER, CategoryName TEXT)",
        "expected": "SELECT T2.CategoryName, COUNT(*) FROM Products AS T1 INNER JOIN Categories AS T2 ON T1.CategoryID = T2.CategoryID GROUP BY T2.CategoryName"
    },
    {
        "type": "GROUP BY + HAVING - Sales",
        "question": "Find customers who placed more than 5 orders.",
        "schema": "Orders(OrderID INTEGER, CustomerID INTEGER, OrderDate TEXT); Customers(CustomerID INTEGER, CustomerName TEXT)",
        "expected": "SELECT T2.CustomerName, COUNT(*) AS OrderCount FROM Orders AS T1 INNER JOIN Customers AS T2 ON T1.CustomerID = T2.CustomerID GROUP BY T2.CustomerName HAVING COUNT(*) > 5"
    },
    {
        "type": "Subquery - Products",
        "question": "Find products with price higher than average.",
        "schema": "Products(ProductID INTEGER, ProductName TEXT, Price REAL)",
        "expected": "SELECT ProductName FROM Products WHERE Price > (SELECT AVG(Price) FROM Products)"
    },
    {
        "type": "Subquery - HR",
        "question": "Find employees earning more than the average salary in their department.",
        "schema": "Employees(EmployeeID INTEGER, Name TEXT, Salary REAL, DepartmentID INTEGER)",
        "expected": "SELECT Name FROM Employees AS T1 WHERE Salary > (SELECT AVG(Salary) FROM Employees AS T2 WHERE T2.DepartmentID = T1.DepartmentID)"
    },
    {
        "type": "ORDER BY + LIMIT - Products",
        "question": "List top 3 most expensive products.",
        "schema": "Products(ProductID INTEGER, ProductName TEXT, Price REAL)",
        "expected": "SELECT ProductName, Price FROM Products ORDER BY Price DESC LIMIT 3"
    },
    {
        "type": "ORDER BY + LIMIT - HR",
        "question": "Find the 5 highest paid employees.",
        "schema": "Employees(EmployeeID INTEGER, FirstName TEXT, LastName TEXT, Salary REAL)",
        "expected": "SELECT FirstName, LastName, Salary FROM Employees ORDER BY Salary DESC LIMIT 5"
    },
    {
        "type": "Complex - Business Analytics",
        "question": "Find customers who ordered products from the highest priced category.",
        "schema": "Customers(CustomerID INTEGER, CustomerName TEXT); Orders(OrderID INTEGER, CustomerID INTEGER, ProductID INTEGER); Products(ProductID INTEGER, ProductName TEXT, CategoryID INTEGER, Price REAL); Categories(CategoryID INTEGER, CategoryName TEXT)",
        "expected": "SELECT DISTINCT T1.CustomerName FROM Customers AS T1 INNER JOIN Orders AS T2 ON T1.CustomerID = T2.CustomerID INNER JOIN Products AS T3 ON T2.ProductID = T3.ProductID INNER JOIN Categories AS T4 ON T3.CategoryID = T4.CategoryID WHERE T4.CategoryID = (SELECT CategoryID FROM Products GROUP BY CategoryID ORDER BY AVG(Price) DESC LIMIT 1)"
    },
    {
        "type": "Complex - HR Analytics",
        "question": "Find departments where the average salary is above company average.",
        "schema": "Employees(EmployeeID INTEGER, Name TEXT, Salary REAL, DepartmentID INTEGER); Departments(DepartmentID INTEGER, DepartmentName TEXT)",
        "expected": "SELECT T2.DepartmentName, AVG(T1.Salary) AS AvgSalary FROM Employees AS T1 INNER JOIN Departments AS T2 ON T1.DepartmentID = T2.DepartmentID GROUP BY T2.DepartmentName HAVING AVG(T1.Salary) > (SELECT AVG(Salary) FROM Employees)"
    }
]

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
        dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    base_model.eval()
    
    # Load fine-tuned model
    finetuned_path = Path(FINETUNED_PATH)
    if finetuned_path.exists():
        base_for_ft = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        finetuned_model = PeftModel.from_pretrained(base_for_ft, str(finetuned_path))
        finetuned_model.eval()
    else:
        st.error(f"❌ Fine-tuned model not found at: {finetuned_path}")
        finetuned_model = None
    
    st.success("✅ Models loaded!")
    return tokenizer, base_model, finetuned_model

def generate_sql(model, tokenizer, question, schema, max_tokens=100):
    """Generate SQL using local model"""
    prompt = f"""[INSTRUCTION]
You are a SQL expert. Generate a valid SQL query based on the question and database schema.

[DATABASE SCHEMA]
{schema}

[QUESTION]
{question}

[SQL QUERY]
"""
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3
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
    
    # Stop at these markers
    for marker in ["Human:", "Assistant:", "\n\n[", "[INSTRUCTION]", "[DATABASE", "[QUESTION]"]:
        if marker in sql:
            sql = sql.split(marker)[0].strip()
    
    lines = [l.strip() for l in sql.split('\n') if l.strip()]
    if lines:
        first_line = lines[0]
        if any(first_line.upper().startswith(kw) for kw in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER', 'WITH']):
            sql_lines = []
            for line in lines:
                if line and not line.startswith('#') and not line.startswith('--') and not line.lower().startswith(('note:', 'explanation:', 'this query')):
                    sql_lines.append(line)
                else:
                    break
            sql = ' '.join(sql_lines)
    
    return sql

def generate_sql_gemini(question, schema):
    """Generate SQL using Gemini API"""
    if not GEMINI_API_KEY:
        return "⚠️ Gemini API key not configured. Set GEMINI_API_KEY environment variable."
    
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        prompt = f"""You are a SQL expert. Generate ONLY a valid SQL query based on the question and database schema. Do not include any explanations or markdown.

Database Schema:
{schema}

Question:
{question}

SQL Query:"""
        
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        sql = response.text.strip()
        sql = sql.replace("```sql", "").replace("```", "").strip()
        for marker in ["SQL:", "Query:", "Answer:"]:
            if marker in sql:
                sql = sql.split(marker)[-1].strip()
        return sql
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            return "⚠️ Gemini API quota exceeded."
        elif "API key" in error_msg:
            return "⚠️ Invalid API key."
        else:
            return f"⚠️ Gemini Error: {error_msg[:100]}"

st.title("🔍 SQL Generation Comparison")
st.markdown("**Compare Base Model vs Fine-tuned vs Gemini**")

# Load models
tokenizer, base_model, finetuned_model = load_models()

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
    else:
        question = test_data["question"]
        schema = test_data["schema"]

# Display inputs
if not test_all:
    st.header("📋 Input")
    
    st.markdown("### ❓ Question")
    with st.container(border=True):
        st.markdown(question)
    
    st.markdown("### 🗄️ Database Schema")
    with st.container(border=True):
        st.code(schema, language="sql")

# Generate button
if test_all:
    if st.button("🚀 Test All Query Types", type="primary"):
        st.header("📊 Complete Test Results")
        
        all_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, test in enumerate(DEFAULT_QUESTIONS):
            status_text.text(f"Testing {idx+1}/{len(DEFAULT_QUESTIONS)}: {test['type']}")
            
            # Generate with all three models
            base_sql = generate_sql(base_model, tokenizer, test['question'], test['schema'])
            ft_sql = generate_sql(finetuned_model, tokenizer, test['question'], test['schema']) if finetuned_model else ""
            gemini_sql = generate_sql_gemini(test['question'], test['schema'])
            
            # Check similarities
            base_vs_ft_similar, base_vs_ft_score = check_semantic_similarity(base_sql, ft_sql)
            ft_vs_gemini_similar, ft_vs_gemini_score = check_semantic_similarity(ft_sql, gemini_sql)
            
            all_results.append({
                "Query Type": test['type'],
                "Base": base_sql[:50] + "..." if len(base_sql) > 50 else base_sql,
                "Fine-tuned": ft_sql[:50] + "..." if len(ft_sql) > 50 else ft_sql,
                "Gemini": gemini_sql[:50] + "..." if len(gemini_sql) > 50 else gemini_sql,
                "FT vs Gemini": f"{ft_vs_gemini_score*100:.0f}%",
                "Match": "✅" if ft_vs_gemini_similar else "❌"
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
        
        matches = sum(1 for r in all_results if r['Match'] == '✅')
        total = len(all_results)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Tests", total)
        with col2:
            st.metric("Matches", matches)
        with col3:
            st.metric("Match Rate", f"{matches/total*100:.1f}%")

elif st.button("🚀 Generate SQL", type="primary"):
    st.header("📊 Results")
    
    # Show expected first if available
    if not use_custom:
        st.markdown("### ⭐ Expected SQL")
        with st.container(border=True):
            st.code(test_data["expected"], language="sql")
        st.divider()
    
    st.markdown("### 🔵 Base Model")
    with st.container(border=True):
        with st.spinner("Generating..."):
            start = time.time()
            base_sql = generate_sql(base_model, tokenizer, question, schema)
            base_time = time.time() - start
        
        st.code(base_sql, language="sql")
        
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"⏱️ **Time:** {base_time:.2f}s")
        with col2:
            st.caption(f"📏 **Length:** {len(base_sql)} chars")
    
    st.divider()
    
    st.markdown("### 🟢 Fine-tuned Model")
    with st.container(border=True):
        if finetuned_model:
            with st.spinner("Generating..."):
                start = time.time()
                ft_sql = generate_sql(finetuned_model, tokenizer, question, schema)
                ft_time = time.time() - start
            
            st.code(ft_sql, language="sql")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"⏱️ **Time:** {ft_time:.2f}s")
            with col2:
                st.caption(f"📏 **Length:** {len(ft_sql)} chars")
            with col3:
                st.caption(f"⚡ **vs Base:** {ft_time-base_time:+.2f}s")
        else:
            st.error("Model not loaded")
    
    st.divider()
    
    st.markdown("### 🟣 Gemini")
    with st.container(border=True):
        with st.spinner("Generating..."):
            start = time.time()
            gemini_sql = generate_sql_gemini(question, schema)
            gemini_time = time.time() - start
        
        st.code(gemini_sql, language="sql")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption(f"⏱️ **Time:** {gemini_time:.2f}s")
        with col2:
            st.caption(f"📏 **Length:** {len(gemini_sql)} chars")
        with col3:
            st.caption(f"⚡ **vs Base:** {gemini_time-base_time:+.2f}s")
    
    st.divider()
    st.markdown("### 📈 Performance Summary")
    
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    with metrics_col1:
        st.metric("🔵 Base", f"{base_time:.2f}s")
    with metrics_col2:
        if finetuned_model:
            st.metric("🟢 Fine-tuned", f"{ft_time:.2f}s", delta=f"{ft_time-base_time:+.2f}s")
    with metrics_col3:
        st.metric("🟣 Gemini", f"{gemini_time:.2f}s", delta=f"{gemini_time-base_time:+.2f}s")
    
    st.divider()
    st.markdown("### 🎯 Similarity Analysis")
    
    comparison_data = []
    if not base_sql.startswith('⚠️'):
        is_similar, similarity = check_semantic_similarity(base_sql, ft_sql)
        comparison_data.append({
            "Comparison": "Base vs Fine-tuned",
            "Similarity": f"{similarity*100:.1f}%",
            "Match": "✅ Yes" if is_similar else "❌ No"
        })
    if not gemini_sql.startswith('⚠️'):
        is_similar, similarity = check_semantic_similarity(base_sql, gemini_sql)
        comparison_data.append({
            "Comparison": "Base vs Gemini",
            "Similarity": f"{similarity*100:.1f}%",
            "Match": "✅ Yes" if is_similar else "❌ No"
        })
    if not gemini_sql.startswith('⚠️'):
        is_similar, similarity = check_semantic_similarity(ft_sql, gemini_sql)
        comparison_data.append({
            "Comparison": "Fine-tuned vs Gemini",
            "Similarity": f"{similarity*100:.1f}%",
            "Match": "✅ Yes" if is_similar else "❌ No"
        })
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Query Types:")
for q in DEFAULT_QUESTIONS:
    st.sidebar.markdown(f"- {q['type']}")
