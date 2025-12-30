#!/usr/bin/env python3
"""
Interactive Streamlit Demo for Fine-Tuned Text2SQL Model

Features:
- Test your LoRA fine-tuned model interactively
- Compare base model vs fine-tuned model
- Visualize SQL generation in real-time
- Test custom questions and schemas
"""

import streamlit as st
import torch
import time
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_ADAPTER = CHECKPOINT_DIR / "qwen_1.5b_text2sql_lora_final"

# ============================================================================
# MODEL LOADING (CACHED)
# ============================================================================

@st.cache_resource
def load_fine_tuned_model():
    """Load fine-tuned model with LoRA adapter"""
    
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        padding_side="right"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    
    model = PeftModel.from_pretrained(
        base_model,
        str(LORA_ADAPTER),
        torch_dtype=torch.float32
    )
    
    model.eval()
    
    return model, tokenizer


@st.cache_resource
def load_base_model():
    """Load base model without LoRA"""
    
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        padding_side="right"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    
    model.eval()
    
    return model, tokenizer


# ============================================================================
# INFERENCE
# ============================================================================

def generate_sql(model, tokenizer, question: str, schema: str) -> tuple:
    """Generate SQL from question and schema"""
    
    prompt = f"""[INSTRUCTION]
You are a SQL expert. Generate a valid SQL query based on the question and database schema.

[DATABASE SCHEMA]
{schema}

[QUESTION]
{question}

[SQL QUERY]
"""
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=False
    )
    
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=150,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3
        )
    
    generation_time = time.time() - start_time
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "[SQL QUERY]" in generated_text:
        sql = generated_text.split("[SQL QUERY]")[-1].strip()
    else:
        sql = generated_text[len(prompt):].strip()
    
    # Clean up
    import re
    sql = re.sub(r'```sql\s*', '', sql)
    sql = re.sub(r'```\s*', '', sql)
    
    if ';' in sql:
        sql = sql.split(';')[0].strip() + ';'
    else:
        sql = sql.split('\n')[0].strip()
    
    for stop_word in ['Human:', 'Assistant:', 'Question:', 'Write', 'Create', '[']:
        if stop_word in sql:
            sql = sql.split(stop_word)[0].strip()
    
    return sql, generation_time


# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.set_page_config(
        page_title="Text2SQL Demo - LoRA Fine-Tuned",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Text2SQL Interactive Demo")
    st.markdown("### Fine-Tuned Model with LoRA Adapter")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.markdown("### 📊 Model Info")
        st.info(f"**Base Model:** {BASE_MODEL.split('/')[-1]}")
        st.info(f"**LoRA Adapter:** qwen_1.5b_text2sql_lora_final")
        st.success(f"**Status:** Loaded ✅")
    
    # Load both models
    with st.spinner("Loading models..."):
        try:
            fine_tuned_model, fine_tuned_tokenizer = load_fine_tuned_model()
            st.sidebar.success("Fine-tuned model loaded!")
            
            base_model, base_tokenizer = load_base_model()
            st.sidebar.success("Base model loaded!")
        except Exception as e:
            st.error(f"❌ Error loading models: {e}")
            st.stop()
    
    # Example schemas
    example_schemas = {
        "Student Database": """Student(student_id PK, name VARCHAR, age INT, major VARCHAR, gpa FLOAT);
Enrollment(enrollment_id PK, student_id FK, course_id FK, grade VARCHAR, semester VARCHAR);
Course(course_id PK, course_name VARCHAR, credits INT, department VARCHAR);""",
        
        "E-commerce": """Customer(customer_id PK, name VARCHAR, email VARCHAR, city VARCHAR);
Product(product_id PK, product_name VARCHAR, price DECIMAL, category VARCHAR);
Order(order_id PK, customer_id FK, order_date DATE, total_amount DECIMAL);
OrderItem(item_id PK, order_id FK, product_id FK, quantity INT, price DECIMAL);""",
        
        "Company HR": """Employee(emp_id PK, name VARCHAR, salary DECIMAL, dept_id FK, hire_date DATE);
Department(dept_id PK, dept_name VARCHAR, location VARCHAR, budget DECIMAL);
Project(project_id PK, project_name VARCHAR, dept_id FK, start_date DATE, end_date DATE);"""
    }
    
    example_questions = {
        "Student Database": [
            "What are the names of all students?",
            "How many students are in each major?",
            "What is the average GPA by department?",
            "Which students have a GPA above 3.5?"
        ],
        "E-commerce": [
            "List all customers from New York",
            "What is the total revenue by product category?",
            "Which products have never been ordered?",
            "What is the average order value per customer?"
        ],
        "Company HR": [
            "Find the average salary by department",
            "Which employees were hired in 2023?",
            "List all projects in the IT department",
            "What is the total budget across all departments?"
        ]
    }
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Input")
        
        # Schema selection
        schema_type = st.selectbox(
            "Select Example Schema",
            ["Custom"] + list(example_schemas.keys())
        )
        
        if schema_type == "Custom":
            schema = st.text_area(
                "Database Schema",
                height=150,
                placeholder="Enter your database schema here...\nExample: Table1(col1 PK, col2 VARCHAR);"
            )
        else:
            schema = st.text_area(
                "Database Schema",
                value=example_schemas[schema_type],
                height=150
            )
        
        # Question selection
        if schema_type != "Custom" and schema_type in example_questions:
            question_example = st.selectbox(
                "Select Example Question",
                ["Custom"] + example_questions[schema_type]
            )
            
            if question_example == "Custom":
                question = st.text_input("Your Question", placeholder="Enter your question here...")
            else:
                question = st.text_input("Your Question", value=question_example)
        else:
            question = st.text_input("Your Question", placeholder="Enter your question here...")
        
        generate_button = st.button("🚀 Generate SQL", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("🔮 Generated SQL")
        
        if generate_button:
            if not question or not schema:
                st.warning("⚠️ Please provide both a question and schema!")
            else:
                # Fine-tuned model
                with st.spinner("Generating with fine-tuned model..."):
                    try:
                        sql_fine_tuned, time_fine_tuned = generate_sql(
                            fine_tuned_model,
                            fine_tuned_tokenizer,
                            question,
                            schema
                        )
                        
                        st.markdown("#### ✨ Fine-Tuned Model (LoRA)")
                        st.code(sql_fine_tuned, language="sql")
                        st.caption(f"⏱️ Generation time: {time_fine_tuned:.3f}s")
                        
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                
                # Base model
                st.markdown("---")
                with st.spinner("Generating with base model..."):
                    try:
                        sql_base, time_base = generate_sql(
                            base_model,
                            base_tokenizer,
                            question,
                            schema
                        )
                        
                        st.markdown("#### 📦 Base Model (No Fine-tuning)")
                        st.code(sql_base, language="sql")
                        st.caption(f"⏱️ Generation time: {time_base:.3f}s")
                        
                        # Comparison
                        st.markdown("#### 📊 Comparison")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Fine-Tuned Time", f"{time_fine_tuned:.3f}s")
                        with col_b:
                            st.metric("Base Model Time", f"{time_base:.3f}s")
                        
                    except Exception as e:
                        st.error(f"❌ Base model error: {e}")
    



if __name__ == "__main__":
    main()
