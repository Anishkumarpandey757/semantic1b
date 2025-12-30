#!/usr/bin/env python3
"""
Ollama-based SQL inference for Text2SQL
Uses local Ollama qwen3:1.7b model instead of transformers
"""

import json
import requests
import time
import re
import yaml
from typing import Dict, List, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaInferenceEngine:
    def __init__(self, config_path: str = "configs/sft_qwen.yaml"):
        """Initialize Ollama inference engine with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_name = self.config['model_name']
        self.base_url = self.config['ollama']['base_url']
        self.temperature = self.config['ollama']['temperature']
        self.top_p = self.config['ollama']['top_p']
        self.max_tokens = self.config['ollama']['max_tokens']
        self.stop_sequences = self.config['ollama']['stop_sequences']
        self.timeout = self.config['inference']['timeout']
        self.retry_attempts = self.config['inference']['retry_attempts']
    
    def check_ollama_status(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [model['name'] for model in models]
                if self.model_name in available_models:
                    logger.info(f"✅ Ollama is running, model {self.model_name} is available")
                    return True
                else:
                    logger.error(f"❌ Model {self.model_name} not found. Available: {available_models}")
                    return False
            else:
                logger.error(f"❌ Ollama API returned status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Cannot connect to Ollama at {self.base_url}: {e}")
            return False
    
    def generate_sql(self, prompt: str) -> Optional[str]:
        """Generate SQL using Ollama API"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": self.max_tokens,
                "stop": self.stop_sequences
            }
        }
        
        for attempt in range(self.retry_attempts):
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    sql_output = result.get('response', '').strip()
                    return sql_output
                else:
                    logger.warning(f"Attempt {attempt + 1} failed with status {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(1)  # Wait before retry
        
        logger.error("All retry attempts failed")
        return None
    
    def validate_sql_safety(self, sql: str) -> Tuple[bool, str]:
        """Check if SQL is safe (SELECT-only, basic validation)"""
        if not sql:
            return False, "Empty SQL"
        
        # Remove comments and normalize
        sql_clean = re.sub(r'--.*?\n', '', sql, flags=re.MULTILINE)
        sql_clean = re.sub(r'/\*.*?\*/', '', sql_clean, flags=re.DOTALL)
        sql_clean = sql_clean.strip().upper()
        
        # Must start with SELECT
        if not sql_clean.startswith('SELECT'):
            return False, "Only SELECT queries are allowed"
        
        # Block dangerous keywords
        dangerous_keywords = [
            'DROP', 'DELETE', 'INSERT', 'UPDATE', 'CREATE', 'ALTER', 
            'TRUNCATE', 'EXEC', 'EXECUTE', 'CALL', 'GRANT', 'REVOKE'
        ]
        
        for keyword in dangerous_keywords:
            if keyword in sql_clean:
                return False, f"Dangerous keyword '{keyword}' detected"
        
        # Basic syntax check (very simple)
        if sql_clean.count('(') != sql_clean.count(')'):
            return False, "Unmatched parentheses"
        
        return True, "Valid"
    
    def build_prompt(self, question: str, schema: str, retrieved_context: str = "") -> str:
        """Build the complete prompt for SQL generation"""
        prompt_parts = [
            "[INSTRUCTION] Return only SQL. No explanations.",
        ]
        
        if retrieved_context:
            prompt_parts.extend([
                "[RETRIEVED CONTEXT]",
                retrieved_context,
            ])
        
        prompt_parts.extend([
            "[SCHEMA]",
            schema,
            "[QUESTION]",
            question,
            "[OUTPUT SQL]"
        ])
        
        return "\n".join(prompt_parts)
    
    def infer_single(self, question: str, schema: str, retrieved_context: str = "") -> Dict:
        """Perform inference for a single question"""
        if not self.check_ollama_status():
            return {
                "question": question,
                "sql": None,
                "error": "Ollama not available",
                "is_safe": False,
                "validation_msg": "Service unavailable"
            }
        
        # Build prompt
        prompt = self.build_prompt(question, schema, retrieved_context)
        
        # Generate SQL
        start_time = time.time()
        sql_output = self.generate_sql(prompt)
        generation_time = time.time() - start_time
        
        if sql_output is None:
            return {
                "question": question,
                "sql": None,
                "error": "Generation failed",
                "is_safe": False,
                "validation_msg": "No output",
                "generation_time": generation_time
            }
        
        # Extract just the SQL part (everything after [OUTPUT SQL])
        sql_lines = sql_output.split('\n')
        sql_only = '\n'.join(sql_lines).strip()
        
        # Remove any remaining prompt tokens
        for token in ["[OUTPUT SQL]", "[QUESTION]", "[SCHEMA]", "[INSTRUCTION]"]:
            sql_only = sql_only.replace(token, "").strip()
        
        # Remove thinking tags and other model artifacts
        import re
        sql_only = re.sub(r'<think>.*?</think>', '', sql_only, flags=re.DOTALL | re.IGNORECASE)
        sql_only = re.sub(r'<thinking>.*?</thinking>', '', sql_only, flags=re.DOTALL | re.IGNORECASE)
        sql_only = sql_only.strip()
        
        # Extract just the SQL statement (look for SELECT/WITH/INSERT/UPDATE/DELETE)
        lines = sql_only.split('\n')
        sql_lines = []
        found_sql = False
        for line in lines:
            line_stripped = line.strip().upper()
            if line_stripped.startswith(('SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER')):
                found_sql = True
            if found_sql:
                sql_lines.append(line)
        
        if sql_lines:
            sql_only = '\n'.join(sql_lines).strip()
        
        # Final cleanup - remove any trailing explanations
        sql_only = sql_only.split('\n')[0] if '\n' in sql_only else sql_only
        sql_only = sql_only.strip(' ;') + ';' if sql_only and not sql_only.endswith(';') else sql_only
        
        # Safety validation
        is_safe, validation_msg = self.validate_sql_safety(sql_only)
        
        return {
            "question": question,
            "sql": sql_only,
            "error": None if sql_only else "Empty SQL generated",
            "is_safe": is_safe,
            "validation_msg": validation_msg,
            "generation_time": generation_time,
            "prompt_length": len(prompt),
            "response_length": len(sql_output) if sql_output else 0
        }
    
    def infer_batch(self, examples: List[Dict]) -> List[Dict]:
        """Perform inference on a batch of examples"""
        results = []
        
        for i, example in enumerate(examples):
            logger.info(f"Processing example {i+1}/{len(examples)}")
            
            question = example.get('question', '')
            schema = example.get('schema', '')
            retrieved_context = example.get('retrieved_context', '')
            
            result = self.infer_single(question, schema, retrieved_context)
            result['example_id'] = example.get('id', i)
            results.append(result)
            
            # Small delay to be nice to Ollama
            time.sleep(0.1)
        
        return results

def main():
    """Test the inference engine"""
    # Initialize engine
    engine = OllamaInferenceEngine()
    
    # Test example
    test_question = "What are the names of all teachers in the computer science department?"
    test_schema = """
    Teacher(teacher_id PK, name, department_id FK);
    Department(department_id PK, name);
    """
    
    print("🔍 Testing Ollama SQL Generation...")
    print(f"Question: {test_question}")
    print(f"Schema: {test_schema}")
    print("-" * 50)
    
    result = engine.infer_single(test_question, test_schema)
    
    print("📊 Results:")
    print(f"Generated SQL: {result['sql']}")
    print(f"Is Safe: {result['is_safe']}")
    print(f"Validation: {result['validation_msg']}")
    print(f"Generation Time: {result['generation_time']:.2f}s")
    
    if result['error']:
        print(f"❌ Error: {result['error']}")

if __name__ == "__main__":
    main()
