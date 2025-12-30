#!/usr/bin/env python3
"""
Phase 0: Sanity Check & Environment Preparation
Verifies all required folders, data files, and dependencies
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

# Project paths
PROJECT_ROOT = Path(r"d:\WORKSPACE\PYTHON\SEMANTIC_!B")
TEXT2SQL_DIR = PROJECT_ROOT / "text2sql"
DATA_DIR = TEXT2SQL_DIR / "data"
BIRD_ROOT = PROJECT_ROOT / "minidev" / "MINIDEV"  # Using MINIDEV as primary BIRD source
SPIDER_DIR = PROJECT_ROOT / "spider_data"
OUTPUTS = TEXT2SQL_DIR / "outputs"

def check_folders():
    """Verify all required folders exist"""
    print("🔍 Checking folder structure...")
    
    required_folders = {
        'PROJECT_ROOT': PROJECT_ROOT,
        'TEXT2SQL_DIR': TEXT2SQL_DIR,
        'DATA_DIR': DATA_DIR,
        'BIRD_ROOT': BIRD_ROOT,
        'SPIDER_DIR': SPIDER_DIR,
        'OUTPUTS': OUTPUTS
    }
    
    results = {}
    for name, path in required_folders.items():
        exists = path.exists()
        results[name] = {
            'path': str(path),
            'exists': exists,
            'is_dir': path.is_dir() if exists else False
        }
        
        status = "✅" if exists else "❌"
        print(f"  {status} {name}: {path}")
    
    return results

def check_bird_files():
    """Check BIRD dataset files"""
    print("\n🗄️  Checking BIRD dataset files...")
    
    bird_files = {
        'train_json': BIRD_ROOT / 'mini_dev_sqlite.json',
        'train_tables': BIRD_ROOT / 'dev_tables.json',
        'train_databases': BIRD_ROOT / 'dev_databases'
    }
    
    results = {}
    for name, path in bird_files.items():
        exists = path.exists()
        results[name] = {
            'path': str(path),
            'exists': exists
        }
        
        status = "✅" if exists else "❌"
        print(f"  {status} {name}: {path}")
        
        # Count databases if folder exists
        if name == 'train_databases' and exists:
            db_count = len([d for d in path.iterdir() if d.is_dir()])
            results[name]['database_count'] = db_count
            print(f"      Found {db_count} databases")
    
    return results

def check_scripts():
    """Check required Python scripts exist"""
    print("\n📜 Checking required scripts...")
    
    scripts = [
        'src/infer_ollama.py',
        'src/validate_sql.py',
        'src/build_jsonl.py',
        'src/complex_sql_tester.py',
        'quick_complex_test.py'
    ]
    
    results = {}
    for script in scripts:
        path = TEXT2SQL_DIR / script
        exists = path.exists()
        results[script] = {
            'path': str(path),
            'exists': exists
        }
        
        status = "✅" if exists else "❌"
        print(f"  {status} {script}")
    
    return results

def create_output_folders():
    """Create all required output subfolders"""
    print("\n📁 Creating output folder structure...")
    
    output_folders = [
        OUTPUTS / 'checkpoints',
        OUTPUTS / 'reports',
        OUTPUTS / 'jsonl',
        OUTPUTS / 'artifacts',
        OUTPUTS / 'serving' / 'ollama',
        OUTPUTS / 'reports' / 'feedback'
    ]
    
    results = {}
    for folder in output_folders:
        folder.mkdir(parents=True, exist_ok=True)
        results[str(folder.relative_to(OUTPUTS))] = str(folder)
        print(f"  ✅ {folder.relative_to(TEXT2SQL_DIR)}")
    
    return results

def check_dependencies():
    """Check installed Python packages"""
    print("\n📦 Checking dependencies...")
    
    # Map package names to import names
    package_imports = {
        'requests': 'requests',
        'sqlglot': 'sqlglot',
        'pyyaml': 'yaml',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'torch': 'torch',
        'transformers': 'transformers',
        'peft': 'peft',
        'datasets': 'datasets',
        'accelerate': 'accelerate',
        'bitsandbytes': 'bitsandbytes'
    }
    
    results = {}
    for package, import_name in package_imports.items():
        try:
            # Try importing the package
            __import__(import_name)
            
            # Try to get version
            try:
                result = subprocess.run(
                    [sys.executable, '-m', 'pip', 'show', package],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                version = 'installed'
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if line.startswith('Version:'):
                            version = line.split(':', 1)[1].strip()
                            break
            except:
                version = 'installed'
            
            results[package] = {
                'installed': True,
                'version': version
            }
            
            status = "✅"
            print(f"  {status} {package}: {version}")
            
        except ImportError:
            results[package] = {
                'installed': False,
                'version': 'N/A'
            }
            print(f"  ❌ {package}: Not installed")
        except Exception as e:
            results[package] = {
                'installed': False,
                'error': str(e)
            }
            print(f"  ❌ {package}: Error - {e}")
    
    return results

def check_ollama():
    """Check if Ollama is running"""
    print("\n🤖 Checking Ollama status...")
    
    try:
        import requests
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            
            results = {
                'running': True,
                'models': model_names
            }
            
            print(f"  ✅ Ollama is running")
            print(f"  📋 Available models: {', '.join(model_names)}")
            
            # Check for qwen model
            qwen_models = [m for m in model_names if 'qwen' in m.lower()]
            if qwen_models:
                print(f"  🎯 Qwen models found: {', '.join(qwen_models)}")
            
            return results
        else:
            print(f"  ⚠️  Ollama responded with status {response.status_code}")
            return {'running': False, 'error': f'Status {response.status_code}'}
            
    except Exception as e:
        print(f"  ❌ Ollama not available: {e}")
        return {'running': False, 'error': str(e)}

def generate_sanity_report():
    """Generate comprehensive sanity check report"""
    print("\n" + "="*60)
    print("🔬 GENERATING SANITY REPORT")
    print("="*60)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'project_info': {
            'project_root': str(PROJECT_ROOT),
            'text2sql_dir': str(TEXT2SQL_DIR),
            'python_version': sys.version
        },
        'folders': check_folders(),
        'bird_files': check_bird_files(),
        'scripts': check_scripts(),
        'output_folders': create_output_folders(),
        'dependencies': check_dependencies(),
        'ollama': check_ollama()
    }
    
    # Save JSON report
    json_path = OUTPUTS / 'reports' / '00_sanity.json'
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate text summary
    txt_path = OUTPUTS / 'reports' / '00_sanity.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("SANITY CHECK REPORT\n")
        f.write(f"Generated: {report['timestamp']}\n")
        f.write("="*60 + "\n\n")
        
        f.write("PROJECT PATHS:\n")
        f.write(f"  Root: {report['project_info']['project_root']}\n")
        f.write(f"  Text2SQL: {report['project_info']['text2sql_dir']}\n")
        f.write(f"  Python: {report['project_info']['python_version']}\n\n")
        
        f.write("FOLDER STATUS:\n")
        for name, info in report['folders'].items():
            status = "✅" if info['exists'] else "❌"
            f.write(f"  {status} {name}: {info['path']}\n")
        
        f.write("\nBIRD DATASET FILES:\n")
        for name, info in report['bird_files'].items():
            status = "✅" if info['exists'] else "❌"
            f.write(f"  {status} {name}: {info['path']}\n")
            if 'database_count' in info:
                f.write(f"      Database count: {info['database_count']}\n")
        
        f.write("\nREQUIRED SCRIPTS:\n")
        for name, info in report['scripts'].items():
            status = "✅" if info['exists'] else "❌"
            f.write(f"  {status} {name}\n")
        
        f.write("\nDEPENDENCIES:\n")
        for package, info in report['dependencies'].items():
            status = "✅" if info['installed'] else "❌"
            version = info.get('version', 'N/A')
            f.write(f"  {status} {package}: {version}\n")
        
        f.write("\nOLLAMA STATUS:\n")
        if report['ollama']['running']:
            f.write(f"  ✅ Running\n")
            f.write(f"  Models: {', '.join(report['ollama']['models'])}\n")
        else:
            f.write(f"  ❌ Not running: {report['ollama'].get('error', 'Unknown')}\n")
    
    print(f"\n✅ Sanity report saved:")
    print(f"   JSON: {json_path}")
    print(f"   TXT:  {txt_path}")
    
    # Check for critical failures
    critical_failures = []
    
    if not report['folders']['BIRD_ROOT']['exists']:
        critical_failures.append("BIRD_ROOT folder not found")
    
    if not report['bird_files']['train_json']['exists']:
        critical_failures.append("BIRD training JSON not found")
    
    if not report['ollama']['running']:
        critical_failures.append("Ollama is not running")
    
    missing_deps = [pkg for pkg, info in report['dependencies'].items() 
                   if not info['installed']]
    if missing_deps:
        critical_failures.append(f"Missing dependencies: {', '.join(missing_deps)}")
    
    if critical_failures:
        print("\n⚠️  CRITICAL ISSUES DETECTED:")
        for issue in critical_failures:
            print(f"  ❌ {issue}")
        print("\n⚠️  Please resolve these issues before proceeding.")
        return False
    else:
        print("\n✅ ALL SANITY CHECKS PASSED!")
        print("🚀 Ready to proceed with fine-tuning pipeline.")
        return True

def main():
    """Run sanity checks"""
    print("🔬 TEXT-TO-SQL FINE-TUNING PIPELINE")
    print("Phase 0: Sanity Check & Environment Preparation")
    print("="*60)
    
    success = generate_sanity_report()
    
    if success:
        print("\n✅ Environment is ready for fine-tuning!")
        return 0
    else:
        print("\n❌ Environment check failed. Please fix issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
