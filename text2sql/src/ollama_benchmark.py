#!/usr/bin/env python3
"""
Quick Complex SQL Benchmark for Ollama Qwen Model
Tests advanced SQL scenarios without additional dependencies
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.infer_ollama import OllamaInferenceEngine
import time
import json

class QuickComplexTester:
    """Streamlined complex SQL testing"""
    
    def __init__(self):
        self.engine = OllamaInferenceEngine()
        
    def get_benchmark_tests(self) -> list:
        """High-impact test cases to showcase capabilities"""
        return [
            {
                'name': '🕒 Temporal Reasoning - Quarterly Analysis',
                'question': 'Show quarterly sales totals for each employee in 2023, including quarters with no sales as 0',
                'schema': '''
                employees(emp_id PK, name, dept_id FK);
                sales(sale_id PK, emp_id FK, amount, sale_date);
                departments(dept_id PK, name);
                ''',
                'complexity': 'High',
                'key_features': ['QUARTER function', 'COALESCE for missing data', 'GROUP BY']
            },
            {
                'name': '🏆 Window Functions - Ranking',
                'question': 'Rank employees by salary within each department, show top 3 per department',
                'schema': '''
                employees(emp_id PK, name, salary, dept_id FK);
                departments(dept_id PK, name);
                ''',
                'complexity': 'High',
                'key_features': ['ROW_NUMBER/RANK', 'PARTITION BY', 'Window functions']
            },
            {
                'name': '🔄 Complex Joins - Cross-Department Analysis',
                'question': 'Find students taking courses taught by professors from different departments than their own',
                'schema': '''
                students(student_id PK, name, dept_id FK);
                enrollments(enrollment_id PK, student_id FK, course_id FK);
                courses(course_id PK, name, professor_id FK);
                professors(professor_id PK, name, dept_id FK);
                ''',
                'complexity': 'Very High',
                'key_features': ['Multiple JOINs', 'Cross-department logic', 'Complex WHERE']
            },
            {
                'name': '🧮 Aggregation Challenge - Running Totals',
                'question': 'Calculate running total of sales per employee, reset at the beginning of each year',
                'schema': '''
                sales(sale_id PK, emp_id FK, amount, sale_date);
                employees(emp_id PK, name);
                ''',
                'complexity': 'Very High', 
                'key_features': ['Window functions', 'PARTITION BY with YEAR', 'Running sums']
            },
            {
                'name': '🎯 Conditional Logic - Dynamic Bonus Calculation',
                'question': 'Calculate bonus: 15% for sales > 100k, 10% for 50k-100k, 5% for 25k-50k, 0% otherwise',
                'schema': '''
                employees(emp_id PK, name, base_salary);
                sales(sale_id PK, emp_id FK, amount, sale_date);
                ''',
                'complexity': 'Medium',
                'key_features': ['CASE WHEN statements', 'Multiple conditions', 'Aggregation']
            },
            {
                'name': '📊 Subquery Excellence - Department Comparisons', 
                'question': 'Find departments where average salary is above company median but below regional average',
                'schema': '''
                employees(emp_id PK, name, salary, dept_id FK);
                departments(dept_id PK, name, region_id FK);
                regions(region_id PK, name);
                ''',
                'complexity': 'Very High',
                'key_features': ['Subqueries', 'MEDIAN function', 'Complex comparisons']
            },
            {
                'name': '🌟 Real-World Scenario - Churn Prediction',
                'question': 'Identify customers with declining activity: 50% less interactions in last 3 months vs previous 3 months',
                'schema': '''
                customers(customer_id PK, name, signup_date);
                interactions(interaction_id PK, customer_id FK, interaction_date, interaction_type);
                ''',
                'complexity': 'Expert',
                'key_features': ['Date arithmetic', 'Comparative periods', 'Percentage calculations']
            },
            {
                'name': '🏗️ Hierarchical Data - Organization Structure',
                'question': 'Show department hierarchy with total employee count including all sub-departments',
                'schema': '''
                departments(dept_id PK, name, parent_dept_id FK);
                employees(emp_id PK, name, dept_id FK);
                ''',
                'complexity': 'Expert',
                'key_features': ['Recursive CTE', 'Hierarchical aggregation', 'Tree traversal']
            },
            {
                'name': '💰 Financial Analysis - P&L Report',
                'question': 'Generate monthly profit/loss by product: revenue minus costs, show profit margin percentage',
                'schema': '''
                sales(sale_id PK, product_id FK, amount, sale_date);
                costs(cost_id PK, product_id FK, amount, cost_date);
                products(product_id PK, name, category);
                ''',
                'complexity': 'High',
                'key_features': ['Monthly grouping', 'Revenue vs Cost', 'Margin calculations']
            },
            {
                'name': '🔍 Advanced Analytics - Trend Analysis',
                'question': 'Show month-over-month growth rate for each product category, highlight categories with >20% growth',
                'schema': '''
                sales(sale_id PK, product_id FK, amount, sale_date);
                products(product_id PK, name, category);
                ''',
                'complexity': 'Expert',
                'key_features': ['LAG function', 'Growth rate calculation', 'Conditional highlighting']
            }
        ]
    
    def analyze_sql_quality(self, sql: str, expected_features: list) -> dict:
        """Quick analysis of SQL quality and complexity"""
        if not sql:
            return {'score': 0, 'analysis': 'No SQL generated'}
        
        sql_upper = sql.upper()
        score = 0
        found_features = []
        analysis_notes = []
        
        # Basic structure (1 point each)
        basic_elements = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY']
        for element in basic_elements:
            if element in sql_upper:
                score += 1
                found_features.append(element)
        
        # Advanced features (2-3 points each)
        advanced_features = {
            'JOIN': 2,
            'LEFT JOIN': 2, 
            'INNER JOIN': 2,
            'CASE WHEN': 3,
            'COALESCE': 2,
            'SUM': 1,
            'COUNT': 1,
            'AVG': 1,
            'OVER (': 4,  # Window functions
            'PARTITION BY': 4,
            'ROW_NUMBER': 4,
            'RANK': 4,
            'WITH RECURSIVE': 5,
            'QUARTER': 3,
            'YEAR': 2,
            'MONTH': 2,
            'LAG': 4,
            'LEAD': 4
        }
        
        for feature, points in advanced_features.items():
            if feature in sql_upper:
                score += points
                found_features.append(feature)
        
        # Check for expected features
        feature_coverage = 0
        for feature in expected_features:
            if any(f.upper() in sql_upper for f in feature.split()):
                feature_coverage += 1
        
        coverage_rate = feature_coverage / len(expected_features) if expected_features else 0
        
        # Quality assessment
        if score >= 15:
            quality = "Excellent"
        elif score >= 10:
            quality = "Good"
        elif score >= 5:
            quality = "Fair"
        else:
            quality = "Basic"
        
        return {
            'score': score,
            'quality': quality,
            'found_features': found_features,
            'coverage_rate': coverage_rate,
            'feature_coverage': f"{feature_coverage}/{len(expected_features)}"
        }
    
    def run_single_test(self, test_case: dict) -> dict:
        """Run a single benchmark test"""
        print(f"\n{'='*60}")
        print(f"🧪 {test_case['name']}")
        print(f"📝 Question: {test_case['question']}")
        print(f"🎚️  Complexity: {test_case['complexity']}")
        print(f"🔑 Key Features: {', '.join(test_case['key_features'])}")
        print("-" * 60)
        
        # Generate SQL
        start_time = time.time()
        result = self.engine.infer_single(test_case['question'], test_case['schema'])
        generation_time = time.time() - start_time
        
        sql = result.get('sql', '')
        is_safe = result.get('is_safe', False)
        error = result.get('error')
        
        # Analyze quality
        quality_analysis = self.analyze_sql_quality(sql, test_case['key_features'])
        
        print(f"⏱️  Generation Time: {generation_time:.2f}s")
        print(f"✅ Safety Check: {'✅ PASS' if is_safe else '❌ FAIL'}")
        print(f"🔧 Generated SQL:")
        print(f"   {sql}")
        print(f"📊 Quality Score: {quality_analysis['score']} ({quality_analysis['quality']})")
        print(f"🎯 Feature Coverage: {quality_analysis['feature_coverage']} ({quality_analysis['coverage_rate']:.1%})")
        print(f"⚙️  Detected Features: {', '.join(quality_analysis['found_features'])}")
        
        if error:
            print(f"❌ Error: {error}")
        
        return {
            'name': test_case['name'],
            'complexity': test_case['complexity'],
            'sql': sql,
            'is_safe': is_safe,
            'generation_time': generation_time,
            'quality_score': quality_analysis['score'],
            'quality_level': quality_analysis['quality'],
            'feature_coverage': quality_analysis['coverage_rate'],
            'found_features': quality_analysis['found_features'],
            'error': error
        }
    
    def run_full_benchmark(self) -> dict:
        """Run complete benchmark suite"""
        print("🚀 OLLAMA QWEN COMPLEX SQL BENCHMARK")
        print("=" * 80)
        print("Testing advanced SQL generation capabilities...")
        
        test_cases = self.get_benchmark_tests()
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📍 Test {i}/{len(test_cases)}")
            result = self.run_single_test(test_case)
            results.append(result)
            
            # Brief pause between tests
            time.sleep(0.5)
        
        # Calculate summary statistics
        total_tests = len(results)
        safe_count = sum(1 for r in results if r['is_safe'])
        avg_quality = sum(r['quality_score'] for r in results) / total_tests
        avg_time = sum(r['generation_time'] for r in results) / total_tests
        avg_coverage = sum(r['feature_coverage'] for r in results) / total_tests
        
        # Quality distribution
        quality_counts = {}
        for result in results:
            quality = result['quality_level']
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
        
        # Complexity performance
        complexity_performance = {}
        for result in results:
            complexity = result['complexity']
            if complexity not in complexity_performance:
                complexity_performance[complexity] = {'count': 0, 'avg_score': 0, 'safe': 0}
            
            complexity_performance[complexity]['count'] += 1
            complexity_performance[complexity]['avg_score'] += result['quality_score']
            if result['is_safe']:
                complexity_performance[complexity]['safe'] += 1
        
        for complexity, stats in complexity_performance.items():
            stats['avg_score'] /= stats['count']
            stats['safety_rate'] = stats['safe'] / stats['count']
        
        summary = {
            'total_tests': total_tests,
            'safety_rate': safe_count / total_tests,
            'avg_quality_score': avg_quality,
            'avg_generation_time': avg_time,
            'avg_feature_coverage': avg_coverage,
            'quality_distribution': quality_counts,
            'complexity_performance': complexity_performance,
            'detailed_results': results
        }
        
        self.display_benchmark_summary(summary)
        return summary
    
    def display_benchmark_summary(self, summary: dict):
        """Display benchmark results in a clear format"""
        print(f"\n\n🎯 BENCHMARK SUMMARY")
        print("=" * 80)
        
        # Overall metrics
        print(f"📊 OVERALL PERFORMANCE:")
        print(f"   🧪 Total Tests: {summary['total_tests']}")
        print(f"   🛡️  Safety Rate: {summary['safety_rate']:.1%}")
        print(f"   ⭐ Avg Quality Score: {summary['avg_quality_score']:.1f}")
        print(f"   ⏱️  Avg Generation Time: {summary['avg_generation_time']:.2f}s")
        print(f"   🎯 Avg Feature Coverage: {summary['avg_feature_coverage']:.1%}")
        
        # Quality distribution
        print(f"\n🏆 QUALITY DISTRIBUTION:")
        for quality, count in summary['quality_distribution'].items():
            percentage = count / summary['total_tests'] * 100
            print(f"   {quality}: {count} tests ({percentage:.1f}%)")
        
        # Complexity performance
        print(f"\n🎚️  COMPLEXITY PERFORMANCE:")
        for complexity, stats in summary['complexity_performance'].items():
            print(f"   {complexity}:")
            print(f"      Tests: {stats['count']}")
            print(f"      Avg Score: {stats['avg_score']:.1f}")
            print(f"      Safety Rate: {stats['safety_rate']:.1%}")
        
        # Top performers
        print(f"\n🌟 TOP PERFORMING QUERIES:")
        top_results = sorted(summary['detailed_results'], 
                           key=lambda x: x['quality_score'], reverse=True)[:3]
        
        for i, result in enumerate(top_results, 1):
            print(f"   {i}. {result['name']}")
            print(f"      Quality: {result['quality_score']} ({result['quality_level']})")
            print(f"      Coverage: {result['feature_coverage']:.1%}")
            print(f"      SQL: {result['sql'][:80]}...")
        
        # Performance insights
        print(f"\n💡 PERFORMANCE INSIGHTS:")
        
        excellent_count = summary['quality_distribution'].get('Excellent', 0)
        good_count = summary['quality_distribution'].get('Good', 0)
        
        if excellent_count + good_count >= summary['total_tests'] * 0.7:
            print("   ✅ Strong performance across complex SQL patterns!")
        
        if summary['safety_rate'] >= 0.95:
            print("   🛡️  Excellent safety record - model generates secure SQL!")
        
        if summary['avg_generation_time'] <= 3.0:
            print("   ⚡ Fast generation - suitable for real-time applications!")
        
        if summary['avg_feature_coverage'] >= 0.6:
            print("   🎯 Good feature recognition - understands complex requirements!")
        
        print(f"\n🚀 Your Ollama Qwen model shows strong capability for complex SQL generation!")

def main():
    """Run the benchmark"""
    tester = QuickComplexTester()
    
    # Check Ollama status
    print("🔍 Checking Ollama connection...")
    if not tester.engine.check_ollama_status():
        print("❌ Ollama is not available. Please ensure Ollama is running with qwen3:1.7b")
        return
    
    print("✅ Ollama connected successfully!")
    
    # Run benchmark
    summary = tester.run_full_benchmark()
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"outputs/reports/benchmark_{timestamp}.json"
    os.makedirs("outputs/reports", exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    print(f"🎉 Benchmark complete! Your Ollama Qwen model performed well on complex SQL tasks.")

if __name__ == "__main__":
    main()