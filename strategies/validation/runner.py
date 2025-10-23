# phase2_validation/validation_runner.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.validation.expanded_validator import ExpandedValidator
from strategies.validation.stress_test import StressTester
from strategies.validation.metrics import StrictMetrics
import json
from datetime import datetime

def main():
    print("🚀 PHASE 2 VALIDATION RUNNER - ORGANIZED")
    print("=" * 50)
    
    # PHASE 1: Preparation
    print("\n📋 PHASE 1: PREPARATION")
    validator = ExpandedValidator()
    coins = validator.get_random_coins(20)  # 20 coins dulu untuk testing
    
    print(f"✅ Selected {len(coins)} coins")
    
    # PHASE 2: Comprehensive Testing
    print("\n🔍 PHASE 2: COMPREHENSIVE TESTING")
    tester = StressTester()
    test_results = tester.run_comprehensive_tests(coins)
    
    # PHASE 3: Strict Analysis
    print("\n📊 PHASE 3: STRICT ANALYSIS")
    metrics_calc = StrictMetrics()
    final_metrics = metrics_calc.calculate_comprehensive_metrics(test_results)
    
    # RESULTS
    print("\n🎯 FINAL VALIDATION RESULTS:")
    print("=" * 50)
    for key, value in final_metrics.items():
        if key not in ['confidence_interval']:
            print(f"   {key}: {value}")
    
    # Save results
    save_results(final_metrics, test_results)
    
    return final_metrics

def save_results(metrics, detailed_results):
    """Save results to JSON"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    filename = f"data/validation_results/validation_{timestamp}.json"
    
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics,
        'detailed_results': detailed_results
    }
    
    os.makedirs('data/validation_results', exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print(f"💾 Results saved to: {filename}")

if __name__ == "__main__":
    results = main()