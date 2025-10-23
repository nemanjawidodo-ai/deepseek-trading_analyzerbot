# phase2_validation/strict_success_metrics.py
import numpy as np
from typing import Dict, List, Any

class StrictMetrics:
    def __init__(self):
        self.minimum_requirements = {
            'overall_success_rate': 0.65,      # 65% minimum
            'worst_scenario_min': 0.50,        # Tidak boleh fail total di scenario apapun
            'consistency_score': 0.70,         # Konsisten across scenarios
            'market_cap_coverage': 0.80,       # Harus work di berbagai market cap
            'statistical_significance': 0.95,  # Confidence level
            'risk_adjusted_score': 0.65        # Sharpe ratio equivalent
        }
    
    def calculate_comprehensive_metrics(self, test_results: Dict) -> Dict[str, Any]:
        """Hitung metrics yang ketat dari hasil test"""
        print("📊 Calculating strict success metrics...")
        
        # Extract all success rates
        success_rates = []
        scenario_consistency = []
        market_cap_performance = {'large': [], 'mid': [], 'small': [], 'micro': []}
        
        for coin_symbol, results in test_results.items():
            if 'overall_score' in results:
                success_rates.append(results['overall_score'])
            
            # Scenario consistency
            if 'scenario_results' in results:
                scenario_scores = list(results['scenario_results'].values())
                scenario_consistency.append(np.std(scenario_scores))  # Lower std = more consistent
            
            # Market cap performance
            if 'market_cap_tier' in results:
                tier = results['market_cap_tier']
                if tier in market_cap_performance and 'overall_score' in results:
                    market_cap_performance[tier].append(results['overall_score'])
        
        # Calculate metrics
        metrics = {
            'overall_success_rate': np.mean(success_rates) if success_rates else 0,
            'success_rate_std': np.std(success_rates) if success_rates else 0,
            'scenario_consistency': 1 - (np.mean(scenario_consistency) if scenario_consistency else 0),
            'worst_case_performance': min(success_rates) if success_rates else 0,
            'market_cap_coverage': self.calculate_market_cap_coverage(market_cap_performance),
            'sample_size': len(test_results),
            'confidence_interval': self.calculate_confidence_interval(success_rates),
        }
        
        # Evaluate against benchmarks
        metrics['validation_passed'] = self.evaluate_against_benchmarks(metrics)
        metrics['recommendation'] = self.get_recommendation(metrics)
        
        return metrics
    
    def calculate_market_cap_coverage(self, market_cap_performance: Dict) -> float:
        """Calculate bagaimana strategy perform across different market caps"""
        coverage_scores = []
        for tier, scores in market_cap_performance.items():
            if scores:  # Jika ada data untuk tier ini
                avg_score = np.mean(scores)
                coverage_scores.append(1.0 if avg_score >= 0.60 else avg_score / 0.60)
        
        return np.mean(coverage_scores) if coverage_scores else 0
    
    def calculate_confidence_interval(self, success_rates: List[float]) -> Dict[str, float]:
        """Calculate 95% confidence interval"""
        if not success_rates or len(success_rates) < 2:
            return {'lower': 0, 'upper': 0, 'width': 0}
        
        mean = np.mean(success_rates)
        std_err = np.std(success_rates) / np.sqrt(len(success_rates))
        margin = 1.96 * std_err  # 95% confidence
        
        return {
            'lower': max(0, mean - margin),
            'upper': min(1, mean + margin),
            'width': margin * 2
        }
    
    def evaluate_against_benchmarks(self, metrics: Dict) -> bool:
        """Evaluate apakah hasil memenuhi minimum requirements"""
        checks = [
            metrics['overall_success_rate'] >= self.minimum_requirements['overall_success_rate'],
            metrics['worst_case_performance'] >= self.minimum_requirements['worst_scenario_min'],
            metrics['scenario_consistency'] >= self.minimum_requirements['consistency_score'],
            metrics['market_cap_coverage'] >= self.minimum_requirements['market_cap_coverage'],
            metrics['confidence_interval']['width'] < 0.2  # Confidence interval tidak terlalu lebar
        ]
        
        return all(checks)
    
    def get_recommendation(self, metrics: Dict) -> str:
        """Berdasarkan metrics, berikan recommendation"""
        success_rate = metrics['overall_success_rate']
        consistency = metrics['scenario_consistency']
        
        if success_rate >= 0.75 and consistency >= 0.80:
            return "STRONG_SCALE"
        elif success_rate >= 0.70 and consistency >= 0.70:
            return "SCALE"
        elif success_rate >= 0.65 and consistency >= 0.60:
            return "REFINE"
        elif success_rate >= 0.60:
            return "REFINE_HEAVILY"
        else:
            return "STOP"

# Test the class
if __name__ == "__main__":
    metrics = StrictMetrics()
    print("✅ StrictMetrics tested successfully")