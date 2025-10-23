# Buat file: validation_reporter.py
"""
GENERATE COMPREHENSIVE VALIDATION REPORT
"""
class ValidationReporter:
    def generate_validation_report(self, all_validation_results):
        """Generate comprehensive validation report"""
        report = {
            'executive_summary': self.generate_executive_summary(all_validation_results),
            'walkforward_analysis': all_validation_results['walkforward'],
            'regime_analysis': all_validation_results['regime'],
            'liquidity_analysis': all_validation_results['liquidity'],
            'correlation_analysis': all_validation_results['correlation'],
            'risk_metrics': all_validation_results['risk'],
            'go_no_go_recommendation': self.make_go_no_go_decision(all_validation_results)
        }
        
        self.save_report(report)
        return report