# Buat file: risk_analyzer.py
"""
QUANTIFICATION RISK METRICS
"""
class RiskAnalyzer:
    def calculate_strategy_risk_metrics(self, strategy_returns):
        """Calculate comprehensive risk metrics"""
        metrics = {
            'max_drawdown': self.calculate_max_drawdown(strategy_returns),
            'volatility': self.calculate_volatility(strategy_returns),
            'sharpe_ratio': self.calculate_sharpe_ratio(strategy_returns),
            'calmar_ratio': self.calculate_calmar_ratio(strategy_returns),
            'var_95': self.calculate_var(strategy_returns, 0.95),
            'cvar_95': self.calculate_cvar(strategy_returns, 0.95)
        }
        
        return metrics
    
    def stress_test_strategy(self, symbols, stress_scenarios):
        """Test strategy performance under stress scenarios"""
        stress_results = {}
        
        for scenario_name, scenario_params in stress_scenarios.items():
            scenario_performance = self.simulate_stress_scenario(
                symbols, scenario_params
            )
            stress_results[scenario_name] = scenario_performance
        
        return stress_results