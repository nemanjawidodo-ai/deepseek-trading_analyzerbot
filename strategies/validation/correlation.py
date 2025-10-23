# Buat file: correlation_analyzer.py
"""
ANALYSIS KORELASI ANTAR COINS
"""
class CorrelationAnalyzer:
    def calculate_portfolio_correlation(self, symbols, period_days=30):
        """Calculate correlation matrix untuk portfolio"""
        price_data = self.get_symbols_price_data(symbols, period_days)
        returns_data = self.calculate_returns(price_data)
        
        correlation_matrix = returns_data.corr()
        
        # Analyze diversification benefits
        analysis = {
            'correlation_matrix': correlation_matrix,
            'avg_correlation': correlation_matrix.mean().mean(),
            'max_correlation': correlation_matrix.max().max(),
            'diversification_score': self.calculate_diversification_score(correlation_matrix),
            'highly_correlated_pairs': self.find_highly_correlated_pairs(correlation_matrix)
        }
        
        return analysis