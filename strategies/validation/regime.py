# Buat file: regime_analyzer.py
"""
ANALYSIS PERFORMA DI BULL/BEAR/SIDEWAYS MARKETS
"""
class RegimeAnalyzer:
    def classify_market_regime(self, df):
        """Classify market regime berdasarkan price action"""
        returns = df['close'].pct_change().dropna()
        volatility = returns.rolling(20).std()
        
        # Simple regime classification
        if returns.mean() > 0.001 and volatility.mean() < 0.02:
            return 'BULL_CALM'
        elif returns.mean() > 0.001 and volatility.mean() >= 0.02:
            return 'BULL_VOLATILE'
        elif returns.mean() < -0.001:
            return 'BEAR'
        else:
            return 'SIDEWAYS'
    
    def test_strategy_across_regimes(self, symbols):
        """Test strategy performance across different market conditions"""
        regime_performance = {}
        
        for symbol in symbols:
            # Get extended historical data
            df = self.get_multi_month_data(symbol)
            if df is None:
                continue
                
            # Split data into regimes
            regimes = self.split_data_by_regime(df)
            
            regime_results = {}
            for regime_name, regime_data in regimes.items():
                if len(regime_data) > 10:  # Minimum data points
                    performance = self.test_strategy_on_data(regime_data)
                    regime_results[regime_name] = performance
            
            regime_performance[symbol] = regime_results
        
        return regime_performance