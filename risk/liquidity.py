# Buat file: liquidity_analyzer.py
"""
ANALYSIS LIQUIDITY DAN SLIPPAGE IMPACT
"""
class LiquidityAnalyzer:
    def __init__(self):
        self.position_sizes = [100, 500, 1000, 5000]  # USD
        
    def estimate_slippage_impact(self, symbols):
        """Estimate slippage untuk berbagai position sizes"""
        slippage_results = {}
        
        for symbol in symbols:
            symbol_data = self.get_order_book_data(symbol)
            if not symbol_data:
                continue
                
            slippage_by_size = {}
            for size in self.position_sizes:
                slippage = self.calculate_expected_slippage(symbol_data, size)
                slippage_by_size[size] = slippage
            
            slippage_results[symbol] = {
                'avg_slippage_bps': self.calculate_avg_slippage(slippage_by_size),
                'liquidity_score': self.calculate_liquidity_score(symbol_data),
                'max_recommended_size': self.get_max_recommended_size(slippage_by_size)
            }
        
        return slippage_results