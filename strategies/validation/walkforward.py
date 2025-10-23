# Buat file: walkforward_validator.py
"""
WALK-FORWARD VALIDATION ENGINE
Test strategy di multiple rolling windows
"""
import pandas as pd
from datetime import datetime, timedelta

class WalkForwardValidator:
    def __init__(self):
        self.window_sizes = [7, 14, 30]  # days
        self.step_sizes = [3, 7]  # days
        
    def run_walkforward_analysis(self, symbols, total_period_days=90):
        """
        Test strategy across multiple time periods
        """
        results = {}
        
        for symbol in symbols:
            symbol_results = []
            
            # Get historical data untuk 90 hari
            df = self.get_historical_data(symbol, total_period_days)
            if df is None or len(df) < 30:
                continue
                
            for window in self.window_sizes:
                for step in self.step_sizes:
                    performance = self.test_rolling_window(df, window, step)
                    symbol_results.append({
                        'symbol': symbol,
                        'window_days': window,
                        'step_days': step,
                        'success_rate': performance['success_rate'],
                        'sharpe_ratio': performance['sharpe_ratio']
                    })
            
            results[symbol] = symbol_results
        
        return self.analyze_walkforward_results(results)