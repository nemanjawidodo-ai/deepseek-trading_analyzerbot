# phase2_validation/validation_orchestrator.py
"""
SIMPLIFIED VALIDATION ORCHESTRATOR - Basic version first
"""
import sys
import os
import json
from datetime import datetime
from enum import Enum

# Import paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from strategies.validation.expanded_validator import ExpandedValidator
from binance_client import BinanceClient
from src.historical_validator import HistoricalValidator

class ValidationMode(Enum):
    QUICK_TEST = "quick"
    STANDARD = "standard" 
    COMPREHENSIVE = "comprehensive"

class ValidationOrchestrator:
    def __init__(self):
        self.validator = ExpandedValidator()
        self.binance_client = BinanceClient()
        self.historical_validator = HistoricalValidator()
    
    def run_validation(self, mode: ValidationMode) -> dict:
        """Simple validation runner"""
        print(f"🚀 RUNNING {mode.value.upper()} VALIDATION")
        print("=" * 50)
        
        # Determine sample size
        sample_sizes = {
            ValidationMode.QUICK_TEST: 10,
            ValidationMode.STANDARD: 20,
            ValidationMode.COMPREHENSIVE: 30  # Reduced for testing
        }
        
        sample_size = sample_sizes[mode]
        
        # PHASE 1: Get coins
        print(f"\n📋 PHASE 1: GETTING {sample_size} COINS")
        coins = self.validator.get_random_coins(sample_size)
        print(f"✅ Got {len(coins)} coins")
        
        # PHASE 2: Test coins
        print(f"\n🔍 PHASE 2: TESTING {len(coins)} COINS")
        results = {}
        success_rates = []
        
        for i, coin in enumerate(coins):
            symbol = coin['symbol']
            print(f"  {i+1}/{len(coins)} Testing {symbol}...")
            
            try:
                # Get historical data
                df = self.binance_client.get_klines(symbol, interval='4h', limit=50)
                
                if df is not None and len(df) > 10:
                    # Validate support levels
                    result = self.historical_validator.validate_support_levels(symbol, df)
                    bounce_rate = result.get('bounce_rate', 0)
                    success_rates.append(bounce_rate)
                    
                    status = "✅" if bounce_rate >= 0.65 else "⚠️" if bounce_rate >= 0.5 else "❌"
                    print(f"     {status} {symbol}: {bounce_rate:.1%}")
                    
                    results[symbol] = result
                else:
                    print(f"     ❌ {symbol}: No data")
                    results[symbol] = {'bounce_rate': 0, 'success': False}
                    
            except Exception as e:
                print(f"     ❌ {symbol}: Error - {e}")
                results[symbol] = {'bounce_rate': 0, 'success': False}
        
        # PHASE 3: Analyze results
        print(f"\n📊 PHASE 3: ANALYZING RESULTS")
        if success_rates:
            overall_success = sum(success_rates) / len(success_rates)
            success_count = len([r for r in success_rates if r >= 0.5])
        else:
            overall_success = 0
            success_count = 0
        
        # Determine recommendation
        if overall_success >= 0.70:
            recommendation = "SCALE"
            validation_passed = True
        elif overall_success >= 0.60:
            recommendation = "REFINE" 
            validation_passed = True
        else:
            recommendation = "STOP"
            validation_passed = False
        
        final_results = {
            'sample_size': len(coins),
            'overall_success_rate': overall_success,
            'successful_coins': success_count,
            'total_tested': len(coins),
            'validation_passed': validation_passed,
            'recommendation': recommendation,
            'mode': mode.value,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        self._save_results(final_results, mode)
        
        return final_results
    
    def _save_results(self, results: dict, mode: ValidationMode):
        """Save results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f"data/validation_results/{mode.value}_validation_{timestamp}.json"
        
        # Create directory if not exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {filename}")