#!/usr/bin/env python3
"""
TEST HistoricalValidator sebelum run full validation
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.historical_validator import HistoricalValidator
from binance_client import BinanceClient

def test_validator():
    print("🧪 TESTING HISTORICAL VALIDATOR")
    
    # Test 1: Check if class can be instantiated
    try:
        validator = HistoricalValidator()
        print("✅ HistoricalValidator instantiated successfully")
    except Exception as e:
        print(f"❌ Failed to instantiate: {e}")
        return False
    
    # Test 2: Check if method exists
    if hasattr(validator, 'validate_support_levels'):
        print("✅ validate_support_levels method exists")
    else:
        print("❌ validate_support_levels method missing")
        return False
    
    # Test 3: Test with real data
    try:
        client = BinanceClient()
        df = client.get_klines('BTCUSDT', interval='4h', limit=50)
        
        if df is not None and len(df) > 0:
            result = validator.validate_support_levels('BTCUSDT', df)
            print(f"✅ Test with BTCUSDT successful: {result}")
        else:
            print("❌ No data from Binance")
            
    except Exception as e:
        print(f"❌ Test with real data failed: {e}")
        return False
    
    print("🎉 ALL TESTS PASSED!")
    return True

if __name__ == "__main__":
    test_validator()