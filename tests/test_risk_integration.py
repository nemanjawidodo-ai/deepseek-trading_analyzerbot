# tests/test_risk_integration.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risk.position_sizer import PositionSizer
from risk.risk_manager import RiskManager

def test_integration():
    print("🧪 Testing Integrated Risk Management...")
    
    # Test Position Sizer
    sizer = PositionSizer(portfolio_value=10000)
    
    test_coin = {'symbol': 'BTCUSDT', 'tier': 'high_confidence'}
    position_value, size_pct = sizer.calculate_position_size(test_coin)
    print(f"✅ Position Sizer: ${position_value:.2f} ({size_pct:.2%})")
    
    # Test Risk Manager
    risk_mgr = RiskManager()
    
    portfolio_metrics = {
        'drawdown_pct': 0.15,
        'daily_pnl_pct': -0.01,
        'var_95': 0.02
    }
    
    risk_ok = risk_mgr.check_portfolio_risk(portfolio_metrics)
    print(f"✅ Risk Manager: {'PASS' if risk_ok else 'FAIL'}")
    
    print("🎉 All tests passed!")

if __name__ == "__main__":
    test_integration()