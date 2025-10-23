# risk/position_sizer.py
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class PositionSizer:
    """
    Enhanced Position Sizer dengan semua fitur dari kedua file
    Menggabungkan: 
    - src/position_sizer.py (basic sizing)
    - extended_validation/phase3_portfolio/position_sizer.py (advanced risk-adjusted sizing)
    """
    
    def __init__(self, portfolio_value: float = 10000.0, risk_params: Dict = None):
        self.portfolio_value = portfolio_value
        self.risk_params = risk_params or {}
        
        # Tier allocations yang konservatif (dari kedua file)
        self.tier_allocations = {
            'high_confidence': 0.02,    # 2% per trade
            'medium_confidence': 0.0125, # 1.25% per trade  
            'low_confidence': 0.01       # 1% per trade
        }
        
        # Risk limits (dari kedua file)
        self.max_portfolio_risk = 0.20  # 20% max portfolio exposure
        self.min_position_size = 10.0   # $10 minimum (dari file kedua)
        self.max_position_size = 0.05   # 5% maximum (hard cap)
        
        # Additional parameters dari file kedua
        self.min_position_size_pct = 0.005  # 0.5% minimum (dari file pertama)
    
    def calculate_position_size(self, coin: Dict, volatility_adjustment: float = 1.0) -> Tuple[float, float]:
        """
        Calculate position size berdasarkan confidence tier dan risk parameters
        Menggabungkan logika dari kedua file
        """
        # Dapatkan tier dari coin data (dari file kedua)
        tier = coin.get('tier', 'low_confidence')
        
        # Base allocation berdasarkan tier
        base_allocation = self.tier_allocations.get(tier, 0.01)
        
        # Adjust for volatility (dari file pertama)
        adjusted_size_pct = base_allocation * volatility_adjustment
        adjusted_size_pct = min(adjusted_size_pct, self.max_position_size)
        adjusted_size_pct = max(adjusted_size_pct, self.min_position_size_pct)
        
        # Calculate position value
        position_value = self.portfolio_value * adjusted_size_pct
        
        # Ensure minimum position size (dari file kedua)
        position_value = max(position_value, self.min_position_size)
        
        logger.info(
            f"Position sizing: {coin.get('symbol', 'Unknown')} "
            f"({tier}) -> ${position_value:.2f} ({adjusted_size_pct:.3%}) "
            f"(vol_adj: {volatility_adjustment:.3f})"
        )
        
        return position_value, adjusted_size_pct
    
    def calculate_position_size_legacy(self, signal_confidence: str, volatility_adjustment: float = 1.0) -> Tuple[float, float]:
        """
        Legacy method untuk kompatibilitas dengan code yang menggunakan approach lama
        """
        base_allocation = self.tier_allocations.get(signal_confidence, 0.01)
        
        # Adjust for volatility
        adjusted_size = base_allocation * volatility_adjustment
        adjusted_size = min(adjusted_size, self.max_position_size)
        adjusted_size = max(adjusted_size, self.min_position_size_pct)
        
        # Final safety check
        position_value = self.portfolio_value * adjusted_size
        
        logger.info(
            f"Position sizing: {signal_confidence} -> {adjusted_size:.3%} "
            f"(vol_adj: {volatility_adjustment:.3f})"
        )
        
        return position_value, adjusted_size
    
    def validate_portfolio_risk(self, portfolio: Dict, portfolio_value: float = None) -> bool:
        """
        Validate total portfolio risk within limits (dari file kedua)
        """
        if portfolio_value is None:
            portfolio_value = self.portfolio_value
            
        total_exposure = 0.0
        
        for tier in ['high_confidence', 'medium_confidence', 'low_confidence']:
            coins = portfolio.get(tier, [])
            for coin in coins:
                position_size, _ = self.calculate_position_size(coin)
                total_exposure += position_size
        
        exposure_pct = total_exposure / portfolio_value
        is_acceptable = exposure_pct <= self.max_portfolio_risk
        
        logger.info(f"Portfolio Risk Check:")
        logger.info(f"  Total Exposure: ${total_exposure:.2f} ({exposure_pct:.1%})")
        logger.info(f"  Max Allowed: {self.max_portfolio_risk:.1%}")
        logger.info(f"  Status: {'✅ ACCEPTABLE' if is_acceptable else '❌ TOO HIGH'}")
        
        return is_acceptable
    
    def update_portfolio_value(self, new_portfolio_value: float):
        """Update portfolio value untuk perhitungan real-time"""
        self.portfolio_value = new_portfolio_value
        logger.info(f"Updated portfolio value: ${new_portfolio_value:,.2f}")