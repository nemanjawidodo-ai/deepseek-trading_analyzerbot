"""
Base Strategy Class
Abstract base class for all trading strategies
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, List

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.name = self.__class__.__name__
        self.is_active = True
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signals from market data
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Dictionary containing signals and metadata
        """
        pass
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate strategy performance metrics"""
        return {
            'total_signals': 0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0
        }
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters"""
        required_params = self.config.get('required_parameters', [])
        return all(param in self.config for param in required_params)
    
    def __str__(self):
        return f"{self.name}(active={self.is_active})"