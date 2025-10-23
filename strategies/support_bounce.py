# strategies/support_bounce.py
"""
SUPPORT BOUNCE STRATEGY - Implementasi edge dari historical validator
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from .base import BaseStrategy
from config.config_loader import load_strategies
class SupportBounceStrategy(BaseStrategy):
    """Strategy berdasarkan support level bounce detection dengan statistical edge"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = "SupportBounceStrategy"
        self.version = "1.0"
        
    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical features untuk strategy"""
        # Support level detection (dari edge code)
        data['support_levels'] = self._detect_support_levels(data)
        data['volume_ma'] = data['volume'].rolling(20).mean()
        data['volume_spike'] = data['volume'] / data['volume_ma']
        data['rsi'] = self._calculate_rsi(data['close'])
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals berdasarkan edge logic"""
        signals = pd.Series(0, index=data.index)
        
        for i in range(2, len(data)):
            if self._is_bounce_signal(data, i):
                signals.iloc[i] = 1  # Buy signal
            elif self._is_exit_signal(data, i):
                signals.iloc[i] = -1  # Sell signal
                
        return signals
    
    def _detect_support_levels(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Detect support levels menggunakan rolling local minima"""
        support_levels = []
        
        for i in range(window, len(data)):
            window_lows = data['low'].iloc[i-window:i]
            local_min = window_lows.min()
            
            # Validasi sebagai support level
            touches = self._count_support_touches(data, local_min, i, window)
            if touches >= self.params['min_touches']:
                support_levels.append(local_min)
            else:
                support_levels.append(np.nan)
                
        # Padding untuk awal data
        support_levels = [np.nan] * window + support_levels
        return pd.Series(support_levels, index=data.index)
    
    def _count_support_touches(self, data: pd.DataFrame, level: float, 
                             current_idx: int, lookback: int) -> int:
        """Count berapa kali price menyentuh support level"""
        touches = 0
        tolerance = self.params['touch_tolerance']
        
        for i in range(current_idx - lookback, current_idx):
            low = data['low'].iloc[i]
            if abs(low - level) / level <= tolerance:
                touches += 1
                
        return touches
    
    def _is_bounce_signal(self, data: pd.DataFrame, idx: int) -> bool:
        """Check apakah kondisi bounce terpenuhi"""
        current = data.iloc[idx]
        prev = data.iloc[idx-1]
        
        # Cek support touch
        if pd.isna(current['support_levels']):
            return False
            
        # Cek bounce dari support
        touch_distance = abs(prev['low'] - current['support_levels']) / current['support_levels']
        bounce_strength = (current['close'] - prev['low']) / prev['low']
        
        # Volume confirmation
        volume_ok = current['volume_spike'] >= self.params['volume_threshold']
        
        # RSI filter (opsional)
        rsi_ok = current['rsi'] < self.params.get('rsi_oversold', 35) if self.params.get('rsi_filter', False) else True
        
        return (touch_distance <= self.params['touch_tolerance'] and 
                bounce_strength >= self.params['bounce_threshold'] and
                volume_ok and rsi_ok)
    
    def _is_exit_signal(self, data: pd.DataFrame, idx: int) -> bool:
        """Check exit conditions"""
        # Implement profit target, stop loss, time-based exit
        # Sesuai dengan parameter di strategies.yaml
        pass