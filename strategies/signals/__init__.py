# strategies/signals/__init__.py
"""
Trading Signals
Signal generation based on technical indicators and market conditions
"""

from .signal_generator import SignalGenerator
from .signal_processor import SignalProcessor

__all__ = [
    'SignalGenerator',
    'SignalProcessor'
]