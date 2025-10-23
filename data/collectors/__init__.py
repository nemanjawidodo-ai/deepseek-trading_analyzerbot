# data/collectors/__init__.py
"""
Data Collection Module
"""

# Import only when modules are available
try:
    from .binance import BinanceClient, BinanceDataValidator
    __all__ = ['BinanceClient', 'BinanceDataValidator']
except ImportError as e:
    print(f"⚠️  Warning: Could not import data collectors: {e}")
    __all__ = []