# backtesting/__init__.py
"""
Backtesting Engine
Strategy backtesting, performance analysis, and optimization
"""

from .engine import BacktestingEngine
from .performance import PerformanceAnalyzer
from .optimizer import StrategyOptimizer

__all__ = [
    'BacktestingEngine',
    'PerformanceAnalyzer',
    'StrategyOptimizer'
]