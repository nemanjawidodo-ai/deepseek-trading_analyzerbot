# execution/__init__.py
"""
Trade Execution
Order management, exchange interfaces, and execution algorithms
"""

from .order_manager import OrderManager
from .exchange_interface import ExchangeInterface
from .paper_trading import PaperTrading
from .deployment import DeploymentManager

__all__ = [
    'OrderManager',
    'ExchangeInterface',
    'PaperTrading',
    'DeploymentManager'
]