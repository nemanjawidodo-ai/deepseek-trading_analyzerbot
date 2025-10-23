# data/__init__.py
"""
Data Management
Data collection, processing, storage, and validation modules
"""
import sys
from pathlib import Path

# CRITICAL: Setup path untuk consistent imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import with error handling
__all__ = []

# Collectors
try:
    from .collectors.binance import BinanceClient, BinanceDataValidator
    __all__.extend(['BinanceClient', 'BinanceDataValidator'])
except ImportError as e:
    print(f"⚠️  Warning: Could not import collectors: {e}")

# Processors
try:
    from .processors.database_builder import DatabaseBuilder
    __all__.append('DatabaseBuilder')
except ImportError as e:
    print(f"⚠️  Warning: Could not import processors: {e}")

# Validators
try:
    from .validators.historical import HistoricalValidator
    __all__.append('HistoricalValidator')
except ImportError as e:
    print(f"⚠️  Warning: Could not import validators: {e}")

# Storage (jika ada)
try:
    from .storage import *
except ImportError:
    pass