# config/__init__.py
"""
Configuration Management
Centralized configuration for trading strategies, risk parameters, and exchange settings
"""
import sys
from pathlib import Path
from .config_loader import (
    ConfigLoader,
    load_config, 
    load_strategies,
    load_validation,
    load_paths
)

# Export utama
__all__ = [
    'ConfigLoader',
    'load_config',
    'load_strategies', 
    'load_validation',
    'load_paths'
]

# Path setup
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from .config_loader import load_paths, load_config
    __all__ = ['load_paths', 'load_config']
except ImportError as e:
    print(f"❌ config: Critical import failed: {e}")
    raise