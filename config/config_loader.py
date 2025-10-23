# config/config_loader.py
"""
Centralized Configuration Loader
Load semua YAML config files dengan caching dan environment variable support
"""
import yaml
from pathlib import Path
from typing import Dict, Any
import os
import logging

logger = logging.getLogger(__name__)

class ConfigLoader:
    """Singleton config loader dengan caching"""
    
    _instance = None
    _cache = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance
    
    @staticmethod
    def load(config_name: str) -> Dict[str, Any]:
        """
        Load YAML config file dengan caching
        
        Args:
            config_name: Nama file (tanpa .yaml)
                        Options: 'config', 'strategies', 'validation', 'paths'
        
        Returns:
            Dictionary berisi config
        
        Example:
            >>> config = ConfigLoader.load('strategies')
            >>> print(config['metadata']['name'])
        """
        if config_name in ConfigLoader._cache:
            logger.debug(f"Loading {config_name} from cache")
            return ConfigLoader._cache[config_name]
        
        # Resolve config path
        config_path = Path(__file__).parent / f"{config_name}.yaml"
        
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Replace environment variables ${VAR_NAME}
            config = ConfigLoader._replace_env_vars(config)
            
            # Cache config
            ConfigLoader._cache[config_name] = config
            logger.info(f"✅ Loaded config: {config_name}.yaml")
            
            return config
            
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in {config_name}.yaml: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading {config_name}.yaml: {e}")
            raise
    
    @staticmethod
    def _replace_env_vars(config: Any) -> Any:
        """
        Recursively replace ${VAR_NAME} dengan environment variables
        
        Example:
            telegram_token: ${TELEGRAM_BOT_TOKEN}
            → telegram_token: "actual_token_from_env"
        """
        if isinstance(config, dict):
            return {k: ConfigLoader._replace_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [ConfigLoader._replace_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Check if string is ${VAR_NAME} format
            if config.startswith('${') and config.endswith('}'):
                var_name = config[2:-1]
                env_value = os.getenv(var_name)
                if env_value is None:
                    logger.warning(f"Environment variable {var_name} not set, using placeholder")
                    return config  # Return original ${VAR_NAME}
                return env_value
        return config
    
    @staticmethod
    def get(config_name: str, *keys, default=None):
        """
        Get nested config value dengan dot notation
        
        Args:
            config_name: Nama config file ('strategies', 'config', etc)
            *keys: Nested keys untuk access value
            default: Default value jika key tidak ditemukan
        
        Returns:
            Config value atau default
        
        Examples:
            >>> ConfigLoader.get('strategies', 'entry', 'timeframe')
            '4h'
            
            >>> ConfigLoader.get('config', 'risk', 'max_drawdown')
            0.25
            
            >>> ConfigLoader.get('validation', 'walkforward', 'min_train_period')
            730
        """
        try:
            config = ConfigLoader.load(config_name)
            
            # Navigate nested keys
            for key in keys:
                config = config[key]
            
            return config
            
        except (KeyError, TypeError) as e:
            logger.warning(f"Config key not found: {config_name}.{'.'.join(keys)}")
            return default
    
    @staticmethod
    def reload(config_name: str = None):
        """
        Reload config (hapus cache)
        
        Args:
            config_name: Nama config untuk reload, atau None untuk reload semua
        """
        if config_name:
            ConfigLoader._cache.pop(config_name, None)
            logger.info(f"Reloaded config: {config_name}")
        else:
            ConfigLoader._cache.clear()
            logger.info("Reloaded all configs")

# ========================================
# CONVENIENCE FUNCTIONS (Recommended Usage)
# ========================================

def load_config() -> Dict[str, Any]:
    """
    Load config.yaml (global settings)
    
    Returns:
        Global configuration dictionary
    
    Example:
        >>> config = load_config()
        >>> log_level = config['logging']['level']
        >>> print(log_level)  # 'INFO'
    """
    return ConfigLoader.load('config')

def load_strategies() -> Dict[str, Any]:
    """
    Load strategies.yaml (trading strategy config)
    
    Returns:
        Strategy configuration dictionary
    
    Example:
        >>> strategy = load_strategies()
        >>> timeframe = strategy['entry']['timeframe']
        >>> print(timeframe)  # '4h'
    """
    return ConfigLoader.load('strategies')

def load_validation() -> Dict[str, Any]:
    """
    Load validation.yaml (backtesting config)
    
    Returns:
        Validation configuration dictionary
    
    Example:
        >>> validation = load_validation()
        >>> period = validation['validation']['period_days']
        >>> print(period)  # 1825
    """
    return ConfigLoader.load('validation')

def load_paths() -> Dict[str, Any]:
    """
    Load paths.yaml (file paths config)
    
    Returns:
        Paths configuration dictionary
    
    Example:
        >>> paths = load_paths()
        >>> csv_keywords = paths['csv_detection']['priority_keywords']
        >>> print(csv_keywords)  # ['recap', 'sinyal', 'trading']
    """
    return ConfigLoader.load('paths')

# ========================================
# DOT NOTATION HELPER (Advanced Usage)
# ========================================

def get_config_value(*path, default=None):
    """
    Get nested config value dengan path notation
    
    Args:
        *path: Path dalam format: 'config_file', 'key1', 'key2', ...
        default: Default value jika tidak ditemukan
    
    Returns:
        Config value atau default
    
    Examples:
        >>> # Get strategies -> entry -> timeframe
        >>> timeframe = get_config_value('strategies', 'entry', 'timeframe')
        >>> print(timeframe)  # '4h'
        
        >>> # Get config -> risk -> max_drawdown
        >>> max_dd = get_config_value('config', 'risk', 'max_drawdown')
        >>> print(max_dd)  # 0.25
        
        >>> # Get dengan default value
        >>> timeout = get_config_value('config', 'api', 'timeout', default=30)
        >>> print(timeout)  # 10 (from config) or 30 (default)
    """
    if not path:
        raise ValueError("Path cannot be empty")
    
    config_name = path[0]
    keys = path[1:]
    
    return ConfigLoader.get(config_name, *keys, default=default)