# config/risk_config.py
"""
Risk Management Configuration - Integrated dengan Kill Switch
"""

RISK_MANAGER_CONFIG = {
    # Portfolio Risk Limits
    'max_drawdown': 0.25,  # 25%
    'daily_loss_limit': 0.02,  # 2%
    'var_limit': 0.03,  # 3%
    'max_portfolio_exposure': 0.5,  # 50%
    
    # Position Risk Limits
    'max_position_size': 0.1,  # 10% per position
    'max_correlation': 0.7,  # 70% correlation limit
    'max_concentration': 0.3,  # 30% in single asset
    
    # Kill Switch Settings
    'kill_switch_enabled': True,
    'auto_shutdown': True
}

KILL_SWITCH_CONFIG = {
    'redis': {
        'host': 'localhost',
        'port': 6379,
        'db': 0,
        'password': None
    },
    'thresholds': {
        'max_daily_loss': 0.02,  # 2%
        'max_drawdown': 0.20,  # 20%
        'max_var_breach': 0.03,  # 3%
        'max_position_loss': 0.05,  # 5%
        'data_feed_timeout': 600,  # 10 minutes
        'max_consecutive_losses': 5
    }
}