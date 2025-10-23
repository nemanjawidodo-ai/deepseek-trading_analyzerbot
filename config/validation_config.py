# File: trading_analyzerbot/src/phase2_settings.py

VALIDATION_CONFIG = {
    'timeframe': '4h',
    'period_days': 1825,  # Diubah dari 365 ke 1825 (5 tahun)
    'initial_balance': 10000,
    'base_commission': 0.001,
    'slippage_model': 'proportional',
    'min_trade_threshold': 30,  # Minimum trades untuk statistical significance
    
    # Risk Limits
    'max_drawdown_limit': 0.25,
    'var_confidence_level': 0.95,
    'correlation_threshold': 0.7

    # Statistical Significance
    'confidence_level': 0.95,
    'min_sample_size': 100,
    'monte_carlo_simulations': 10000,

    # Walk-Forward Analysis
    'walkforward': {
        'min_train_period': 730 ,  # 2 tahun training minimum
        'max_train_period': 1095 * 3,  # 3 tahun training maksimum  
        'test_period': 180,  # 6 bulan testing
        'step_size': 90,  # 3 bulan step
        'min_periods_required': 10
    },
    # Performance Benchmarks
    'performance_benchmarks': {
        'min_sharpe_ratio': 1.0,
        'min_profit_factor': 1.3,
        'max_drawdown': 0.15,
        'min_win_rate': 0.45,
        'min_avg_trade': 0.005
    }
}

# Coin prioritization untuk validation
HIGH_PRIORITY_COINS = [
    'ADAUSDT', 'DOTUSDT', 'LINKUSDT', 'MATICUSDT', 'AVAXUSDT',
    'ATOMUSDT', 'ALGOUSDT', 'NEARUSDT', 'FTMUSDT', 'SANDUSDT'
]