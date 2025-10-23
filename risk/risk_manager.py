# risk/risk_manager.py
"""
ENHANCED RISK MANAGER - Integrated dengan Kill Switch & Portfolio Risk
Menggabungkan semua fitur risk management dengan konfigurasi terpusat
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from .kill_switch import KillSwitchManager, KillSwitchConfig

logger = logging.getLogger(__name__)

@dataclass
class RiskManagerConfig:
    """Configuration untuk Risk Manager"""
    # Portfolio Risk Limits
    max_drawdown: float = 0.25  # 25%
    daily_loss_limit: float = 0.02  # 2%
    var_limit: float = 0.03  # 3%
    max_portfolio_exposure: float = 0.5  # 50%
    
    # Position Risk Limits
    max_position_size: float = 0.1  # 10% per position
    max_correlation: float = 0.7  # 70% correlation limit
    max_concentration: float = 0.3  # 30% in single asset
    
    # Kill Switch Configuration
    kill_switch_enabled: bool = True
    auto_shutdown: bool = True

class RiskManager:
    """
    Comprehensive Risk Management System
    Integrated dengan Kill Switch untuk emergency protection
    """
    
    def __init__(self, config: RiskManagerConfig = None, kill_switch_config: KillSwitchConfig = None):
        self.config = config or RiskManagerConfig()
        
        # Initialize Kill Switch Manager
        self.kill_switch = KillSwitchManager(
            config=kill_switch_config or KillSwitchConfig(),
            redis_config=self._get_redis_config()
        )
        
        # Risk state tracking
        self.risk_state = {
            'violations_today': 0,
            'last_violation': None,
            'current_exposure': 0.0,
            'peak_portfolio_value': 0.0
        }
        
        logger.info("Risk Manager initialized with Kill Switch integration")
    
    def _get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration from environment or defaults"""
        # TODO: Integrate dengan config system
        return {
            'host': 'localhost',
            'port': 6379,
            'db': 0
        }
    
    def validate_trade(self, trade_data: Dict[str, Any], 
                      portfolio_metrics: Dict[str, Any],
                      current_positions: List[Dict] = None) -> Dict[str, Any]:
        """
        Comprehensive trade validation dengan risk checks
        
        Returns:
            Dict dengan validation result dan reasons
        """
        validation_result = {
            'approved': False,
            'reasons': [],
            'warnings': [],
            'adjusted_size': None
        }
        
        # 1. Check Kill Switch first (immediate rejection)
        if self.kill_switch.is_kill_switch_active():
            validation_result['reasons'].append("Kill switch active - trading halted")
            logger.warning("Trade rejected: Kill switch active")
            return validation_result
        
        # 2. Portfolio-level risk checks
        portfolio_ok, portfolio_reasons = self._check_portfolio_risk(portfolio_metrics, current_positions)
        if not portfolio_ok:
            validation_result['reasons'].extend(portfolio_reasons)
        
        # 3. Trade-specific risk checks
        trade_ok, trade_reasons = self._check_trade_risk(trade_data, current_positions)
        if not trade_ok:
            validation_result['reasons'].extend(trade_reasons)
        
        # 4. Position sizing validation
        size_ok, size_reasons, adjusted_size = self._validate_position_size(trade_data, current_positions)
        if not size_ok:
            validation_result['reasons'].extend(size_reasons)
        elif adjusted_size:
            validation_result['adjusted_size'] = adjusted_size
            validation_result['warnings'].append(f"Position size adjusted to {adjusted_size:.1%}")
        
        # 5. Final decision
        validation_result['approved'] = portfolio_ok and trade_ok and size_ok
        
        if validation_result['approved']:
            logger.info(f"Trade approved: {trade_data.get('symbol', 'Unknown')}")
        else:
            logger.warning(f"Trade rejected: {validation_result['reasons']}")
            
            # Check if we should trigger kill switch untuk severe violations
            self._check_severe_violations(validation_result['reasons'], portfolio_metrics)
        
        return validation_result
    
    def _check_portfolio_risk(self, portfolio_metrics: Dict[str, Any], 
                            current_positions: List[Dict] = None) -> tuple:
        """Check portfolio-level risk limits"""
        reasons = []
        
        # Update peak portfolio value
        current_value = portfolio_metrics.get('current_value', 0)
        self.risk_state['peak_portfolio_value'] = max(
            self.risk_state['peak_portfolio_value'], 
            current_value
        )
        
        # Drawdown check
        drawdown = portfolio_metrics.get('drawdown_pct', 0)
        if drawdown > self.config.max_drawdown:
            reasons.append(f"Drawdown {drawdown:.2%} > {self.config.max_drawdown:.1%}")
        
        # Daily loss check
        daily_pnl = portfolio_metrics.get('daily_pnl_pct', 0)
        if daily_pnl < -self.config.daily_loss_limit:
            reasons.append(f"Daily loss {abs(daily_pnl):.2%} > {self.config.daily_loss_limit:.1%}")
        
        # VaR check
        var_95 = portfolio_metrics.get('var_95', 0)
        if var_95 > self.config.var_limit:
            reasons.append(f"VaR {var_95:.2%} > {self.config.var_limit:.1%}")
        
        # Portfolio exposure check
        if current_positions:
            exposure = self.calculate_total_exposure(current_positions)
            self.risk_state['current_exposure'] = exposure
            
            if exposure > self.config.max_portfolio_exposure:
                reasons.append(f"Portfolio exposure {exposure:.1%} > {self.config.max_portfolio_exposure:.1%}")
        
        return len(reasons) == 0, reasons
    
    def _check_trade_risk(self, trade_data: Dict[str, Any], 
                         current_positions: List[Dict] = None) -> tuple:
        """Check trade-specific risk limits"""
        reasons = []
        
        symbol = trade_data.get('symbol')
        position_size = trade_data.get('position_size_pct', 0)
        
        # Position size limit
        if position_size > self.config.max_position_size:
            reasons.append(f"Position size {position_size:.1%} > {self.config.max_position_size:.1%}")
        
        # Concentration risk
        if current_positions:
            concentration = self._calculate_concentration(current_positions, symbol)
            if concentration > self.config.max_concentration:
                reasons.append(f"Concentration {concentration:.1%} > {self.config.max_concentration:.1%}")
        
        # Correlation risk (simplified)
        correlation = self._estimate_correlation_risk(trade_data, current_positions)
        if correlation > self.config.max_correlation:
            reasons.append(f"Correlation risk {correlation:.1%} > {self.config.max_correlation:.1%}")
        
        return len(reasons) == 0, reasons
    
    def _validate_position_size(self, trade_data: Dict[str, Any], 
                              current_positions: List[Dict] = None) -> tuple:
        """Validate and potentially adjust position size"""
        original_size = trade_data.get('position_size_pct', 0)
        symbol = trade_data.get('symbol')
        
        # Check against absolute limit
        if original_size > self.config.max_position_size:
            return False, [f"Position size exceeds maximum"], None
        
        # Check available portfolio capacity
        current_exposure = self.risk_state['current_exposure']
        available_capacity = self.config.max_portfolio_exposure - current_exposure
        
        if original_size > available_capacity:
            # Adjust size to fit within portfolio limits
            adjusted_size = min(original_size, available_capacity)
            if adjusted_size > 0:
                return True, [], adjusted_size
            else:
                return False, ["No available portfolio capacity"], None
        
        return True, [], None
    
    def _check_severe_violations(self, reasons: List[str], portfolio_metrics: Dict[str, Any]):
        """Check for severe violations that should trigger kill switch"""
        severe_triggers = []
        
        for reason in reasons:
            if any(severity in reason.lower() for severity in ['drawdown', 'daily loss', 'var']):
                severe_triggers.append(reason)
        
        if severe_triggers and self.config.kill_switch_enabled:
            self.kill_switch.check_emergency_conditions(portfolio_metrics)
    
    def calculate_total_exposure(self, current_positions: List[Dict]) -> float:
        """Calculate total portfolio exposure"""
        if not current_positions:
            return 0.0
        
        total_value = sum(position.get('current_value', 0) for position in current_positions)
        portfolio_value = sum(position.get('portfolio_value', 10000) for position in current_positions[:1])
        
        return total_value / portfolio_value if portfolio_value > 0 else 0.0
    
    def _calculate_concentration(self, current_positions: List[Dict], symbol: str) -> float:
        """Calculate concentration for specific symbol"""
        symbol_value = sum(
            position.get('current_value', 0) 
            for position in current_positions 
            if position.get('symbol') == symbol
        )
        portfolio_value = sum(position.get('portfolio_value', 10000) for position in current_positions[:1])
        
        return symbol_value / portfolio_value if portfolio_value > 0 else 0.0
    
    def _estimate_correlation_risk(self, trade_data: Dict[str, Any], 
                                 current_positions: List[Dict] = None) -> float:
        """Estimate correlation risk (simplified implementation)"""
        # TODO: Implement actual correlation calculation
        # Untuk sekarang, return conservative estimate
        return 0.3
    
    def get_risk_dashboard(self, portfolio_metrics: Dict[str, Any], 
                          current_positions: List[Dict] = None) -> Dict[str, Any]:
        """Generate comprehensive risk dashboard"""
        exposure = self.calculate_total_exposure(current_positions) if current_positions else 0.0
        
        dashboard = {
            'risk_limits': {
                'max_drawdown': self.config.max_drawdown,
                'daily_loss_limit': self.config.daily_loss_limit,
                'var_limit': self.config.var_limit,
                'max_exposure': self.config.max_portfolio_exposure,
                'max_position_size': self.config.max_position_size,
                'max_concentration': self.config.max_concentration,
                'max_correlation': self.config.max_correlation
            },
            'current_metrics': {
                'drawdown': portfolio_metrics.get('drawdown_pct', 0),
                'daily_pnl': portfolio_metrics.get('daily_pnl_pct', 0),
                'var_95': portfolio_metrics.get('var_95', 0),
                'current_exposure': exposure,
                'portfolio_value': portfolio_metrics.get('current_value', 0),
                'peak_value': self.risk_state['peak_portfolio_value']
            },
            'kill_switch': {
                'active': self.kill_switch.is_kill_switch_active(),
                'state': self.kill_switch.get_kill_switch_state() if self.kill_switch.is_kill_switch_active() else None
            },
            'utilization': {
                'drawdown_utilization': portfolio_metrics.get('drawdown_pct', 0) / self.config.max_drawdown,
                'exposure_utilization': exposure / self.config.max_portfolio_exposure,
                'daily_loss_utilization': abs(portfolio_metrics.get('daily_pnl_pct', 0)) / self.config.daily_loss_limit
            }
        }
        
        # Add warnings for high utilization
        warnings = []
        for metric, utilization in dashboard['utilization'].items():
            if utilization > 0.8:
                warnings.append(f"High {metric} utilization: {utilization:.1%}")
        
        dashboard['warnings'] = warnings
        
        return dashboard
    
    def reset_risk_limits(self, new_limits: Dict[str, Any]):
        """Update risk limits dynamically"""
        for key, value in new_limits.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated risk limit: {key} = {value}")
        
        logger.info("Risk limits updated successfully")
    
    def emergency_shutdown(self, reason: str = "Manual emergency shutdown"):
        """Trigger emergency shutdown manually"""
        if self.config.kill_switch_enabled:
            self.kill_switch.set_kill_switch(reason, severity="EMERGENCY")
            return True
        return False
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get current risk summary"""
        return {
            'kill_switch_active': self.kill_switch.is_kill_switch_active(),
            'violations_today': self.risk_state['violations_today'],
            'current_exposure': self.risk_state['current_exposure'],
            'peak_portfolio_value': self.risk_state['peak_portfolio_value'],
            'config': {
                'max_drawdown': self.config.max_drawdown,
                'daily_loss_limit': self.config.daily_loss_limit,
                'max_exposure': self.config.max_portfolio_exposure
            }
        }