# risk/kill_switch.py
"""
KILL SWITCH MANAGER - Emergency shutdown mechanism
Integrated dengan risk management framework
"""

import redis
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class KillSwitchConfig:
    """Configuration untuk kill switch thresholds"""
    max_daily_loss: float = 0.02  # 2%
    max_drawdown: float = 0.20    # 20%
    max_var_breach: float = 0.03  # 3%
    max_position_loss: float = 0.05  # 5% per position
    data_feed_timeout: int = 600  # 10 minutes
    max_consecutive_losses: int = 5

class KillSwitchManager:
    """
    Manages emergency shutdown mechanism untuk trading bot
    Integrated dengan risk management framework
    """
    
    def __init__(self, config: KillSwitchConfig = None, redis_config: Dict[str, Any] = None):
        self.config = config or KillSwitchConfig()
        self.redis_config = redis_config or {}
        self._init_storage()
        
    def _init_storage(self):
        """Initialize storage backend (Redis atau in-memory)"""
        try:
            self.redis_client = redis.Redis(
                host=self.redis_config.get('host', 'localhost'),
                port=self.redis_config.get('port', 6379),
                db=self.redis_config.get('db', 0),
                password=self.redis_config.get('password'),
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Kill switch using Redis storage")
        except (redis.ConnectionError, redis.AuthenticationError):
            self.redis_client = None
            self.memory_state = {}
            logger.warning("Redis not available, using in-memory kill switch (not persistent)")
    
    def set_kill_switch(self, reason: str, metrics: Dict[str, Any] = None, 
                       severity: str = "HIGH") -> bool:
        """
        Aktifkan kill switch secara permanen sampai manual reset
        
        Args:
            reason: Alasan aktivasi
            metrics: Metrics terkait
            severity: HIGH, CRITICAL, EMERGENCY
            
        Returns:
            bool: True jika berhasil diaktifkan
        """
        kill_state = {
            'active': True,
            'timestamp': datetime.utcnow().isoformat(),
            'reason': reason,
            'severity': severity,
            'metrics': metrics or {},
            'triggered_by': 'risk_manager'
        }
        
        try:
            if self.redis_client:
                self.redis_client.set('risk:kill_switch', json.dumps(kill_state))
                self.redis_client.set('risk:kill_switch:active', 'true')
                # Set expiry untuk auto-reset setelah 24 jam (safety measure)
                self.redis_client.expire('risk:kill_switch', 86400)
                self.redis_client.expire('risk:kill_switch:active', 86400)
            else:
                self.memory_state['kill_switch'] = kill_state
                
            logger.critical(f"🚨 KILL SWITCH ACTIVATED: {reason} (Severity: {severity})")
            
            # Trigger emergency alerts
            self._trigger_emergency_alerts(kill_state)
            return True
            
        except Exception as e:
            logger.error(f"Failed to activate kill switch: {e}")
            return False
    
    def is_kill_switch_active(self) -> bool:
        """Cek status kill switch"""
        try:
            if self.redis_client:
                active = self.redis_client.get('risk:kill_switch:active')
                return active == 'true'
            else:
                return self.memory_state.get('kill_switch', {}).get('active', False)
        except Exception as e:
            logger.error(f"Error checking kill switch status: {e}")
            return True  # Fail-safe: assume active jika error
    
    def get_kill_switch_state(self) -> Dict[str, Any]:
        """Dapatkan detail state kill switch"""
        try:
            if self.redis_client:
                state = self.redis_client.get('risk:kill_switch')
                return json.loads(state) if state else {}
            else:
                return self.memory_state.get('kill_switch', {})
        except Exception as e:
            logger.error(f"Error getting kill switch state: {e}")
            return {}
    
    def reset_kill_switch(self, reason: str, authorized_by: str = "manual") -> bool:
        """
        Reset kill switch (hanya manual intervention)
        
        Args:
            reason: Alasan reset
            authorized_by: Siapa yang authorize reset
            
        Returns:
            bool: True jika berhasil direset
        """
        try:
            if self.redis_client:
                self.redis_client.delete('risk:kill_switch')
                self.redis_client.delete('risk:kill_switch:active')
            else:
                self.memory_state.pop('kill_switch', None)
                
            logger.warning(f"Kill switch reset by {authorized_by}: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset kill switch: {e}")
            return False
    
    def check_emergency_conditions(self, portfolio_metrics: Dict[str, Any], 
                                 position_metrics: Dict[str, Any] = None) -> bool:
        """
        Cek semua kondisi emergency yang memicu kill switch
        
        Args:
            portfolio_metrics: Portfolio-level metrics
            position_metrics: Position-level metrics
            
        Returns:
            bool: True jika kill switch diaktifkan
        """
        triggers = []
        severity = "HIGH"
        
        # Portfolio-level checks
        if portfolio_metrics.get('daily_pnl_pct', 0) < -self.config.max_daily_loss:
            triggers.append(f"Daily loss {portfolio_metrics['daily_pnl_pct']:.2%} > {self.config.max_daily_loss:.1%}")
            severity = "CRITICAL"
        
        if portfolio_metrics.get('drawdown_pct', 0) > self.config.max_drawdown:
            triggers.append(f"Drawdown {portfolio_metrics['drawdown_pct']:.2%} > {self.config.max_drawdown:.1%}")
            severity = "CRITICAL"
        
        if portfolio_metrics.get('var_95', 0) > self.config.max_var_breach:
            triggers.append(f"VaR {portfolio_metrics['var_95']:.2%} > {self.config.max_var_breach:.1%}")
        
        # Data feed checks
        if portfolio_metrics.get('data_feed_stale', False):
            stale_minutes = portfolio_metrics.get('data_stale_minutes', 0)
            if stale_minutes > self.config.data_feed_timeout / 60:
                triggers.append(f"Data feed stale > {self.config.data_feed_timeout/60} minutes")
        
        # Position-level checks
        if position_metrics:
            max_position_loss = position_metrics.get('max_position_loss_pct', 0)
            if max_position_loss > self.config.max_position_loss:
                triggers.append(f"Position loss {max_position_loss:.2%} > {self.config.max_position_loss:.1%}")
            
            consecutive_losses = position_metrics.get('consecutive_losses', 0)
            if consecutive_losses >= self.config.max_consecutive_losses:
                triggers.append(f"Consecutive losses: {consecutive_losses}")
        
        # System health checks
        if portfolio_metrics.get('memory_usage_pct', 0) > 0.9:
            triggers.append(f"High memory usage: {portfolio_metrics['memory_usage_pct']:.1%}")
        
        if triggers:
            return self.set_kill_switch(
                reason=" | ".join(triggers),
                metrics={**portfolio_metrics, **(position_metrics or {})},
                severity=severity
            )
            
        return False
    
    def _trigger_emergency_alerts(self, kill_state: Dict[str, Any]):
        """Trigger emergency alerts melalui monitoring system"""
        try:
            # Format alert message
            message = {
                "type": "kill_switch_activated",
                "timestamp": kill_state['timestamp'],
                "severity": kill_state.get('severity', 'HIGH'),
                "reason": kill_state['reason'],
                "metrics": kill_state['metrics'],
                "action_required": "MANUAL_INTERVENTION"
            }
            
            # TODO: Integrate dengan alerting system (Slack/Telegram/Email)
            # Untuk sekarang, log dan print
            logger.critical(f"EMERGENCY ALERT: {json.dumps(message, indent=2)}")
            
            # Bisa juga trigger system shutdown di sini
            self._initiate_graceful_shutdown()
            
        except Exception as e:
            logger.error(f"Failed to trigger emergency alerts: {e}")
    
    def _initiate_graceful_shutdown(self):
        """Initiate graceful shutdown process"""
        try:
            # Cancel semua open orders
            # Close semua positions
            # Backup state
            # Notify monitoring
            logger.info("Initiating graceful shutdown sequence...")
            
            # Placeholder untuk shutdown logic
            # Akan diintegrasikan dengan execution manager
            
        except Exception as e:
            logger.error(f"Error during graceful shutdown: {e}")