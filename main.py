#!/usr/bin/env python3
"""
TRADING ANALYZER BOT - MAIN ENTRY POINT
Integrated dengan semua sistem: Risk Management, Strategy, Validation, Execution
"""

import sys
import os
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
import signal
import json
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 🔥 CRITICAL: Setup project root path SEBELUM import apapun
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Setup paths
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def setup_logging():
    """Setup comprehensive logging system"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/trading_bot.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

class TradingAnalyzerBot:
    """
    Main Trading Bot Class - Mengintegrasikan semua komponen
    """
    
    def __init__(self):
        self.logger = setup_logging()
        self.is_running = False
        self.components = {}
        
    async def initialize(self):
        """Initialize semua komponen trading bot"""
        try:
            self.logger.info("🚀 Initializing Trading Analyzer Bot...")
            
            # 1. Load Configuration
            await self._load_configurations()
            
            # 2. Initialize Risk Management System
            await self._initialize_risk_system()
            
            # 3. Initialize Strategy Engine
            await self._initialize_strategy_engine()
            
            # 4. Initialize Data System
            await self._initialize_data_system()
            
            # 5. Initialize Execution System
            await self._initialize_execution_system()
            
            # 6. Initialize Monitoring
            await self._initialize_monitoring()
            
            self.logger.info("✅ All components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Initialization failed: {e}")
            return False
    
    async def _load_configurations(self):
        """Load semua konfigurasi"""
        self.logger.info("📁 Loading configurations...")
        
        try:
            # Load strategy configuration
            from config.config_loader import load_strategies
            STRATEGY_CONFIG = load_strategies()
            
            # Load risk configuration
            from config.config_loader import load_config
            risk_config_data = load_config()
            self.risk_config = risk_config_data.get('risk', {})
            
            # Load exchange configuration
            from config.config_loader import load_config
            exchange_config_data = load_config() 
            self.exchange_config = exchange_config_data.get('exchanges', {})
            
            self.logger.info("✅ Configurations loaded")
            
        except ImportError as e:
            self.logger.warning(f"⚠️ Some configurations missing: {e}")
            # Fallback to default configs
            self._setup_default_configs()
    
    def _setup_default_configs(self):
        """Setup default configurations jika file config tidak ada"""
        self.strategy_config = {
            'support_bounce_v1': {
                'enabled': True,
                'timeframe': '4h',
                'min_volume': 1000000
            }
        }
        self.risk_config = {
            'max_drawdown': 0.25,
            'daily_loss_limit': 0.02,
            'max_portfolio_exposure': 0.5
        }
        self.logger.info("✅ Default configurations applied")
    
    async def _initialize_risk_system(self):
        """Initialize risk management system dengan kill switch"""
        self.logger.info("⚡ Initializing Risk Management System...")
        
        try:
            from risk.risk_manager import RiskManager, RiskManagerConfig
            from risk.kill_switch import KillSwitchConfig
            
            # Setup risk manager
            risk_config = RiskManagerConfig(**self.risk_config)
            kill_config = KillSwitchConfig(**self.kill_switch_config['thresholds'])
            
            self.components['risk_manager'] = RiskManager(
                config=risk_config,
                kill_switch_config=kill_config
            )
            
            self.logger.info("✅ Risk Management System ready")
            
        except Exception as e:
            self.logger.error(f"❌ Risk system initialization failed: {e}")
            raise
    
    async def _initialize_strategy_engine(self):
        """Initialize strategy engine dengan validation"""
        self.logger.info("🎯 Initializing Strategy Engine...")
        
        try:
            # Initialize strategy validator
            from strategies.validation.historical import AdvancedHistoricalValidator
            self.components['strategy_validator'] = AdvancedHistoricalValidator()
            
            # Initialize active strategies
            await self._load_active_strategies()
            
            self.logger.info("✅ Strategy Engine ready")
            
        except Exception as e:
            self.logger.error(f"❌ Strategy engine initialization failed: {e}")
            raise
    
    async def _load_active_strategies(self):
        """Load dan validate active strategies"""
        self.active_strategies = {}
        
        for strategy_name, config in self.strategy_config.items():
            if config.get('enabled', False):
                try:
                    # Load strategy berdasarkan nama
                    strategy = await self._load_strategy(strategy_name, config)
                    if strategy:
                        self.active_strategies[strategy_name] = strategy
                        self.logger.info(f"✅ Strategy loaded: {strategy_name}")
                        
                except Exception as e:
                    self.logger.error(f"❌ Failed to load strategy {strategy_name}: {e}")
    
    async def _load_strategy(self, strategy_name: str, config: Dict) -> Any:
        """Load individual strategy"""
        if strategy_name == 'support_bounce_v1':
            from strategies.support_bounce import SupportBounceStrategy
            return SupportBounceStrategy(config)
        
        # Tambahkan strategy lain di sini
        self.logger.warning(f"⚠️ Unknown strategy: {strategy_name}")
        return None
    
    async def _initialize_data_system(self):
        """Initialize data collection dan processing system"""
        self.logger.info("📊 Initializing Data System...")
        
        try:
            # Initialize data validators
            from data.validators.historical_validator import HistoricalValidator
            self.components['data_validator'] = HistoricalValidator()
            
            # Initialize database builder jika diperlukan
            from data.database_builder import main as build_database
            self.components['db_builder'] = build_database
            
            self.logger.info("✅ Data System ready")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Data system partial initialization: {e}")
    
    async def _initialize_execution_system(self):
        """Initialize execution system"""
        self.logger.info("⚡ Initializing Execution System...")
        
        try:
            # Initialize exchange interface
            from execution.exchange_interface import ExchangeInterface
            self.components['exchange_interface'] = ExchangeInterface(
                self.exchange_config.get('binance', {})
            )
            
            # Initialize order manager
            from execution.order_manager import OrderManager
            self.components['order_manager'] = OrderManager()
            
            self.logger.info("✅ Execution System ready")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Execution system partial initialization: {e}")
    
    async def _initialize_monitoring(self):
        """Initialize monitoring dan alert system"""
        self.logger.info("📈 Initializing Monitoring System...")
        
        try:
            from monitoring.metrics import MetricsCollector
            from monitoring.alerts import AlertManager
            
            self.components['metrics'] = MetricsCollector()
            self.components['alerts'] = AlertManager()
            
            self.logger.info("✅ Monitoring System ready")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Monitoring system partial initialization: {e}")
    
    async def run_validation_mode(self):
        """Run dalam mode validation saja"""
        self.logger.info("🔍 Starting Validation Mode...")
        
        try:
            # Run comprehensive strategy validation
            validator = self.components.get('strategy_validator')
            if validator:
                # TODO: Load sample data untuk validation
                validation_results = await self._run_strategy_validation(validator)
                return validation_results
            else:
                self.logger.error("❌ Strategy validator not available")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Validation mode failed: {e}")
            return None
    
    async def _run_strategy_validation(self, validator):
        """Run comprehensive strategy validation"""
        self.logger.info("📋 Running strategy validation...")
        
        # TODO: Implement actual validation dengan historical data
        # Untuk sekarang return sample results
        return {
            'validation_timestamp': '2024-01-01T00:00:00Z',
            'strategies_tested': list(self.active_strategies.keys()),
            'overall_score': 0.85,
            'status': 'PASSED'
        }
    
    async def run_live_mode(self):
        """Run dalam live trading mode"""
        if self.components['risk_manager'].kill_switch.is_kill_switch_active():
            self.logger.critical("🚨 Cannot start live mode - Kill Switch ACTIVE!")
            return False
        
        self.logger.info("🎯 Starting Live Trading Mode...")
        self.is_running = True
        
        try:
            # Main trading loop
            while self.is_running:
                await self._trading_cycle()
                await asyncio.sleep(60)  # Check setiap 1 menit
                
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Live trading failed: {e}")
            return False
    
    async def _trading_cycle(self):
        """Single trading cycle execution"""
        try:
            # 1. Check risk limits
            risk_ok = await self._check_risk_limits()
            if not risk_ok:
                self.logger.warning("⏸️ Trading paused due to risk limits")
                return
            
            # 2. Generate signals dari semua active strategies
            signals = await self._generate_signals()
            if not signals:
                return
            
            # 3. Validate dan execute signals
            executed_trades = await self._execute_signals(signals)
            
            # 4. Update monitoring
            await self._update_monitoring(executed_trades)
            
        except Exception as e:
            self.logger.error(f"❌ Trading cycle error: {e}")
    
    async def _check_risk_limits(self) -> bool:
        """Check semua risk limits"""
        risk_manager = self.components['risk_manager']
        
        # Get current portfolio state
        portfolio_state = await self._get_portfolio_state()
        
        # Check risk limits
        return risk_manager.check_portfolio_risk(
            portfolio_state.get('metrics', {}),
            portfolio_state.get('positions', [])
        )
    
    async def _generate_signals(self) -> Dict[str, Any]:
        """Generate trading signals dari semua strategies"""
        signals = {}
        
        for strategy_name, strategy in self.active_strategies.items():
            try:
                # TODO: Get market data untuk strategy
                market_data = await self._get_market_data(strategy)
                
                # Generate signals
                strategy_signals = strategy.generate_signals(market_data)
                if strategy_signals:
                    signals[strategy_name] = strategy_signals
                    
            except Exception as e:
                self.logger.error(f"❌ Signal generation failed for {strategy_name}: {e}")
        
        return signals
    
    async def _execute_signals(self, signals: Dict[str, Any]) -> List[Dict]:
        """Validate dan execute trading signals"""
        executed_trades = []
        
        for strategy_name, signal_data in signals.items():
            try:
                # Validate signal dengan risk manager
                validation_result = self.components['risk_manager'].validate_trade(
                    trade_data=signal_data,
                    portfolio_metrics=await self._get_portfolio_metrics(),
                    current_positions=await self._get_current_positions()
                )
                
                if validation_result['approved']:
                    # Execute trade
                    trade_result = await self._execute_trade(signal_data, validation_result)
                    if trade_result:
                        executed_trades.append(trade_result)
                        
            except Exception as e:
                self.logger.error(f"❌ Trade execution failed for {strategy_name}: {e}")
        
        return executed_trades
    
    async def _execute_trade(self, signal_data: Dict, validation_result: Dict) -> Optional[Dict]:
        """Execute individual trade"""
        try:
            order_manager = self.components.get('order_manager')
            if order_manager:
                return await order_manager.execute_order(
                    signal_data, 
                    validation_result
                )
        except Exception as e:
            self.logger.error(f"❌ Trade execution error: {e}")
        
        return None
    
    async def _get_portfolio_state(self) -> Dict[str, Any]:
        """Get current portfolio state"""
        # TODO: Implement actual portfolio state retrieval
        return {
            'metrics': {
                'current_value': 10000,
                'daily_pnl_pct': 0.001,
                'drawdown_pct': 0.05,
                'var_95': 0.02
            },
            'positions': []
        }
    
    async def _get_portfolio_metrics(self) -> Dict[str, Any]:
        """Get portfolio metrics untuk risk management"""
        state = await self._get_portfolio_state()
        return state['metrics']
    
    async def _get_current_positions(self) -> List[Dict]:
        """Get current positions"""
        state = await self._get_portfolio_state()
        return state['positions']
    
    async def _get_market_data(self, strategy) -> Any:
        """Get market data untuk strategy"""
        # TODO: Implement actual market data retrieval
        return None
    
    async def _update_monitoring(self, trades: List[Dict]):
        """Update monitoring system dengan trade results"""
        try:
            metrics = self.components.get('metrics')
            if metrics and trades:
                await metrics.record_trades(trades)
                
        except Exception as e:
            self.logger.error(f"❌ Monitoring update failed: {e}")
    
    async def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("🛑 Shutting down Trading Bot...")
        self.is_running = False
        
        # Cancel pending orders
        try:
            order_manager = self.components.get('order_manager')
            if order_manager:
                await order_manager.cancel_all_orders()
        except Exception as e:
            self.logger.error(f"❌ Error during shutdown: {e}")
        
        self.logger.info("✅ Trading Bot shutdown complete")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print(f"\n⚠️ Received signal {signum}, shutting down...")
    sys.exit(0)

async def main():
    """Main execution function"""
    bot = TradingAnalyzerBot()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize bot
        if not await bot.initialize():
            print("❌ Bot initialization failed!")
            return 1
        
        print("\n" + "="*60)
        print("🎯 TRADING ANALYZER BOT - READY")
        print("="*60)
        print("1. Validation Mode - Test strategies dengan historical data")
        print("2. Live Mode - Run live trading (RISKY!)")
        print("3. Risk Dashboard - View current risk status")
        print("4. Exit")
        print("="*60)
        
        while True:
            choice = input("\nSelect mode (1-4): ").strip()
            
            if choice == "1":
                results = await bot.run_validation_mode()
                if results:
                    print(f"✅ Validation Results: {json.dumps(results, indent=2)}")
                
            elif choice == "2":
                confirm = input("⚠️  LIVE TRADING - ARE YOU SURE? (yes/no): ")
                if confirm.lower() == 'yes':
                    await bot.run_live_mode()
                else:
                    print("Live trading cancelled")
                    
            elif choice == "3":
                await show_risk_dashboard(bot)
                
            elif choice == "4":
                print("👋 Exiting...")
                break
            else:
                print("❌ Invalid choice")
        
        await bot.shutdown()
        return 0
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1

async def show_risk_dashboard(bot):
    """Show risk management dashboard"""
    risk_manager = bot.components.get('risk_manager')
    if risk_manager:
        dashboard = risk_manager.get_risk_dashboard(
            await bot._get_portfolio_metrics(),
            await bot._get_current_positions()
        )
        print("\n📊 RISK DASHBOARD")
        print("="*50)
        print(f"Kill Switch Active: {dashboard['kill_switch']['active']}")
        print(f"Current Exposure: {dashboard['current_metrics']['current_exposure']:.1%}")
        print(f"Drawdown: {dashboard['current_metrics']['drawdown']:.1%}")
        print(f"Daily P&L: {dashboard['current_metrics']['daily_pnl']:.2%}")
        
        if dashboard['warnings']:
            print("\n⚠️ WARNINGS:")
            for warning in dashboard['warnings']:
                print(f"  - {warning}")
    else:
        print("❌ Risk manager not available")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Run main async function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)