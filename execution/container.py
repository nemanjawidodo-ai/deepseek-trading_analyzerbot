# execution/container.py
from dependency_injector import containers, providers

class TradingContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    binance_client = providers.Singleton(BinanceClient)
    risk_manager = providers.Singleton(RiskManager, config=config.risk)