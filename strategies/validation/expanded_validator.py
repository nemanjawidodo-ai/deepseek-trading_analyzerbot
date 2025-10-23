# expanded_validation.py
import random
import pandas as pd
from datetime import datetime, timedelta   
from config.config_loader import load_strategies
from strategies.base_strategy import BaseStrategy  # Absolute import



class ExpandedValidator:
    def get_random_coins(self, sample_size):
        # Pakai config dari strategy
        min_volume = self.config.SELECTION_CONFIG['min_daily_volume']
        # ... implementasi

class ExpandedValidator:
    def __init__(self):
        self.sample_size = 50  # Default size
        self.validation_criteria = {
            'market_cap_tiers': ['large', 'mid', 'small', 'micro'],
            'volume_threshold': 100000,
            'time_periods': ['last_week', 'last_month', 'volatile_period'],
            'liquidity_requirements': True
        }
    
    def get_random_coins(self, sample_size=50):  # ⬅️ TAMBAH PARAMETER OPTIONAL
        """Ambil N coins random dengan kriteria ketat"""
        self.sample_size = sample_size
        print(f"🔄 Getting {sample_size} random coins...")
        
        # GET REAL COINS FROM BINANCE (gunakan existing code)
        try:
            from binance_client import BinanceClient
            client = BinanceClient()
            all_symbols = client.get_all_symbols()
            
            if not all_symbols:
                print("❌ No symbols from Binance, using mock data")
                return self.get_mock_coins(sample_size)
                
            # Filter USDT pairs only
            usdt_pairs = [s for s in all_symbols if s.endswith('USDT')]
            print(f"📊 Found {len(usdt_pairs)} USDT pairs")
            
            # Take random sample
            if len(usdt_pairs) < sample_size:
                print(f"⚠️  Only {len(usdt_pairs)} available, using all")
                selected_symbols = usdt_pairs
            else:
                selected_symbols = random.sample(usdt_pairs, sample_size)
            
            # Convert to coin objects
            coins = []
            for symbol in selected_symbols:
                coin = {
                    'symbol': symbol,
                    'market_cap_tier': self.assign_market_cap_tier(symbol),
                    'volume_24h': self.get_volume_for_symbol(symbol)
                }
                coins.append(coin)
            
            print(f"✅ Successfully collected {len(coins)} coins")
            return coins
            
        except Exception as e:
            print(f"❌ Error getting real coins: {e}")
            print("🔄 Using mock data for testing...")
            return self.get_mock_coins(sample_size)
    
    def get_mock_coins(self, sample_size):
        """Fallback mock data untuk testing"""
        mock_symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOTUSDT', 
            'LINKUSDT', 'LTCUSDT', 'BCHUSDT', 'XLMUSDT', 'XRPUSDT',
            'EOSUSDT', 'TRXUSDT', 'ETCUSDT', 'XTZUSDT', 'ATOMUSDT',
            'ALGOUSDT', 'ZECUSDT', 'BATUSDT', 'COMPUSDT', 'MKRUSDT',
            'SNXUSDT', 'YFIUSDT', 'AAVEUSDT', 'SUSHIUSDT', 'UNIUSDT',
            'CRVUSDT', 'SANDUSDT', 'MANAUSDT', 'ENJUSDT', 'GALAUSDT',
            'AXSUSDT', 'SLPUSDT', 'CHZUSDT', 'FTMUSDT', 'ONEUSDT',
            'VETUSDT', 'HOTUSDT', 'DOGEUSDT', 'SHIBUSDT', 'MATICUSDT',
            'NEARUSDT', 'FTTUSDT', 'SOLUSDT', 'AVAXUSDT', 'LUNAUSDT',
            'ICPUSDT', 'FILUSDT', 'ARUSDT', 'CELRUSDT', 'RENUSDT'
        ]
        
        coins = []
        for i, symbol in enumerate(mock_symbols[:sample_size]):
            # Assign random market cap tiers
            tiers = ['large', 'mid', 'small', 'micro']
            tier = tiers[i % 4]  # Distribute evenly
            
            coin = {
                'symbol': symbol,
                'market_cap_tier': tier,
                'volume_24h': random.randint(1000000, 50000000)
            }
            coins.append(coin)
        
        return coins
    
    def assign_market_cap_tier(self, symbol):
        """Assign market cap tier berdasarkan symbol"""
        large_caps = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT']
        mid_caps = ['DOTUSDT', 'LINKUSDT', 'LTCUSDT', 'BCHUSDT', 'XLMUSDT']
        
        if symbol in large_caps:
            return 'large'
        elif symbol in mid_caps:
            return 'mid'
        else:
            return random.choice(['small', 'micro'])
    
    def get_volume_for_symbol(self, symbol):
        """Get volume untuk symbol (mock untuk sekarang)"""
        return random.randint(1000000, 50000000)
    
    def stratified_sampling(self, coins):
        """Pastikan representasi semua tier market cap"""
        # Implementation yang sudah ada
        return coins