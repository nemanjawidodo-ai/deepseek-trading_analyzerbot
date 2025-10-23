"""
BINANCE CLIENT - REAL MARKET DATA VALIDATION
⚠️ WARNING: Jangan percaya sinyal sebelum divalidasi dengan data real
"""

import os
import pandas as pd
import requests
import time
from pathlib import Path
import logging
from typing import Optional, Dict, List
import json
from datetime import datetime, timedelta
from risk.kill_switch_manager import KillSwitchManager

class BinanceClient:
    def __init__(self, api_key=None, api_secret=None, risk_manager=None):
        self.base_url = "https://api.binance.com/api/v3"
        self.api_key = api_key
        self.api_secret = api_secret
        self.risk_manager = risk_manager
        self.kill_switch = KillSwitchManager()
        self.logger = logging.getLogger(__name__)
    
    def submit_order(self, symbol, side, quantity, order_type='MARKET'):
        """Enhanced order submission dengan kill switch check"""
        
        # PRE-TRADE CHECK: Kill switch active?
        if self.kill_switch.is_kill_switch_active():
            kill_state = self.kill_switch.get_kill_switch_state()
            self.logger.error(f"Order rejected - Kill switch active: {kill_state.get('reason', 'Unknown')}")
            
            # Cancel semua outstanding orders
            self.cancel_all_orders(symbol)
            
            return {
                'status': 'rejected',
                'reason': 'kill_switch_active',
                'kill_switch_state': kill_state
            }
        
        # PRE-TRADE CHECK: Risk manager approval
        portfolio_metrics = self.get_portfolio_metrics()
        if not self.risk_manager.check_portfolio_risk(portfolio_metrics, self.get_positions()):
            self.logger.warning("Order rejected by risk manager")
            return {
                'status': 'rejected', 
                'reason': 'risk_check_failed'
            }
        
        # Jika semua check passed, submit order
        try:
            # Original order submission logic di sini
            order_result = self._submit_order_original(symbol, side, quantity, order_type)
            return order_result
            
        except Exception as e:
            self.logger.error(f"Order submission failed: {str(e)}")
            
            # Jika error kritis, trigger kill switch
            if "balance" in str(e).lower() or "margin" in str(e).lower():
                self.kill_switch.set_kill_switch(
                    reason=f"Critical trading error: {str(e)}",
                    metrics=portfolio_metrics
                )
            
            return {'status': 'error', 'reason': str(e)}
    
    def cancel_all_orders(self, symbol=None):
        """Cancel semua orders saat kill switch activated"""
        # Implementation tergantung exchange API
        self.logger.info(f"Cancelling all orders for {symbol or 'all symbols'}")
        # ... existing cancel logic ...
    
    def get_portfolio_metrics(self):
        """Collect portfolio metrics untuk risk checking"""
        # Implementation untuk mendapatkan real-time metrics
        return {
            'portfolio_value': self.get_current_portfolio_value(),
            'daily_pnl_pct': self.get_daily_pnl_percent(),
            'drawdown_pct': self.get_current_drawdown(),
            'var_95': self.calculate_current_var(),
            'data_feed_stale': self.is_data_feed_stale()
        }
    
    def get_all_symbols(self):
        """Get all trading symbols from Binance"""
        try:
            url = f"{self.base_url}/exchangeInfo"
            response = requests.get(url)
            data = response.json()
            
            symbols = []
            for symbol_info in data['symbols']:
                if symbol_info['status'] == 'TRADING' and symbol_info['quoteAsset'] == 'USDT':
                    symbols.append(symbol_info['symbol'])
            
            return symbols
            
        except Exception as e:
            print(f"❌ Error fetching symbols: {e}")
            return []
    
    def get_klines(self, symbol, interval='1h', limit=100):
        """Get OHLCV data"""
        try:
            url = f"{self.base_url}/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            response = requests.get(url, params=params)
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col])
            
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching klines for {symbol}: {e}")
            return None
# Untuk testing - jika file di-run langsung
if __name__ == "__main__":
    client = BinanceClient()
    symbols = client.get_all_symbols()
    print(f"Found {len(symbols)} USDT pairs")
    print("First 10:", symbols[:10])        
class BinanceDataValidator:
    """Validasi support levels dengan real Binance data"""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.base_url = "https://api.binance.com/api/v3"
        
    def get_historical_klines(self, symbol: str, interval: str = '1d', 
                            limit: int = 500) -> Optional[pd.DataFrame]:
        """Ambil historical data dari Binance dengan cache"""
        # Clean symbol (remove USDT jika ada)
        clean_symbol = symbol.replace('USDT', '') + 'USDT'
        cache_file = self.cache_dir / f"{clean_symbol}_{interval}.csv"
        
        # Try cache first
        if cache_file.exists():
            try:
                df = pd.read_csv(cache_file)
                self.logger.info(f"✅ Loaded cached data for {clean_symbol}")
                return df
            except:
                pass
        
        # Fetch from Binance
        try:
            url = f"{self.base_url}/klines"
            params = {
                'symbol': clean_symbol,
                'interval': interval,
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if not data:
                self.logger.warning(f"❌ No data for {clean_symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df = df.dropna()
            
            # Cache the data
            df.to_csv(cache_file, index=False)
            self.logger.info(f"✅ Fetched & cached {len(df)} bars for {clean_symbol}")
            
            time.sleep(0.1)  # Rate limiting
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Failed to fetch data for {clean_symbol}: {e}")
            return None
    
    def validate_support_level(self, symbol: str, support_price: float, 
                             historical_data: pd.DataFrame) -> Dict:
        """Validasi satu support level dengan data real"""
        if historical_data is None or historical_data.empty:
            return {'error': 'No historical data'}
        
        try:
            # Cari semua instances dimana price touch support level (±2%)
            support_zone_low = support_price * 0.98
            support_zone_high = support_price * 1.02
            
            touches = []
            for i in range(len(historical_data) - 1):
                low_price = historical_data.iloc[i]['low']
                high_price = historical_data.iloc[i]['high']
                
                # Check if price touched support zone
                if low_price <= support_zone_high and high_price >= support_zone_low:
                    touch_data = {
                        'touch_index': i,
                        'touch_date': historical_data.iloc[i]['open_time'],
                        'touch_low': low_price,
                        'touch_high': high_price
                    }
                    touches.append(touch_data)
            
            # Analyze bounces
            successful_bounces = 0
            bounce_details = []
            
            for touch in touches:
                touch_idx = touch['touch_index']
                # Look forward 30 days untuk bounce
                lookforward_end = min(touch_idx + 30, len(historical_data) - 1)
                
                for future_idx in range(touch_idx + 1, lookforward_end + 1):
                    future_high = historical_data.iloc[future_idx]['high']
                    # Consider successful jika price naik 3% dari support
                    if future_high >= support_price * 1.03:
                        successful_bounces += 1
                        bounce_details.append({
                            'touch_date': touch['touch_date'],
                            'bounce_date': historical_data.iloc[future_idx]['open_time'],
                            'bounce_percentage': (future_high / support_price - 1) * 100,
                            'days_to_bounce': future_idx - touch_idx
                        })
                        break
            
            # Calculate metrics
            total_touches = len(touches)
            bounce_rate = successful_bounces / total_touches if total_touches > 0 else 0
            
            return {
                'symbol': symbol,
                'support_price': support_price,
                'total_touches': total_touches,
                'successful_bounces': successful_bounces,
                'bounce_rate': round(bounce_rate, 3),
                'bounce_details': bounce_details,
                'validation_quality': 'GOOD' if total_touches >= 5 else 'LOW_SAMPLES'
            }
            
        except Exception as e:
            return {'error': f'Validation failed: {str(e)}'}