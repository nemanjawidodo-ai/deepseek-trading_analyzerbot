import json
import glob
import os
import pandas as pd
from typing import List, Dict, Any

class PortfolioBuilder:
    def __init__(self):
        self.validated_coins = {}
        self.validation_files = [
            'data/validation_results/quick_validation_20251004_2001.json',
            'data/validation_results/standard_validation_20251004_2007.json', 
            'data/validation_results/comprehensive_validation_20251004_2008.json'
        ]
    
    def load_validation_results(self) -> bool:
        """Load validation results dari JSON files"""
        print("📂 Loading validation files...")
        
        all_coins = {}
        
        for file_path in self.validation_files:
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                continue
                
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    print(f"✅ Loaded: {os.path.basename(file_path)}")
                
                # Extract coin data berdasarkan structure yang ada
                coins_data = self.extract_coin_data(data)
                all_coins.update(coins_data)
                
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
        
        self.validated_coins = all_coins
        print(f"📊 Total validated coins: {len(self.validated_coins)}")
        return len(self.validated_coins) > 0
    
    def extract_coin_data(self, data: Dict) -> Dict:
        """Extract coin performance data dari validation results"""
        coins_data = {}
        
        # Handle different JSON structures
        if 'results' in data and 'detailed_results' in data['results']:
            # Structure dari run_validation.py
            for symbol, metrics in data['results']['detailed_results'].items():
                if 'bounce_rate' in metrics:
                    coins_data[symbol] = {
                        'symbol': symbol,
                        'success_rate': metrics['bounce_rate'],
                        'total_tests': metrics.get('total_tests', 0),
                        'successful_bounces': metrics.get('successful_bounces', 0),
                        'support_levels_count': metrics.get('support_levels_count', 0)
                    }
        
        elif 'detailed_results' in data:
            # Alternative structure
            for symbol, metrics in data['detailed_results'].items():
                if 'bounce_rate' in metrics:
                    coins_data[symbol] = {
                        'symbol': symbol,
                        'success_rate': metrics['bounce_rate'],
                        'total_tests': metrics.get('total_tests', 0),
                        'successful_bounces': metrics.get('successful_bounces', 0)
                    }
        
        elif 'overall_success_rate' in data:
            # Simple structure - create mock data
            print("⚠️  Using mock data from summary metrics")
            # We'll handle this case separately
            
        return coins_data
    
    def rank_coins_by_performance(self) -> List[Dict]:
        """Rank coins berdasarkan success rate dan confidence"""
        if not self.validated_coins:
            print("❌ No validation data available")
            return []
        
        ranked = []
        for symbol, metrics in self.validated_coins.items():
            # Calculate confidence score based on sample size
            confidence = min(1.0, metrics.get('total_tests', 0) / 30)
            score = metrics['success_rate'] * 0.7 + confidence * 0.3
            
            ranked.append({
                'symbol': symbol,
                'success_rate': metrics['success_rate'],
                'total_tests': metrics.get('total_tests', 0),
                'confidence': confidence,
                'score': score,
                'tier': self.assign_tier(score)
            })
        
        # Sort by score descending
        ranked.sort(key=lambda x: x['score'], reverse=True)
        return ranked
    
    def assign_tier(self, score: float) -> str:
        """Assign confidence tier berdasarkan score"""
        if score >= 0.65: return 'high_confidence'
        elif score >= 0.55: return 'medium_confidence'
        else: return 'low_confidence'
    
    def build_risk_adjusted_portfolio(self, top_n: int = 20):
        """Build portfolio dengan risk-adjusted allocation"""
        ranked_coins = self.rank_coins_by_performance()
        
        if len(ranked_coins) < top_n:
            print(f"⚠️  Only {len(ranked_coins)} coins available, using all")
            top_n = len(ranked_coins)
        
        # Allocation weights berdasarkan tier
        tier_weights = {
            'high_confidence': 0.05,    # 5% each
            'medium_confidence': 0.03,  # 3% each  
            'low_confidence': 0.02      # 2% each
        }
        
        portfolio = {
            'high_confidence': [],
            'medium_confidence': [], 
            'low_confidence': [],
            'summary': {
                'total_coins': top_n,
                'avg_success_rate': 0,
                'expected_return': 0
            }
        }
        
        total_success = 0
        for coin in ranked_coins[:top_n]:
            coin['allocation'] = tier_weights[coin['tier']]
            portfolio[coin['tier']].append(coin)
            total_success += coin['success_rate']
        
        # Calculate portfolio summary
        if top_n > 0:
            portfolio['summary']['avg_success_rate'] = total_success / top_n
            portfolio['summary']['expected_return'] = (total_success / top_n) * 0.04  # 4% avg bounce
        
        print(f"✅ Built portfolio with {top_n} coins")
        print(f"📈 Average success rate: {portfolio['summary']['avg_success_rate']:.1%}")
        
        return portfolio
    
    def save_portfolio(self, portfolio: Dict, filename: str = "portfolio_allocation.json"):
        """Save portfolio allocation to JSON file"""
        os.makedirs('data/processed', exist_ok=True)
        filepath = f"data/processed/{filename}"
        
        with open(filepath, 'w') as f:
            json.dump(portfolio, f, indent=2)
        
        print(f"💾 Portfolio saved to: {filepath}")